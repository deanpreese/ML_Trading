from time import time
import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

from boruta import BorutaPy

# Function to calculate correlations over parts and summarize them
def calculate_correlation_summary(df, num_parts, target_col):
    part_size = len(df) // num_parts
    features = df.columns.drop(target_col)
    
    correlation_summary = {
        'Feature': [],
        'Highest Correlation': [],
        'Lowest Correlation': [],
        'Mean Correlation': [],
        'Std Correlation': []
    }
    
    for feature in features:
        correlations = []
        
        for i in range(num_parts):
            part_df = df.iloc[i * part_size:(i + 1) * part_size]
            corr_matrix = part_df.corr()
            corr_value = corr_matrix.at[feature, target_col]
            correlations.append(corr_value)
        
        print(f"Correlations for {feature}: {correlations}")
        
        correlation_summary['Feature'].append(feature)
        correlation_summary['Highest Correlation'].append(max(correlations))
        correlation_summary['Lowest Correlation'].append(min(correlations))
        correlation_summary['Mean Correlation'].append(np.mean(correlations))
        correlation_summary['Std Correlation'].append(np.std(correlations))
    
    return pd.DataFrame(correlation_summary)


# =============================================================================

def run():

    datafile = [ 
        '../data/buildSeqInd_Lucky13_5M_3070.csv',   #0
        '../data/buildSeqInd_Lucky13_5M_ALL.csv',  #1
        '../data/buildSeqInd_Lucky13_F.csv',  #2
        '../data/buildSeqInd_Lucky13_D.csv',  #3
        '../data/buildSeqInd_Lucky13_F_3070.csv',  #4
        '../data/Expanded_Lucky13_3070.csv',  #5
        '../data/alt_ex13.csv', #6
        
        
    ]

    file_loaded = pd.read_csv(datafile[6])
    X = file_loaded
    X = X.drop(columns=['output', 'outputC'])
    y = file_loaded['outputC'].values
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #df = file_loaded
    #num_parts = 10
    #target_col = 'outputC'
    #correlation_summary_df = calculate_correlation_summary(df, num_parts, target_col)

    #base_model = XGBRegressor()
    #mod_sel = XGBRegressor()
    #post_sel = XGBRegressor()
    
    #base_model = XGBClassifier()
    #mod_sel = XGBClassifier()
    #post_sel = XGBClassifier() 

    #base_model = LGBMClassifier()
    #mod_sel = LGBMClassifier()
    #post_sel = LGBMClassifier() 
    
    base_model = CatBoostClassifier()
    mod_sel = CatBoostClassifier()
    post_sel = CatBoostClassifier()
    
    #select1 = SelectFromModel(CatBoostClassifier(), threshold="median")
    #select1.fit(X_train, y_train)
    #X_train_rf1 = select1.transform(X_train)
    #X_test_rf1 = select1.transform(X_test)
    
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=15, verbose=2)
        
    tic_fwd = time()        
    sfs_forward = SequentialFeatureSelector(
        rf, n_features_to_select=15, direction="forward", n_jobs=-1
    ).fit(X, y)
    toc_fwd = time()

    tic_bwd = time()
    sfs_backward = SequentialFeatureSelector(
        rf, n_features_to_select=15, direction="backward", n_jobs=-1
    ).fit(X, y)
    toc_bwd = time()

        
    select = SelectFromModel(rf, threshold="median")
    select.fit(X_train, y_train)
    X_train_rf = select.transform(X_train)
    X_test_rf = select.transform(X_test)

    base_model.fit(X_train, y_train)
    base_preds = base_model.predict(X_test)
    base_score = base_model.score(X_test, y_test)
    base_mse = mean_squared_error(y_test, base_preds)
    base_rmse = base_mse**.5
    base_r2 = r2_score(y_test, base_preds)

    post_sel.fit(X_train_rf, y_train)
    post_sel_preds = post_sel.predict(X_test_rf)
    post_sel_score = post_sel.score(X_test_rf, y_test)
    post_mse = mean_squared_error(y_test, post_sel_preds)
    post_rmse = post_mse**.5
    post_r2 = r2_score(y_test, post_sel_preds)

    pi_result = permutation_importance(
        post_sel, X_test_rf, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    
    #print("Correlations")
    #print(correlation_summary_df)
    #correlation_summary_df.to_csv("correlation_summary.csv")

    print("")
    print("Feature Importances Base")
    feature_importance = base_model.feature_importances_
    feature_importancex = list(zip(list(X.columns[select.get_support()]), feature_importance))
    sorted_feature_importance = sorted(feature_importancex, key=lambda x: x[1], reverse=True)
    for feature, weight in sorted_feature_importance:
        print(f"{feature} {weight}")    
    
    print(" ")
    print("Permutation Importance Base")
    sorted_idx = np.argsort(feature_importance)
    for ix in pi_result.importances_mean.argsort()[::-1]:
        print(f"{list(file_loaded.columns)[ix]} {pi_result.importances_mean[ix]}  {pi_result.importances_std[ix]}")


    print("")
    print("Feature Importances POST")
    feature_importance = post_sel.feature_importances_
    feature_importancex = list(zip(list(X.columns[select.get_support()]), feature_importance))
    sorted_feature_importance = sorted(feature_importancex, key=lambda x: x[1], reverse=True)
    for feature, weight in sorted_feature_importance:
        print(f"{feature} {weight}")    
    
    print(" ")
    print("Permutation Importance POST")
    sorted_idx = np.argsort(feature_importance)
    for ix in pi_result.importances_mean.argsort()[::-1]:
        print(f"{list(file_loaded.columns)[ix]} {pi_result.importances_mean[ix]}  {pi_result.importances_std[ix]}")

  
    print(" ******************************************")
    print("")
    print(f"Orig Shape {X_train.shape}")
    print(f"Selected Shape {X_train_rf.shape}")
    print("")
    print("Base Model")
    print(f"Score {base_score}  MSE {base_mse}  RMSE {base_rmse}  R2 {base_r2}") 
    print("")
    print("Post Sel Model")
    print(f"Score {post_sel_score}  MSE {post_mse}  RMSE {post_rmse}  R2 {post_r2}") 
    print("")
    print(f"Selected features: {X.columns[select.get_support()]}")
    print("")
  
    print(
        "Features selected by forward sequential selection: "
        f"{feature_names[sfs_forward.get_support()]}"
    )
    print(f"Done in {toc_fwd - tic_fwd:.3f}s")
    print(
        "Features selected by backward sequential selection: "
        f"{feature_names[sfs_backward.get_support()]}"
    )
    print(f"Done in {toc_bwd - tic_bwd:.3f}s")        
        

if __name__ == "__main__":
    run()






