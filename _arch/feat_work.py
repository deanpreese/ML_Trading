import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier


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

    #file_loaded = pd.read_csv("data/Fractal_ALL_5M_orig.csv")
    #file_loaded = pd.read_csv("data/CleanReFried_5M_ALL.csv")
    #file_loaded = pd.read_csv('data/buildSeqInd_Lucky13_5M_ALL.csv')
    #file_loaded = pd.read_csv('data/buildSeqInd_Lucky13_F.csv')
    #file_loaded = pd.read_csv('data/Ind_F.csv')
    file_loaded = pd.read_csv('data/Ind_F_1M.csv')
    #file_loaded = pd.read_csv("data/IND_LSTM_ALL.csv")
    #file_loaded = file_loaded.drop(columns=['outputC'])
    file_loaded = file_loaded.drop(columns=['output'])


    feature_columns = list(file_loaded.columns[:-1])
    num_columns = len(file_loaded.axes[1]) 
    input_features =  num_columns -1
    X = file_loaded.iloc[:, 0:input_features]  
    y = file_loaded['outputC'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    df = file_loaded
    num_parts = 10
    target_col = 'outputC'
    #correlation_summary_df = calculate_correlation_summary(df, num_parts, target_col)


    #base_model = XGBRegressor()
    #mod_sel = XGBRegressor()
    #post_sel = XGBRegressor()

    base_model = XGBClassifier()
    mod_sel = XGBClassifier()
    post_sel = XGBClassifier() 

    #base_model = LGBMClassifier()
    #mod_sel = LGBMClassifier()
    #post_sel = LGBMClassifier() 
    
    #base_model = CatBoostClassifier()
    #mod_sel = CatBoostClassifier()
    #post_sel = CatBoostClassifier()
    
    

    select = SelectFromModel(mod_sel, threshold="median")
    select.fit(X_train, y_train)
    X_train_rf = select.transform(X_train)
    X_test_rf = select.transform(X_test)

    base_model.fit(X_train, y_train)
    base_preds = base_model.predict(X_test)
    base_score = base_model.score(X_test, y_test)
    base_mse = mean_squared_error(y_test, base_preds)
    base_rmse = base_mse**.5

    post_sel.fit(X_train_rf, y_train)
    post_sel_preds = post_sel.predict(X_test_rf)
    post_sel_score = post_sel.score(X_test_rf, y_test)
    post_mse = mean_squared_error(y_test, post_sel_preds)
    post_rmse = post_mse**.5

    pi_result = permutation_importance(
        post_sel, X_test_rf, y_test, n_repeats=10, random_state=42, n_jobs=2
    )

    """
    print(" ******************************************")
    print("")
    print(f"Orig Shape {X_train.shape}")
    print(f"Selected Shape {X_train_rf.shape}")
    print("")
    print("Base Model")
    print(f"Score {base_score}  MSE {base_mse}  RMSE {base_rmse}") 
    print("")
    print("Post Sel Model")
    print(f"Score {post_sel_score}  MSE {post_mse}  RMSE {post_rmse}") 
    print("")
    print(f"Selected features: {X.columns[select.get_support()]}")
    print("")
    """
    #print("Correlations")
    #print(correlation_summary_df)
    #correlation_summary_df.to_csv("correlation_summary.csv")

    print("")
    print("Feature Importances")
    feature_importance = post_sel.feature_importances_
    feature_importancex = list(zip(list(X.columns[select.get_support()]), feature_importance))
    sorted_feature_importance = sorted(feature_importancex, key=lambda x: x[1], reverse=True)

    features_list_80 = []
    features_list_60 = []
    weight_total_60 = 0
    weight_total_80 = 0

    for feature, weight in sorted_feature_importance:
        print(f"{feature} {weight}")
        
        if weight_total_80 < 0.8:
            weight_total_80 += weight
            features_list_80.append(feature)
            
        if weight_total_60 < 0.6:
            weight_total_60 += weight
            features_list_60.append(feature)

    """
    print(" ")
    print(f"Number of Features selected {len(X.columns[select.get_support()])}")
    print(f"80% List {len(features_list_80)}")    
    print(features_list_80)    
    print(f"60% List {len(features_list_60)}")    
    print(features_list_60)    
    print(" ")
    """
    print(" ")
    print("Permutation Importance")
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    for ix in pi_result.importances_mean.argsort()[::-1]:
        print(f"{list(file_loaded.columns)[ix]} {pi_result.importances_mean[ix]}  {pi_result.importances_std[ix]}")



if __name__ == "__main__":
    run()






