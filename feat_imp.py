import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, mean_squared_error
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

from ml_model.model_stats import gen_reg_stats
from ml_model.data_func import simple_split_and_scale


def calc_importances_and_baseline(models, X_train, y_train, X_test, y_test, features):

    importance_df = pd.DataFrame(features, columns=['Feature'])
    baseline_data =[]
    
    # Train models, calculate importances, and store results
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_r2 = r2_score(y_test,y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        if "Reg" in name:
            perf, total, mse, rmse, mae = gen_reg_stats(y_test, y_pred)
            baseline_performance = perf
        else:           
            baseline_performance = accuracy_score(y_test, y_pred)
            
        baseline_data.append({'Model': name, 'Baseline': baseline_performance, 'R2': model_r2, 'MSE':mse })
        print(f"Baseline {baseline_performance}")        
        print(f"Calculating Importances ... ")        
        
        perm_importances = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
        perm_means = perm_importances.importances_mean
        perm_stds = perm_importances.importances_std
        
        # Calculate standard feature importance from the model
        if 'Cat' in name:
            standard_importances = model.get_feature_importance()
        else:
            standard_importances = model.feature_importances_
        
        # Store results in DataFrame
        importance_df[f'{name}_PI_Mean'] = perm_means
        importance_df[f'{name}_PI_Std'] = perm_stds
        importance_df[f'{name}_Std_Imp'] = standard_importances

    # Compute composite importance
    importance_df['Comp_PI_Mean'] = importance_df[
        [f'{name}_PI_Mean' for name in models.keys()]
    ].mean(axis=1)

    importance_df['Comp_Std_Imp'] = importance_df[
        [f'{name}_Std_Imp' for name in models.keys()]
    ].mean(axis=1)

    importance_df_sorted = importance_df.sort_values(by='Comp_Std_Imp', ascending=False)
    
    baseline_df = pd.DataFrame(baseline_data)
    print(baseline_df)
    
    return importance_df_sorted, baseline_df
    

def select_features(importance_df_sorted, threshold_v):    

    # Define thresholds for feature selection
    threshold_mean = np.percentile(importance_df_sorted['Comp_PI_Mean'], threshold_v)  # top 25% permutation importance mean
    threshold_standard = np.percentile(importance_df_sorted['Comp_Std_Imp'], threshold_v)  # top 25% standard importance

    # Select features that are consistently important
    important_features = importance_df_sorted[
        (importance_df_sorted['Comp_PI_Mean'] > threshold_mean) & 
        (importance_df_sorted['Comp_Std_Imp'] > threshold_standard)
    ]['Feature'].tolist()
    return important_features
    
def retrain_models(models, important_features, X_train, X_test, y_train, y_test, baseline_df):    
    
    composite_results = []
    for name, model in models.items():
        X_train_selected = X_train[important_features]
        X_test_selected = X_test[important_features]

        model.fit(X_train_selected, y_train)
        
        # Evaluate the new model
        y_pred_selected = model.predict(X_test_selected)
        sel_mse = mean_squared_error(y_test, y_pred_selected)
        sel_model_r2 = r2_score(y_test,y_pred_selected)
        
        if "Reg" in name:
            perf, total, mse, rmse, mae = gen_reg_stats(y_test, y_pred_selected)
            selected_performance = perf
        else:           
            selected_performance = accuracy_score(y_test, y_pred_selected)
        
        baseline_perf = baseline_df[baseline_df['Model'] == name]['Baseline'].values[0]
        base_r2 = baseline_df[baseline_df['Model'] == name]['R2'].values[0]
        base_mse = baseline_df[baseline_df['Model'] == name]['MSE'].values[0]
        
        composite_results.append({'Model': name, 'Perf': baseline_perf, 'Sel_Perf': selected_performance,  'R2': base_r2, 'Sel_R2': sel_model_r2, 'MSE': base_mse, 'Sel_MSE': sel_mse})
        #print(f'{name} Selected Features Performance: {selected_performance:.4f}')

    # Convert results to DataFrame and plot performance comparison
    performance_df = pd.DataFrame(composite_results) 
    return performance_df   




def gen_results(models, X_train, y_train, X_test, y_test, columns, threshold):
    
    importance_df_sorted, baseline_df = calc_importances_and_baseline(models, X_train, y_train, X_test, y_test, columns)
    important_features = select_features(importance_df_sorted, threshold)    
    performance_df = retrain_models(models, important_features, X_train, X_test, y_train, y_test, baseline_df)   
   
    return importance_df_sorted, important_features, performance_df 
    
    
# --------------
def run():

    datafile = [ 
            'data/NewModel_3070_oos.csv',   
            'data/NewModel_3070.csv',  #1
            'data/NewModel_ALL_oos.csv',   
            'data/NewModel_ALL.csv',  #3
            'data/NewModel_span3_3070_oos.csv',   
            'data/NewModel_span3_3070.csv',  #5
    ]   

    data = pd.read_csv(datafile[5])
    df = data.drop(columns=['TimeTicks','SeqClose'])
    #df = df[((df['RSI'] > 0) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 100)]  
    
    #22 Items
    #comp = ['L1', 'RSI', 'O5', 'ROC', 'ROC14', 'ATR5', 'ATR21', 'TV41', 'RSI14X', 'TV31', 'ATR2', 
    #    'H2', 'ATR14Y1', 'L5', 'L2', 'STOK1', 'ROC1', 'SDBB91', 'ATR14Y', 'ROC141', 'SDKC9', 'SDLR310']
    #X = df[comp]


    span3 = ['RSI', 'ATR5', 'STOK1', 'H1', 'SDBB9', 'ATR2', 'SDBB91', 'TV11', 'ROC141', 
             'ATR21', 'TV31', 'RSI14X', 'SDKC9', 'ROC1', 'SDLR310', 'ROC14', 'HourOfDay', 'TV41', 'ROC', 'SDKC91']
    X = df[span3]

    #X = df
    #X = X.drop(columns=['output', 'outputC'])
    
    y = df['outputC'].values
    y2 = df['output'].values
    
    threshold = 50


    #X_train_c, X_test_c, y_train_c, y_test_c = simple_split_and_scale(X, y, 0.2, 42)
    #X_train_r, X_test_r, y_train_r, y_test_r = simple_split_and_scale(X, y2, 0.2, 42)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y2, test_size=0.2, random_state=42)
    
    models_c = {
        #'RandomForestClassifier' :RandomForestClassifier(random_state=42, verbose=2, n_jobs=-1),
        'LGBCls': LGBMClassifier(random_state=42, verbose=2, n_jobs=-1),
        'XGBCls': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=2),
        'CatCls': CatBoostClassifier(random_state=42, verbose=2)
    }

    models_r = {
        #'RandomForestRegressor' :RandomForestRegressor(random_state=42, verbose=2, n_jobs=-1),
        'LGBReg': LGBMRegressor(random_state=42, verbose=2),
        'XGBReg': XGBRegressor(random_state=42, use_label_encoder=False, verbosity=2),
        'CatReg': CatBoostRegressor(random_state=42, verbose=2)
    }

    use_class = True
    use_reg = True
    if use_class:
        importance_df_sorted_c, important_features_c, performance_df_c  = gen_results(models_c, X_train_c, y_train_c, X_test_c, y_test_c, X.columns, threshold)    

    if use_reg:
        importance_df_sorted_r, important_features_r, performance_df_r  = gen_results(models_r, X_train_r, y_train_r, X_test_r, y_test_r, X.columns, threshold)            

    if use_class:
        print("Classifier Sorted Importance")
        print(" ")
        print(importance_df_sorted_c)
        print(" ")
        print(f"Features Selected based on threshold of {threshold} percent")
        print("Classifier Selected Features")
        print(important_features_c)
        print("Perf Results")
        print(performance_df_c)
        print(" ")

        
    if use_reg:
        print("------------------------------------------------------------")
        print(" ")
        print("Regressor Sorted Importance")
        print(" ")
        print(importance_df_sorted_r)
        print(" ")
        print(f"Features Selected based on threshold of {threshold} percent")
        print("Regressor Selected Features")
        print(important_features_r)
        print("Perf Results")
        print(performance_df_r)
        print(" ")



if __name__ == "__main__":
    #run()

    #'data/NewModel_3070.csv',  #1
    #Features Selected based on threshold of 50 percent
    #Classifier Selected Features
    #a =  ['RSI', 'ATR2', 'RSI14X', 'ATR21', 'ROC14', 'ROC141', 'SDBB91', 'TV31', 'H2', 'ATR14Y1']

    #Regressor Selected Features
    #b = ['RSI', 'ATR2', 'RSI14X', 'STOK1', 'ATR5', 'ROC141', 'SDBB91', 'ATR21', 'TV41', 'ROC14', 'SDKC9', 'ROC', 'ATR14Y']

    #'data/NewModel_ALL.csv',  #3    
    #Features Selected based on threshold of 50 percent
    #Classifier Selected Features
    #c = ['SDLR310', 'TV41', 'TV31', 'SDBB91', 'RSI', 'ROC1', 'L1', 'O5', 'L5', 'ATR21']

    #Regressor Selected Features
    #d = ['L2', 'ATR2', 'ROC141', 'TV41']

    #22 Items
    #comp = ['L1', 'RSI', 'O5', 'ROC', 'ROC14', 'ATR5', 'ATR21', 'TV41', 'RSI14X', 'TV31', 'ATR2', 
    #'H2', 'ATR14Y1', 'L5', 'L2', 'STOK1', 'ROC1', 'SDBB91', 'ATR14Y', 'ROC141', 'SDKC9', 'SDLR310']

    #unique_values = list(set(a + b + c + d))
    #print(len(unique_values))
    #print(unique_values)
    
    
    #Features Selected based on threshold of 50 percent
    #Classifier Selected Features
    #a = ['SDLR310', 'TV41', 'RSI', 'ROC1', 'ATR21']
    #Regressor Selected Features
    #b = ['TV31', 'RSI', 'STOK1', 'TV41']
    
    #Features Selected based on threshold of 50 percent
    #Classifier Selected Features
    #c = ['RSI', 'ATR2', 'STOK1', 'ATR21', 'RSI14X', 'SDKC9', 'O5']
    
    #Features Selected based on threshold of 50 percent
    #Regressor Selected Features
    #d = ['RSI', 'ATR2', 'STOK1', 'RSI14X', 'ATR5']
    
    #unique_values = list(set(a + b + c + d))
    #print(len(unique_values))
    #print(unique_values)
    
    #comp = ['ATR2', 'RSI14X', 'SDLR310', 'ROC1', 'ATR5', 'SDKC9', 'TV31', 'RSI', 'O5', 'TV41', 'ATR21', 'STOK1']
    
    
    # Span3 
    #Features Selected based on threshold of 50 percent
    #Classifier Selected Features
    #s1 = ['RSI', 'HourOfDay', 'TV11', 'RSI14X', 'TV41', 'SDBB9', 'ROC14', 'ATR2', 'SDLR310', 'SDKC91', 'STOK1', 'SDBB91', 'TV31', 'ATR21', 'SDKC9', 'H1']
    
    
    #Features Selected based on threshold of 50 percent
    #Regressor Selected Features
    #s2 = ['RSI', 'ATR2', 'STOK1', 'HourOfDay', 'ROC141', 'RSI14X', 'TV11', 'ROC14', 'TV31', 'ATR5', 'SDKC9', 'TV41', 'ROC', 'ROC1', 'SDBB9']
    
    #unique_values = list(set(s1 + s2))
    #print(len(unique_values))
    #print(unique_values)
    
    #span3 = ['RSI', 'ATR5', 'STOK1', 'H1', 'SDBB9', 'ATR2', 'SDBB91', 'TV11', 'ROC141', 
    #         'ATR21', 'TV31', 'RSI14X', 'SDKC9', 'ROC1', 'SDLR310', 'ROC14', 'HourOfDay', 'TV41', 'ROC', 'SDKC91']
    
    
    #span3 min
    #Features Selected based on threshold of 50 percent
    #Regressor Selected Features
    #s3m1 = ['RSI', 'STOK1', 'ATR5', 'ROC141', 'ROC', 'TV31']
    
    #Features Selected based on threshold of 50 percent
    #Classifier Selected Features
    #s3m2 = ['RSI', 'TV11', 'STOK1', 'TV41', 'RSI14X', 'ATR2']
    
    #unique_values = list(set(s3m1 + s3m2))
    #print(len(unique_values))
    #print(unique_values)
    
    
    #10
    s3_min = ['ROC', 'TV31', 'ATR2', 'TV11', 'ROC141', 'RSI14X', 'TV41', 'RSI', 'STOK1', 'ATR5']