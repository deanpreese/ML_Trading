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
            'data/Lucky13_3070_oos.csv',   
            'data/Lucky13_3070.csv',  #1
            'data/Model_X_3070_oos.csv',  
            'data/Model_X_3070.csv',  #3        
            'data/The_13_R_3070.csv' #4
    ] 

    df = pd.read_csv(datafile[4])
    
    #df = df[((df['RSIRAW'] > 50))]  
    #df = df[((df['RSIRAW'] > 70))]  
    
    
    #df = df[((df['RSIRAW'] > 0) & (df['RSIRAW'] < 30))|   
    #    ((df['RSIRAW'] > 70) & (df['RSIRAW'] < 100))]  
    
    #df = df[((df['RSI'] > 0) & (df['RSI'] < 30))]  
    #df = df[((df['RSI'] > 70) & (df['RSI'] < 100))]  
    
    fd = [
        'SDBB9L','SDBB9U',
        'SDKC9U','SDKC9L',
        'ROC',
        #'ATR54','ATR53','ATR52','ATR51',
        #'ATR5',
        #'ATR21',
        'ATR2',
        'RSI','STOK1']
    
    
    X = df[fd]
    #X = X.drop(columns=['output', 'outputC', 'SeqClose', 'RSIRAW',])
    
    lucky_13_columns = [
    "SDLR310", "SDBB91", "SDKC91", "SDKC9", "ROC", "ATR54", "ATR53", "ATR52", 
    "ATR51", "ATR5", "ATR21", "ATR2", "RSI", "STOK1"
    ]
    #X = df[lucky_13_columns]
    
    model_x_columns = [
        "Year", "Month", "Day", "DayOfWeek", "HourOfDay", "MinOfHour", "RSIRAW", "SeqClose",
        "SDBB9", "SDBB91", "SDKC9", "SDKC91", "SDBB29", "SDBB291", "SDKC29", "SDKC291",
        "SDBB14CU", "SDBB14CL", "SDBB9CU", "SDBB9CL", "SDKC10CU", "SDKC10CL", "SDKC7CU",
        "SDKC7CL", "ROC14", "ROC9", "ROC7", "ATR14", "ATR9", "ATR5", "ATR2", "RSI14", 
        "RSI9", "ADX14", "ADX9", "STO5135K", "STO5135D", "STO7143K", "STO7143D", "TV1", 
        "TV2", "TV3", "TV4", "TV5", "TV6", "ZH79X", "ZL79X", "ZC79X", "COMP0", "COMP1", 
        "COMP2", "COMP3"
    ]
    #X = df[model_x_columns]
    

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


    print(X.shape)

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
    run()

   