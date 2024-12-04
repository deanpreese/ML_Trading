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
    ] 

    df = pd.read_csv(datafile[1])
    #df = df.drop(columns=['SeqClose'])
    
    #df = df[((df['RSIRAW'] > 50))]  
    #df = df[((df['RSIRAW'] > 0) & (df['RSIRAW'] < 30))]  
    #df = df[((df['RSIRAW'] > 70) & (df['RSIRAW'] < 100))]  
    
    #df = df[((df['RSI'] > 0) & (df['RSI'] < 30))]  
    #df = df[((df['RSI'] > 70) & (df['RSI'] < 100))]  
    
    
    
    X = df
    X = X.drop(columns=['output', 'outputC'])
    
    
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

    if use_class:
        #print("Classifier Sorted Importance")
        #print(" ")
        #print(importance_df_sorted_c)
        print(" ")
        print(f"Features Selected based on threshold of {threshold} percent")
        print("Classifier Selected Features")
        print(important_features_c)
        print("Perf Results")
        print(performance_df_c)
        print(" ")

        
    if use_reg:
        print("------------------------------------------------------------")
        #print(" ")
        #print("Regressor Sorted Importance")
        #print(" ")
        #print(importance_df_sorted_r)
        print(" ")
        print(f"Features Selected based on threshold of {threshold} percent")
        print("Regressor Selected Features")
        print(important_features_r)
        print("Perf Results")
        print(performance_df_r)
        print(" ")



if __name__ == "__main__":
    run()

    model_x_3070_imp_full =['SDKC7CU', 'ZL79X', 'SeqClose', 'TV3', 'ROC14', 'ATR2', 'STO5135D', 'STO7143D', 
                            'TV4', 'SDKC91', 'ZC79X', 'ATR9', 'TV5', 'RSIRAW', 'COMP3', 'SDBB9CL', 'SDKC9', 'TV2', 'TV6', 
                            'SDKC7CL', 'COMP0', 'ZH79X', 'SDBB91', 'TV1', 'COMP2']
    
    model_x_3070_imp_slim = ['SDKC9', 'COMP3', 'STO7143D', 'ATR9', 'SDBB91', 'RSIRAW', 'COMP2', 'TV6', 'SDKC7CU']



    model_x_3070_LT_30_50 = ['TV6', 'TV1', 'STO7143D', 'ATR14', 'SDBB9CL', 'SDKC7CU', 'SDKC10CL', 'ATR5', 'ZH79X', 'COMP3', 'SDKC9', 'TV3', 
                        'SDBB9', 'ATR2', 'STO7143K', 'ZC79X', 'TV2', 'ROC9', 'COMP2', 'ROC14', 'STO5135K', 'STO5135D', 'RSIRAW']
    
    model_x_3070_LT_30_75 = ['RSIRAW', 'ROC14', 'TV1', 'COMP2', 'SDKC9', 'ROC9']



    model_x_3070_GT_70_75 = ['SDKC9', 'ZH79X', 'ROC14', 'COMP3', 'RSIRAW', 'COMP2', 'ATR2', 
                             'STO7143D', 'TV6', 'TV3', 'SDKC7CU', 'ROC9', 'ZC79X', 'ATR14', 'TV1', 'STO5135K']

    model_x_3070_GT_70_50 = ['ZC79X', 'SDKC9', 'MinOfHour', 'SDBB14CU', 'ADX14', 'ATR14', 'SDKC10CU', 'ATR2', 
                             'SDKC7CU', 'TV6', 'COMP2', 'STO5135K', 'RSIRAW', 'STO5135D', 'COMP3', 'TV4', 'ADX9', 'TV1', 
                             'STO7143D', 'ZH79X', 'HourOfDay', 'ROC7', 'SeqClose']


    model_x_3070_GT_50_75 = ['SDKC9', 'COMP2', 'ADX14', 'RSIRAW', 'SDKC91', 'SeqClose', 'TV3', 'STO5135D', 'TV4', 'SDKC7CU']

    model_x_3070_GT_50_50 = ['TV5', 'ROC9', 'ATR14', 'TV4', 'COMP2', 'SDKC7CL', 'SDKC9', 'TV6', 'HourOfDay', 'ZL79X', 
                             'MinOfHour', 'ZH79X', 'RSIRAW', 'ROC14', 'ATR2', 'TV1', 'ZC79X', 'TV3', 'SDKC91', 'STO5135D', 'SDBB9CU', 
                             'SDKC7CU', 'ADX14', 'SeqClose']

    model_x_3070_LT_50_75 = ['RSIRAW', 'ZH79X', 'SDKC9', 'SDKC7CU', 'COMP2']

    model_x_3070_LT_50_50 =  ['TV4', 'SDBB9CL', 'TV1', 'COMP3', 'TV3', 'ZH79X', 'STO7143D', 'HourOfDay', 'ATR2', 'SDKC7CL', 'ADX14', 'SDKC9', 
                            'SeqClose', 'ATR5', 'Day', 'ADX9', 'SDKC10CL', 'ROC9', 'MinOfHour', 'STO5135D', 'RSIRAW', 'SDKC7CU', 
                            'ZC79X', 'ROC14', 'COMP2', 'TV6']
    
    
    lucky_13_3070_RSI_LT_50 = ['ATR2', 'ROC', 'ATR21', 'SDKC9', 'SDBB91', 'RSI']
    lucky_13_3070_RSI_GT_50 = ['RSI', 'ATR2', 'ATR21', 'ROC', 'ATR5']
    lucky13_3070_comp = ['SDKC9', 'ATR5', 'ROC', 'ATR2', 'SDBB91', 'ATR21', 'RSI']
    lucky13_3070_min = ['RSI', 'ATR2', 'ROC']
   
   