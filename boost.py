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


    

def gen_results(models, X_train, y_train, X_test, y_test, columns, threshold):
    
    baseline_data =[]
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_r2 = r2_score(y_test,y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        if "Regressor" in name:
            perf, total, mse, rmse, mae = gen_reg_stats(y_test, y_pred)
            baseline_performance = perf
        else:           
            baseline_performance = accuracy_score(y_test, y_pred)
            
        baseline_data.append({'Model': name, 'Baseline Performance': baseline_performance, 'R2': model_r2, 'MSE':mse })
      
        
      
    baseline_df = pd.DataFrame(baseline_data)
    return baseline_df
    
    
# --------------
def run():

    datafile = [ 
        'data/Lucky13_3070_oos.csv',   
        'data/Lucky13_3070.csv',  #1
        'data/Lucky13_EX_3070_oos.csv',  
        'data/Lucky13_EX_3070.csv',  #3
        'data/new_model_Z_lucky13_3070.csv', #4
        'data/ReFried_5M_ALL.csv' #5
        
    ]

    df = pd.read_csv(datafile[1])
    
    #df = df[((df['RSI'] > 20) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 80)]  
    
    #df = df[(df['RSI'] > 60) & (df['RSI'] < 80)]  #  81%
    df = df[(df['RSI'] > 60) & (df['RSI'] < 75)]   # 878%
    #df = df[(df['RSI'] > 20) & (df['RSI'] < 40)]  # 84%
    #df = df[(df['RSI'] > 25) & (df['RSI'] < 40)]  # 91%

    
    #f_list = ['RSI','ADX1','STOK1','ATR5','ATR51','SDKC9','EMAL21213',
    #            'EMAL10102','ADX2','SDLR93','EMAL10103','EMAL21211','EMAL10101','FOSC','FOSC1','ADX','ATR54','SDLR92','ROC', 'output','outputC'] 
    
    
    X = df
    X = X.drop(columns=['output', 'outputC'])
    
    y = df['outputC'].values
    y2 = df['output'].values
    
    threshold = 75


    X_train_c, X_test_c, y_train_c, y_test_c = simple_split_and_scale(X, y, 0.2, 42)
    X_train_r, X_test_r, y_train_r, y_test_r = simple_split_and_scale(X, y2, 0.2, 42)
    
    
    models_c = {
        #'RandomForestClassifier' :RandomForestClassifier(random_state=42, verbose=2, n_jobs=-1),
        'LightGBM': LGBMClassifier(random_state=42, verbose=2, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=2),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=2)
    }

    models_r = {
        #'RandomForestRegressor' :RandomForestRegressor(random_state=42, verbose=2, n_jobs=-1),
        'LightGBMRegressor': LGBMRegressor(random_state=42, verbose=2, n_jobs=-1),
        'XGBoostRegressor': XGBRegressor(random_state=42, use_label_encoder=False, verbosity=2),
        'CatBoostRegressor': CatBoostRegressor(random_state=42, verbose=2)
    }

    use_class = False
    use_reg = True

    if use_class:
    
        baseline_df = gen_results(models_c, X_train_c, y_train_c, X_test_c, y_test_c, X.columns, threshold)    
        print(baseline_df)
        print(" ")

        
    if use_reg:
        baseline_df  = gen_results(models_r, X_train_r, y_train_r, X_test_r, y_test_r, X.columns, threshold)            
        print(baseline_df)
        print(" ")
        



if __name__ == "__main__":
    run()

