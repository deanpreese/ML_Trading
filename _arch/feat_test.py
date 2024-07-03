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
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def calc_stats( y_test, predicted_values):
    correct1 = 0 
    total = 0
    for i in range(len(y_test)):
        target_output = y_test[i] if i < len(y_test) else 0
        predicted_output = predicted_values[i]  # Predicted output for the i-th sample

        if ( target_output > 0 and predicted_output > 0):
            correct1= correct1 + 1 

        if ( target_output < 0 and predicted_output < 0):
            correct1= correct1 + 1 
        
        if ( target_output == 0 and predicted_output == 0):
            correct1= correct1 + 1     

        total = total + 1    

    per1 = round((correct1)/total,4)
        
    return(total, correct1, per1)


def run_test(model, file_to_load,  list_80, list_60):

    file_loaded = pd.read_csv(file_to_load)
    file_loaded = file_loaded.drop(columns=['outputC'])

    
    base = model
    feature_columns = list(file_loaded.columns[:-1])
    num_columns = len(file_loaded.axes[1]) 
    input_features =  num_columns -1
    X = file_loaded.iloc[:, 0:input_features]  
    y = file_loaded['output'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    base.fit(X_train, y_train)
    base_preds = base.predict(X_test)
    base_score = base.score(X_test, y_test)
    base_mse = mean_squared_error(y_test, base_preds)
    base_rmse = base_mse**.5
    total, correct, per = calc_stats(y_test, base_preds)
    
    agg_80_predicts = []
    agg_80_weighted_predicts = []
    base_80 = model
    data_80 = file_loaded[list_80]
    num_80_columns = len(data_80.axes[1]) 
    input_80_features =  num_80_columns -1
    X80 = data_80.iloc[:, 0:input_80_features]  
    y80 = data_80['output'].values
    X_train80, X_test80, y_train80, y_test80 = train_test_split(X80, y80, test_size=0.3, random_state=42)

    base_80.fit(X_train80, y_train80)
    base_80_preds = base_80.predict(X_test80)
    base_80_score = base_80.score(X_test80, y_test80)
    base_80_mse = mean_squared_error(y_test80, base_80_preds)
    base_80_rmse = base_80_mse**.5
    total80, correct80, per80 = calc_stats(y_test80, base_80_preds)
    
    agg_60_predicts = []
    agg_60_weighted_predicts = []
    base_60 = model
    data_60 = file_loaded[list_60]
    num_60_columns = len(data_60.axes[1]) 
    input_60_features =  num_60_columns -1
    X60 = data_60.iloc[:, 0:input_60_features]  
    y60 = data_60['output'].values
    X_train60, X_test60, y_train60, y_test60 = train_test_split(X60, y60, test_size=0.3, random_state=42)

    base_60.fit(X_train60, y_train60)
    base_60_preds = base_60.predict(X_test60)
    base_60_score = base_60.score(X_test60, y_test60)
    base_60_mse = mean_squared_error(y_test60, base_60_preds)
    base_60_rmse = base_60_mse**.5
    total60, correct60, per60 = calc_stats(y_test60, base_60_preds)
    
    
    
    print(" ******************************************")
    print(f"File: {file_to_load}")
    print(f"Model: {model}")
    print("Base Model")
    print(X.shape)
    print(f"Score {base_score}  MSE {base_mse}  RMSE {base_rmse}") 
    print(f"Total {total}  Correct {correct}  Percent {per}")
    print("")
    print("Model  60")
    print(X60.shape)
    print(f"Score {base_60_score}  MSE {base_60_mse}  RMSE {base_60_rmse}") 
    print(f"Total {total60}  Correct {correct60}  Percent {per60}")
    print("")
    print("Model 80")
    print(X80.shape)
    print(f"Score {base_80_score}  MSE {base_80_mse}  RMSE {base_80_rmse}") 
    print(f"Total {total80}  Correct {correct80}  Percent {per80}")
    print("")




lgb_params= {
            'n_estimators' : 150,
            'objective': 'regression',
            'min_child_samples' : 7,
            'subsample' : 1,
            'num_leaves': 35,
            'colsample_bytree' : 1,
            'random_state' : 0,
            'n_jobs' : -1,
            'learning_rate': 0.01,
            'verbose': 0,
            }

xgb_params = {
        'max_depth': 3,
        'booster' : 'dart', 
        'learning_rate': 0.1,
        'n_estimators':  50, 
        #'gamma': [0, 20], 
        #'subsample': [0.8,1], 
        #'colsample_bytree': [0.8,1], 
        #'lambda': [0, 0.1, 1],
        'tree_method': "hist",
        'eval_metric': "mae",
        "verbosity" : 0
    }


model = LGBMRegressor(**lgb_params)
model2 = XGBRegressor(**xgb_params)


file = 'data/buildSeqInd_Lucky13_F.csv'
list80 = ['SDKC9', 'ATR3', 'STOK1', 'SDKC91', 'ATR21', 'output']
list60 = ['SDKC9', 'ATR3', 'STOK1', 'output']
run_test(model,file,  list80, list60)
run_test(model2,file,  list80, list60)

file = 'data/Ind_F.csv'
list80 = ['CCI20', 'SDKC9', 'VOSC20', 'ADX14', 'STOK15657', 'VOSC7', 'ROC7', 'STOK7217', 'SDKC14', 'RSI3', 'STOD7217', 'SDBB20', 'output']
list60 = ['CCI20', 'SDKC9', 'VOSC20', 'ADX14', 'STOK15657', 'VOSC7', 'ROC7', 'STOK7217', 'output']
run_test(model,file,  list80, list60)
run_test(model2,file,  list80, list60)

file = 'data/buildSeqInd_Lucky13_5M_ALL.csv'
list80 = ['ATR21', 'STOK1', 'ATR34', 'ATR3', 'ATR32', 'output']
list60 = ['ATR21', 'STOK1', 'ATR34', 'output']
run_test(model, file,  list80, list60)
run_test(model2,file,  list80, list60)
