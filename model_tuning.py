import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from sklearn.metrics import mean_squared_error
from ml_model.data_func import simple_split_and_scale
import optuna

datafile = [ 
        'data/Lucky13_3070_oos.csv',   
        'data/Lucky13_3070.csv',  #1
        'data/ndata_diff_lucky13_3070_oos.csv', 
        'data/ndata_diff_lucky13_3070.csv', #3
        'data/ndata_lag_3070_oos.csv', 
        'data/ndata_lag_3070.csv', #5
]

dtx = pd.read_csv(datafile[1])
X = dtx
X = X.drop(columns=['output', 'outputC'])
y = dtx['output'].values
fl_out = list(X.columns)
X_train, X_val, y_train, y_val = simple_split_and_scale(X, y, 0.7, 42)


def objective_xgb(trial):
    params = {
        "objective": "reg:squarederror",
        "n_estimators": 1000,
        "verbosity": 0,
        'lambda': trial.suggest_loguniform('lambda', 7.0, 17.0),
        'alpha': trial.suggest_loguniform('alpha', 7.0, 17.0),
        'eta': trial.suggest_categorical('eta', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'gamma': trial.suggest_categorical('gamma', [18, 19, 20, 21, 22, 23, 24, 25]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.01,0.012,0.014,0.016,0.018, 0.02]),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'colsample_bynode': trial.suggest_categorical('colsample_bynode', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'n_estimators': trial.suggest_int('n_estimators', 400, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 8, 600),  
        'max_depth': trial.suggest_categorical('max_depth', [3, 4, 5, 6, 7]),  
        'subsample': trial.suggest_categorical('subsample', [0.5,0.6,0.7,0.8,1.0]),
        'random_state': 42
        
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, predictions)
    rmse = rmse =  rmse = mse**.5
    
    return rmse


def objective_cat(trial):
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 15),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    model = cb.CatBoostRegressor(**params, silent=True)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, predictions)
    rmse = rmse =  rmse = mse**.5
    return rmse


def objective_lgb(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        'verbose': 0,
        "n_estimators": 1000,
        'metric': 'rmse', 
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.014,0.017,0.02]),
        'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
        'num_leaves' : trial.suggest_int('num_leaves', 1, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, predictions)
    rmse = rmse =  rmse = mse**.5
    return rmse


def main():
    
    
    #print(" ")
    #print("XGBoost Tuning")
    #study_xgb = optuna.create_study(direction='minimize')
    #study_xgb.optimize(objective_xgb, n_trials=20)

    print(" ")
    print("Lightgbm Tuning")
    study_lgb = optuna.create_study(direction='minimize')
    study_lgb.optimize(objective_lgb, n_trials=20)

    #print(" ")
    #print("CatBoost Tuning")
    #study_cat = optuna.create_study(direction='minimize')
    #study_cat.optimize(objective_cat, n_trials=50)
    
    
    #print(" ")
    #print("Best XGB parameters")
    #print(study_xgb.best_trial)
    #print(" ")
    print("Best LGB parameters")
    print(study_lgb.best_trial)
    print(" ")
    #print("Best CAT parameters")
    #print(study_cat.best_trial)
    #print(" ")


    

if __name__ == "__main__":
    main()
    