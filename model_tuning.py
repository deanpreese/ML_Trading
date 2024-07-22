import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from sklearn.metrics import mean_squared_error
from ml_model.data_func import simple_split_and_scale
import optuna



datafile = [ 
        'data/buildSeqInd_Lucky13_5M_3070.csv',   #0
        'data/buildSeqInd_Lucky13_5M_ALL.csv',  #1
        'data/buildSeqInd_Lucky13_F.csv',  #2
        'data/buildSeqInd_Lucky13_D.csv',  #3
        'data/buildSeqInd_Lucky13_F_3070.csv',  #4
    ]

dtx = pd.read_csv(datafile[0])
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
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 50),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    }

    model = xgb.XGBRegressor(**params)
    #model = xgb.XGBRFRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    predictions = model.predict(X_val)
    rmse = mean_squared_error(y_val, predictions, squared=False)
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
    rmse = mean_squared_error(y_val, predictions, squared=False)
    return rmse


def objective_lgb(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        'verbose': 0,
        "n_estimators": 1000,
        "bagging_freq": 1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    rmse = mean_squared_error(y_val, predictions, squared=False)
    return rmse


def main():
    
    #print(" ")
    #print("XGBoost Tuning")
    #study_xgb = optuna.create_study(direction='minimize')
    #study_xgb.optimize(objective_xgb, n_trials=20)

    #print(" ")
    #print("Lightgbm Tuning")
    #study_lgb = optuna.create_study(direction='minimize')
    #study_lgb.optimize(objective_lgb, n_trials=20)

    print(" ")
    print("CatBoost Tuning")
    study_cat = optuna.create_study(direction='minimize')
    study_cat.optimize(objective_cat, n_trials=50)
    
    
    #print(" ")
    #print("Best XGB parameters")
    #print(study_xgb.best_trial)
    #print(" ")
    #print("Best LGB parameters")
    #print(study_lgb.best_trial)
    #print(" ")
    print("Best CAT parameters")
    print(study_cat.best_trial)
    print(" ")


    

if __name__ == "__main__":
    main()
    