import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from sklearn.metrics import mean_squared_error,accuracy_score
from ml_model.data_func import simple_split_and_scale
import optuna


# =====================================
def objective_cat_c(trial, X_train, y_train, X_val, y_val):


    param = {
            "iterations" : trial.suggest_int("iterations", 100, 1000),
            "learning_rate" : trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 1, 10),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "used_ram_limit": "3gb",
            "eval_metric": "Accuracy",
        }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)


    model = cb.CatBoostClassifier(**param)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return acc



def objective_lgb_c(trial, X_train, y_train, X_val, y_val):

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    model = lgb.LGBMClassifier(**param)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return acc



def objective_xgb_c(trial, X_train, y_train, X_val, y_val):
    
    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    model = xgb.XGBClassifier(**param )  
    model.fit(X_train, y_train,verbose=False)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return acc

# =====================================
def objective_xgb_r(trial, X_train, y_train, X_val, y_val):
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


def objective_cat_r(trial, X_train, y_train, X_val, y_val):
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 15),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 3, 50),
    }

    model = cb.CatBoostRegressor(**params, silent=True)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, predictions)
    rmse = rmse =  rmse = mse**.5
    return rmse


def objective_lgb_r(trial, X_train, y_train, X_val, y_val):
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


# =====================================

def study_xgb_r(X_train, y_train, X_val, y_val):
    print(" ")
    print("XGBoost_R Tuning")
    study_xgb_r = optuna.create_study(direction='minimize')
    f_xgb_r = lambda trial: objective_xgb_r(trial, X_train, y_train, X_val, y_val)
    study_xgb_r.optimize(f_xgb_r, n_trials=20)
    return study_xgb_r.best_trial.params

def study_xgb_c(X_train, y_train, X_val, y_val):
    print(" ")
    print("XGBoost_C Tuning")
    study_xgb_c = optuna.create_study(direction='minimize')
    f_xgb_c = lambda trial: objective_xgb_c(trial, X_train, y_train, X_val, y_val)
    study_xgb_c.optimize(f_xgb_c, n_trials=20)
    return study_xgb_c.best_trial.params


def study_lgb_r(X_train, y_train, X_val, y_val):
    print(" ")
    print("Lightgbm_R Tuning")
    study_lgb_r = optuna.create_study(direction='minimize')
    f_lgb_r = lambda trial: objective_lgb_r(trial, X_train, y_train, X_val, y_val)
    study_lgb_r.optimize(f_lgb_r, n_trials=20)
    return study_lgb_r.best_trial.params

def study_lgb_c(X_train, y_train, X_val, y_val):
    print(" ")
    print("Lightgbm_C Tuning")
    study_lgb_c = optuna.create_study(direction='minimize')
    f_lgb_c = lambda trial: objective_lgb_c(trial, X_train, y_train, X_val, y_val)
    study_lgb_c.optimize(f_lgb_c, n_trials=20)    
    return study_lgb_c.best_trial.params

def study_cat_r(X_train, y_train, X_val, y_val):
    print(" ")
    print("CatBoost_R Tuning")
    study_cat_r = optuna.create_study(direction='minimize')
    f_cat_r = lambda trial: objective_cat_r(trial, X_train, y_train, X_val, y_val)
    study_cat_r.optimize(f_cat_r, n_trials=50)
    return study_cat_r.best_trial.params
    
def study_cat_c(X_train, y_train, X_val, y_val):
    print(" ")
    print("CatBoost_C Tuning")
    study_cat_c = optuna.create_study(direction='minimize')
    f_cat_c = lambda trial: objective_cat_c(trial, X_train, y_train, X_val, y_val)
    study_cat_c.optimize(f_cat_c, n_trials=50)
    return study_cat_c.best_trial.params

def main():

    datafile = [ 
            #'data/Lucky13_3070_oos.csv',   
            'data/Lucky13_3070.csv',  #1
            #'data/ndata_diff_lucky13_3070_oos.csv', 
            #'data/ndata_diff_lucky13_3070.csv', #3
            #'data/ndata_lucky_13_lag_3070_oos.csv', 
            #'data/ndata_lucky13_lag_3070.csv', #5
            #'new_model_Z_lucky13_3070_oos.csv',
            #'new_model_Z_lucky13_3070.csv' #7,

    ]


    cols = ['file','model', 'data']
    comp_df = pd.DataFrame(columns=cols)

    for i in range(len(datafile)):

        file = datafile[i]
        dtx = pd.read_csv(file)

        X = dtx
        X = X.drop(columns=['output', 'outputC'])
        y = dtx[['output','outputC']]
        fl_out = list(X.columns)
        X_train, X_val, y_train_o, y_val_o = simple_split_and_scale(X, y, 0.7, 42)

        y_train_r = y_train_o['output'].values
        y_train_c = y_train_o['outputC'].values
        y_val_r = y_val_o["output"].values          
        y_val_c = y_val_o["outputC"].values
        
        
        study_xgb_r_best_trial = study_xgb_r(X_train, y_train_r, X_val, y_val_r)
        t = {'file': file ,'model':'XGBR','data' : study_xgb_r_best_trial}
        comp_df = comp_df._append(t, ignore_index=True)
        
        study_cat_r_best_trial = study_cat_r(X_train, y_train_r, X_val, y_val_r)               
        t = {'file': file ,'model':'CATR','data' : study_cat_r_best_trial}
        comp_df = comp_df._append(t, ignore_index=True)
        
        study_lgb_r_best_trial = study_lgb_r(X_train, y_train_r, X_val, y_val_r) 
        t = {'file': file ,'model':'LGBR','data' : study_lgb_r_best_trial}
        comp_df = comp_df._append(t, ignore_index=True)
        
        """
        
        study_cat_c_best_trial = study_cat_c(X_train, y_train_c, X_val, y_val_c)               
        t = {'file': file ,'model':'CATC','data' : study_cat_c_best_trial}
        comp_df = comp_df._append(t, ignore_index=True)
        
        
        study_xgb_c_best_trial = study_xgb_c(X_train, y_train_c, X_val, y_val_c)
        t = {'file': file ,'model':'XGBC','data' : study_xgb_c_best_trial}
        comp_df = comp_df._append(t, ignore_index=True)
        
        
        study_lgb_c_best_trial = study_lgb_c(X_train, y_train_c, X_val, y_val_c) 
        t = {'file': file ,'model':'LGBC','data' : study_lgb_c_best_trial}
        comp_df = comp_df._append(t, ignore_index=True)
        """
        
      
        comp_df.to_csv('params.csv', mode='a', index=False, header=False)
        
    print(comp_df)

if __name__ == "__main__":
    main()
    