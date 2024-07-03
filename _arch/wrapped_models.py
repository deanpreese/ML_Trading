from xgboost import XGBClassifier, XGBRegressor
from lightgbm  import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from common.common_func import calc_reg_streaks, show_stats, create_param_list

#import mlflow.onnx


import pandas as pd
from enum import Enum

from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

import mlflow
#mlflow.set_tracking_uri(uri="http://127.0.0.1:8888")
#mlflow.set_tracking_uri(uri="http://10.0.0.74:8888")
mlflow.set_tracking_uri(uri="http://10.0.0.50:8888")

# -----------------------------------------------------
def gen_classifier_data(model, X_train, y_train, X_test, y_test, y_pred):
        
        TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
        accuracy = accuracy_score(y_pred, y_test)
        precision = precision_score(y_pred, y_test)
        recall = recall_score(y_pred, y_test)

        mlflow.log_param("FeatureCount" , (X_train.shape[1]))
        mlflow.log_metric('Accuracy', accuracy)
        mlflow.log_metric('Precision', precision)
        mlflow.log_metric('Recall', recall)
        mlflow.log_metric('TrueNeg', TN)
        mlflow.log_metric("FalsePos", FP)
        mlflow.log_metric("FalseNeg", FN)
        mlflow.log_metric("TruePos", TP)
        
        mlflow.log_metric("Perf", accuracy)

        tot = TN + FP + FN + TP
        
        return accuracy, precision, recall, TN/tot, FP/tot, FN/tot, TP/tot, tot


# -----------------------------------------------------
def gen_regressor_data(model, X_train, y_train, X_test, y_test, y_pred):
        
        mse = mean_squared_error(y_test, y_pred, squared=True)
        rmse =mean_squared_error(y_test, y_pred, squared=False)
        r2 =r2_score(y_test, y_pred)
        score = model.score(X_test, y_test)
        mae = float(mean_absolute_error(y_test,y_pred))                
        perf, tot = show_stats(False, y_test, y_pred)
        
        current_streak, longest_win_streak, longest_loss_streak, aws, als, awm, alm, aum, adm = calc_reg_streaks(y_pred, y_test, False, False, False)
                
        
        mlflow.log_param("FeatureCount" , (X_train.shape[1]))
        mlflow.log_metric('MSE', mse)
        mlflow.log_metric('RMSE', rmse)
        mlflow.log_metric('R2', r2)
        mlflow.log_metric('Score', score)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("Perf", perf)
        mlflow.log_metric("Total", tot)
        
        mlflow.log_metric('Longest Win Streak', longest_win_streak)
        mlflow.log_metric('Longest Loss Streak', longest_loss_streak)
        mlflow.log_metric('Ave Win Streak', aws)
        mlflow.log_metric("Ave Loss Streak", als)
        mlflow.log_metric('Ave Win Miss', awm)
        mlflow.log_metric("Ave Loss Miss", alm)
        mlflow.log_metric("Ave Up Miss", aum)
        mlflow.log_metric("Ave Dwn Miss", adm)
        
        
        return perf, tot, mse, rmse, r2, score, mae

# -----------------------------------------------------

class ModelType(Enum):
    CLASS = 2
    REGR = 1

# -----------------------------------------------------
#
#   Classifier Wrappers   
#
# -----------------------------------------------------
class TunableCatBoostClassifier(CatBoostClassifier):
    def __init__(self, **kwargs):
        self.used_params = kwargs
        self.features_used = []
        super().__init__(**kwargs)

    def model_type(self):
        return ModelType.CLASS

    def param_set(self):    
        
        params = {
        'learning_rate' : 0.009,
        'depth' : 3,
        'l2_leaf_reg' :  3.0,
        'min_child_samples' : 32,
        'iterations' : 1000,
        #'random_state' : [0],
        #'thread_count' : [-1],
        }
        return params
    
    def param_grid(self):
       
        params = {
        'learning_rate' : [ 0.009, 0.01],
        #'depth' : [3,5,7,9,11],
        'depth' : [3,7,9,11],
        #'l2_leaf_reg' : [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0 ],
        'l2_leaf_reg' : [ 3.0, 4.0, 5.0 ],
        #'min_child_samples' : [1, 4, 8, 16, 32],
        'min_child_samples' : [ 8, 16, 32],
        #'grow_policy' : ['Depthwise'],
        'iterations' : [1000],
        #'eval_metric' : ['RMSE'],
        #'random_state' : [0],
        #'boosting_type' : ['Ordered', 'Plain'],
        #'thread_count' : [-1],
        }
        
        return create_param_list(params)
    
    def track_model(self, experiment_id, nested, model, X_train, y_train, X_test, y_test):
            
        with mlflow.start_run(experiment_id = experiment_id, nested=nested):
        
            self.run_id = mlflow.active_run().info.run_id  
        
            modelx =  CatBoostClassifier(**self.param_set())
            model.fit(X_train, y_train)
            modelx.fit(X_train, y_train)
        
            y_pred = model.predict(X_test)
            pred_proba = model.predict_proba(X_test)
            mlflow.log_params( self.used_params )
            mlflow.catboost.log_model(modelx, "model")
            #mlflow.catboost.log_model(model, "TunableCatBoostClassifier")
            mlflow.log_table(data=pd.DataFrame(self.features_used), artifact_file="features_used.json")                 
            accuracy, precision, recall, TN, FP, FN, TP, tot = gen_classifier_data(model, X_train, y_train, X_test, y_test, y_pred)
            
        return accuracy, precision, recall, TN, FP, FN, TP, tot, y_pred, pred_proba  
    
    
# -----------------------------------------------------
class TunableLGBMClassifier(LGBMClassifier):
    
    def __init__(self,boosting_type='gbdt', num_leaves=31, learning_rate=0.1, n_estimators=100, objective=None,
                 min_child_samples=20, subsample=1.0, colsample_bytree=1.0, random_state=None, n_jobs=-1):

        #mlflow.lightgbm.autolog() 
        self.features_used = []
         
        kwargs = {
            'boosting_type': boosting_type,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'objective': objective,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'n_jobs': n_jobs
        }
         
        self.used_params = kwargs
        super().__init__(**kwargs) 
         
                
    def model_type(self):
        return ModelType.CLASS
    
    def param_set(self):    
        params = {
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 10,
            'objective': None,
            'min_child_samples': 20,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'random_state': 0,
            'n_jobs': -1
        }
        
        
        return params
    
    def param_grid(self):
        
        params = {
            'boosting_type': 'gbdt',
            #'class_weight': None,
            #'colsample_bytree': 1.0,
            #'importance_type': 'split',
            'learning_rate': 0.1,
            'max_depth': -1,
            'min_child_samples': 20,
            #'min_child_weight': 0.001,
            #'min_split_gain': 0.0,
            'n_estimators': 100,
            'num_leaves': 31,
            #'objective': None,
            #'random_state': None,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'subsample': 1.0,
            'subsample_for_bin': 200000,
            'subsample_freq': 0,            
        }
        
        return create_param_list(params)
        

    
    def track_model(self, experiment_id, nested, model, X_train, y_train, X_test, y_test):
           
        with mlflow.start_run(experiment_id = experiment_id, nested=nested):
        
            self.run_id = mlflow.active_run().info.run_id  
        
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            pred_proba = model.predict_proba(X_test)
            mlflow.lightgbm.log_model(model, "model")
            #mlflow.lightgbm.log_model(model, "TunableLGBMClassifier")
            mlflow.log_table(data=pd.DataFrame(self.features_used), artifact_file="features_used.json")                 
            accuracy, precision, recall, TN, FP, FN, TP, tot = gen_classifier_data(model, X_train, y_train, X_test, y_test, y_pred)
   
        return accuracy, precision, recall, TN, FP, FN, TP, tot, y_pred, pred_proba  
    

# -----------------------------------------------------
class TunableXGBClassifier(XGBClassifier):
    def __init__(self, **kwargs):
        
        #mlflow.xgboost.autolog()
        self.used_params = kwargs        
        super().__init__(**kwargs)

    def model_type(self):
        return ModelType.CLASS
    
    def param_set(self):
        
        params = {
            'colsample_bytree': 0.6655392754230048, 
            'gamma': 4.198875359789924, 
            'max_depth': 17.0, 
            'min_child_weight': 1.0, 
            'reg_alpha': 57.0, 
            'reg_lambda': 0.896332305739873
            }
            
        return params
    
    def param_grid(self):
        params = { 
            'max_depth': [ 3, 18, 1],
            'gamma': [1,9],
            'reg_alpha' : [40,180,1],
            'reg_lambda' : [ 0,1],
            'colsample_bytree' : [0.5,1],
            'min_child_weight' : [0, 10],
            'n_estimators': [180],
            'seed': [0]
    
        }
        
        return create_param_list(params)
        
    def track_model(self, experiment_id, nested, model, X_train, y_train, X_test, y_test):
           
        with mlflow.start_run(experiment_id = experiment_id, nested=nested):
        
            self.run_id = mlflow.active_run().info.run_id  
        
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            pred_proba = model.predict_proba(X_test)
            mlflow.log_params( self.used_params )
            #mlflow.xgboost.log_model(model, "TunableXGBClassifier")
            mlflow.xgboost.log_model(model, "model")
            mlflow.log_table(data=pd.DataFrame(self.features_used), artifact_file="features_used.json")                 
            accuracy, precision, recall, TN, FP, FN, TP, tot = gen_classifier_data(model, X_train, y_train, X_test, y_test, y_pred)
            
        return accuracy, precision, recall, TN, FP, FN, TP, tot, y_pred, pred_proba  
    
     
# -----------------------------------------------------
#
#   Regressor Wrappers   
#
# -----------------------------------------------------        

class TunableCatBoostRegressor(CatBoostRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.used_params = kwargs
        self.features_used = []
    
    def model_type(self):
        return ModelType.REGR
    
    def param_set(self):
        
        params = {
            "iterations": 800,
            "learning_rate": 0.01,
            "depth": 7,
            #"subsample": [0.05, 0.07, 1.0],
            #"colsample_bylevel": [ 0.05, 0.07,  1.0],
            #"min_data_in_leaf": [ 1, 5, 25, 50, 100],

        }
        return params
    
    def param_grid(self):
    
        params = {
            "iterations": [600,700,800,900 ],
            "learning_rate": [0.01, 0.3, 0.7, 0.1 ],
            "depth": [ 1, 3, 5, 7, 10],
            #"subsample": [0.05, 0.07, 1.0],
            #"colsample_bylevel": [ 0.05, 0.07,  1.0],
            #"min_data_in_leaf": [ 1, 5, 25, 50, 100],
        } 
    
        return create_param_list(params)


    def track_model(self, experiment_id, nested, model, X_train, y_train, X_test, y_test):
        
        with mlflow.start_run(experiment_id = experiment_id, nested=nested):
                        
            self.run_id = mlflow.active_run().info.run_id                        
            modelx =  CatBoostRegressor(**self.param_set())
            model.fit(X_train, y_train)
            modelx.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mlflow.log_params( self.used_params )
            mlflow.catboost.log_model(modelx, "model")
            #mlflow.catboost.log_model(model, "TunableCatBoostRegressor")
            mlflow.log_table(data=pd.DataFrame(self.features_used), artifact_file="features_used.json")                 
            perf, tot, mse, rmse, r2, score, mae = gen_regressor_data(model, X_train, y_train, X_test, y_test, y_pred)
        
        return perf, tot, mse, rmse, r2, score, mae, y_pred
    

# -----------------------------------------------------
class TunableLGBMRegressor(LGBMRegressor):
         
    def __init__(self, boosting_type='gbdt', num_leaves=31, learning_rate=0.1, n_estimators=100,
                 objective=None, min_child_samples=20, subsample=1.0, colsample_bytree=1.0,
                 random_state=None, n_jobs=-1, verbose=1):
        
        #mlflow.lightgbm.autolog(False)
        self.features_used = []            
        
         
        super().__init__(boosting_type=boosting_type, num_leaves=num_leaves, learning_rate=learning_rate,
                         n_estimators=n_estimators, objective=objective, min_child_samples=min_child_samples,
                        subsample=subsample, colsample_bytree=colsample_bytree, random_state=random_state, n_jobs=n_jobs, verbose=1)

    def model_type(self):
        return ModelType.REGR

    def param_set(self):
        
        params= {
             'n_estimators' : 150,
            'objective': 'regression',
            'min_child_samples' : 7,
            'subsample' : 1,
            'num_leaves': 35,
            'colsample_bytree' : 1,
            'random_state' : 0,
            'n_jobs' : -1,
            'learning_rate': 0.01,
            'verbose': 1,
            }
        
        return params        

    def param_grid(self):
    
        params = {
            'n_estimators' : [150],
            #'boosting_type': ['gbdt', 'rf', 'dart'],
            #'objective': ['regression'],
            'min_child_samples' : [5,7,9],
            #'subsample' : [1],
            'num_leaves': [19,21,23,25,30,35],
            #'colsample_bytree' : [1,2,3],
            'random_state' : [0],
            'n_jobs' : [-1],
            #'learning_rate': [0.01, 0.02, 0.03, 0.04 ],
            'learning_rate': [0.01],
            'verbose': [1],
        }    
    
        return create_param_list(params)
    
    def track_model(self, experiment_id, nested, model, X_train, y_train, X_test, y_test):
        
        self.experiment_id = experiment_id
            
        
        with mlflow.start_run(experiment_id = experiment_id, nested=nested):

            self.run_id = mlflow.active_run().info.run_id
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mlflow.lightgbm.log_model(model, "model")
            mlflow.log_table(data=pd.DataFrame(self.features_used), artifact_file="features_used.json")                 
            perf, tot, mse, rmse, r2, score, mae = gen_regressor_data(model, X_train, y_train, X_test, y_test, y_pred)
        
        return perf, tot, mse, rmse, r2, score, mae, y_pred
    

# -----------------------------------------------------
class TunableXGBRegressor(XGBRegressor):
    
    def __init__(self, **kwargs):
        self.used_params = kwargs

        self.used_params['device'] = 'cuda'

        self.features_used = []
        super().__init__(**kwargs)

    def model_type(self):
        return ModelType.REGR
        
    def param_set(self):
        
        params = {
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
            "verbosity" : 2
        }
        
        return params
        
    def param_grid(self):
                
        params = {
            'max_depth': [2, 3, 4, 5, 6], 
            'learning_rate': [0.1, 0.2, 0.3],
            'n_estimators': [25, 50, 100, 150], 
            #'gamma': [0, 20], 
            #'subsample': [0.8,1], 
            #'colsample_bytree': [0.8,1], 
            #'lambda': [0, 0.1, 1],
            'tree_method': ["hist"],
            'eval_metric': ["mae"],
            "verbosity" : [2]
        }
        
        return create_param_list(params)       
        
    
    def track_model(self, experiment_id, nested, model, X_train, y_train, X_test, y_test):
            
        with mlflow.start_run(experiment_id = experiment_id, nested=nested):
        
            self.run_id = mlflow.active_run().info.run_id
            
            
            #model.fit(cp.array(X_train), cp.array(y_train))
            model.fit(X_train, y_train)
            
            #y_pred = model.predict(cp.array(X_test))
            y_pred = model.predict(X_test)
            
            mlflow.log_params( self.used_params )
            mlflow.xgboost.log_model(model, "model")
            mlflow.log_table(data=pd.DataFrame(self.features_used), artifact_file="features_used.json")     
            perf, tot, mse, rmse, r2, score, mae = gen_regressor_data(model, X_train, y_train, X_test, y_test, y_pred)
        
        return perf, tot, mse, rmse, r2, score, mae, y_pred
        

