
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

import pandas as pd
from enum import Enum

from ml_model.model_stats import gen_reg_stats

import mlflow
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


def track_classifier_model(model_name, features_used, experiment_id, nested, model, X_train, y_train, X_test, y_test):
        
    with mlflow.start_run(experiment_id = experiment_id, nested=nested):
    
        run_id = mlflow.active_run().info.run_id  
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)
        
        if "XGB" in model_name :
            mlflow.xgboost.log_model(model, "model") 
            p = model.get_params()
            p2 = model.get_xgb_params()
            mlflow.log_params(p)
            mlflow.log_params(p2)
               
        if "LGB" in model_name :
            mlflow.lightgbm.log_model(model, "model")
            p = model.get_params()
            mlflow.log_params(p)
            
        if "Cat" in model_name :
            mlflow.catboost.log_model(model, "model")
            p = model.get_all_params()
            mlflow.log_params(p)
        
        mlflow.log_table(data=pd.DataFrame(features_used), artifact_file="features_used.json")                 
        accuracy, precision, recall, TN, FP, FN, TP, tot = gen_classifier_data(model, X_train, y_train, X_test, y_test, y_pred)
        
    return run_id, accuracy, precision, recall, TN, FP, FN, TP, tot, y_pred, pred_proba      
    


# -----------------------------------------------------
def gen_regressor_data(model, X_train, y_train, X_test, y_test, y_pred):
        
        mse = mean_squared_error(y_test, y_pred, squared=True)
        rmse =mean_squared_error(y_test, y_pred, squared=False)
        r2 =r2_score(y_test, y_pred)
        score = model.score(X_test, y_test)
        mae = float(mean_absolute_error(y_test,y_pred))                
        perf, tot = gen_reg_stats(y_test, y_pred)
        
        mlflow.log_param("FeatureCount" , (X_train.shape[1]))
        mlflow.log_metric('MSE', mse)
        mlflow.log_metric('RMSE', rmse)
        mlflow.log_metric('R2', r2)
        mlflow.log_metric('Score', score)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("Perf", perf)
        mlflow.log_metric("Total", tot)
        
        return perf, tot, mse, rmse, r2, score, mae


def track_regressor_model(model_name, features_used, experiment_id, nested, model, X_train, y_train, X_test, y_test):
        
    with mlflow.start_run(experiment_id = experiment_id, nested=nested):
    
        run_id = mlflow.active_run().info.run_id
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if "XGB" in model_name :
            mlflow.xgboost.log_model(model, "model") 
            p2 = model.get_xgb_params()
            mlflow.log_params(p2)
               
        if "LGB" in model_name :
            mlflow.lightgbm.log_model(model, "model")
            p = model.get_params()
            mlflow.log_params(p)
            
        if "Cat" in model_name :
            mlflow.catboost.log_model(model, "model")
            p = model.get_all_params()
            mlflow.log_params(p)
        
        mlflow.log_table(data=pd.DataFrame(features_used), artifact_file="features_used.json")     
        perf, tot, mse, rmse, r2, score, mae = gen_regressor_data(model, X_train, y_train, X_test, y_test, y_pred)
    
    return run_id, perf, tot, mse, rmse, r2, score, mae, y_pred    
