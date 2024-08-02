
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

import pandas as pd
from enum import Enum
import datetime as dte_time

from ml_model.model_stats import gen_reg_stats

import logging
logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)
logging.getLogger('mlflow.tracking._tracking_service.client').setLevel(logging.ERROR)
logging.getLogger('mlflow.models.model').setLevel(logging.ERROR)

import mlflow
mlflow.set_tracking_uri(uri="http://10.0.0.50:8888")

def train_classifier_model(model_name, features_used, experiment_id, nested, model, X_train, y_train, X_test, y_test, save_to_mlflow):
    
    run_id = 0
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)
    
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_pred, y_test)
    recall = recall_score(y_pred, y_test)
    tot = TN + FP + FN + TP
            
    if save_to_mlflow :
        with mlflow.start_run(experiment_id = experiment_id, nested=nested):
            
            run_id = mlflow.active_run().info.run_id  
            if "XGB" in model_name :
                mlflow.xgboost.log_model(model, "model") 
                p2 = model.get_xgb_params()
                mlflow.log_params(p2)
                mlflow.log_param("ModelName" , model_name)
                
            if "LGB" in model_name :
                mlflow.lightgbm.log_model(model, "model")
                p = model.get_params()
                mlflow.log_params(p)
                mlflow.log_param("ModelName" , model_name)
                
            if "Cat" in model_name :
                mlflow.catboost.log_model(model, "model")
                p = model.get_all_params()
                mlflow.log_params(p)
                mlflow.log_param("ModelName" , model_name)
            
            mlflow.log_table(data=pd.DataFrame(features_used), artifact_file="features_used.json") 
            mlflow.log_param("FeatureCount" , (X_train.shape[1]))
            mlflow.log_metric('Accuracy', accuracy)
            mlflow.log_metric('Precision', precision)
            mlflow.log_metric('Recall', recall)
            mlflow.log_metric('TrueNeg', TN)
            mlflow.log_metric("FalsePos", FP)
            mlflow.log_metric("FalseNeg", FN)
            mlflow.log_metric("TruePos", TP)
            mlflow.log_metric("Perf", accuracy)

    return run_id, accuracy, y_pred, pred_proba   
    #return run_id, accuracy, precision, recall, TN/tot, FP/tot, FN/tot, TP/tot, tot, y_pred, pred_proba      
    

# -----------------------------------------------------
def train_regressor_model(model_name, features_used, experiment_id, nested, model, X_train, y_train, X_test, y_test, save_to_mlflow):
    
    run_id = 0
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = rmse =  rmse = mse**.5
    r2 =r2_score(y_test, y_pred)
    score = model.score(X_test, y_test)
    mae = float(mean_absolute_error(y_test,y_pred))                
    perf, total, mse, rmse, mae = gen_reg_stats(y_test, y_pred)        
    
    if save_to_mlflow :
        with mlflow.start_run(experiment_id = experiment_id, nested=nested):
            
            run_id = mlflow.active_run().info.run_id
            
            if "XGB" in model_name :
                mlflow.xgboost.log_model(model, "model") 
                p2 = model.get_xgb_params()
                mlflow.log_params(p2)
                mlflow.log_param("ModelName" , model_name)
                
            if "LGB" in model_name :
                mlflow.lightgbm.log_model(model, "model")
                p = model.get_params()
                mlflow.log_params(p)
                mlflow.log_param("ModelName" , model_name)
                
            if "Cat" in model_name :
                mlflow.catboost.log_model(model, "model")
                p = model.get_all_params()
                mlflow.log_params(p)
                mlflow.log_param("ModelName" , model_name)
            
            mlflow.log_table(data=pd.DataFrame(features_used), artifact_file="features_used.json")  
            mlflow.log_param("FeatureCount" , (X_train.shape[1]))
            
            mlflow.log_metric('MSE', mse)
            mlflow.log_metric('RMSE', rmse)
            mlflow.log_metric('R2', r2)
            mlflow.log_metric('Score', score)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("Perf", perf)
            mlflow.log_metric("Total", total)
               
    
    return run_id, perf, y_pred        
    #return run_id, perf, total, mse, rmse, r2, score, mae, y_pred    


# -----------------------------------------------------
def save_reg_ens_data(ens_perf_df):
    
    step = 0
    time_stamp = dte_time.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    exp_name = f"mixer_output_{time_stamp}"
    
    try:
        experiment_id = mlflow.create_experiment(exp_name)
    except Exception as e:
        print(f"{e}")    
            
    experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id        

    for run_uuid, input_features, e_perf, features_list, correctX, correctY, correctP, totalX, cxp, cyp, cpp, mse, rmse, r2, mae in ens_perf_df.values.tolist() :
    
        with mlflow.start_run(experiment_id = experiment_id, nested=False): 
                        
            #mlflow.log_param('FeatureCount', input_features)
            mlflow.log_param('run_uuid', run_uuid)
            #mlflow.log_metric('FeatureCount', input_features, step)
            #mlflow.log_metric('correctX', correctX, step)
            #mlflow.log_metric('correctP', correctP, step)
            #mlflow.log_metric('correctY', correctY, step)
            mlflow.log_metric('totalX', totalX, step)
            mlflow.log_metric('cxp', cxp, step)
            mlflow.log_metric("cyp", cyp, step)
            mlflow.log_metric("cpp", cpp, step)

            mlflow.log_metric('MSE', mse, step)
            mlflow.log_metric('RMSE', rmse, step)
            mlflow.log_metric('R2', r2, step)
            mlflow.log_metric('Score', r2, step)
            mlflow.log_metric("MAE", mae, step)
            mlflow.log_metric("Perf", cpp, step)
            mlflow.log_metric("Total", totalX, step)
                            
            
            mlflow.log_table(data=pd.DataFrame(e_perf), artifact_file="all_perf_data.json")        
            step += 1 