import uuid
import mlflow
import pandas as pd
import random as rand
from catboost import  Pool
import lightgbm as lgb
import xgboost as xgb
import datetime as dte_time


import logging
logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)
logging.getLogger('mlflow.tracking._tracking_service.client').setLevel(logging.ERROR)
logging.getLogger('mlflow.models.model').setLevel(logging.ERROR)
logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)
mlflow.set_tracking_uri(uri="http://10.0.0.50:8888")


from sklearn.metrics import mean_absolute_error,r2_score, root_mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

from ml_model.model_stats import calc_mse_rmse_mae, calc_ensemble_results
from ml_model.data_func import simple_split_and_scale,split_three_ways_full
from ml_model.model_stats import gen_reg_stats, gen_reg_stats_x

# -----------------------------------------------------
def process_models(exp_name, data, models, run_test_size, save_to_mlflow, feat_data, do_random, random_size):
        
        run_uuid = str(uuid.uuid1())[:6]
        
        features_list = []
        all_predict_data = pd.DataFrame()
        estimator_perf = []
        estimator_run_ids = []     
        output_text = []               
                
        for f, e in enumerate(models):
                
                fl_out = []
                model_run_uuid = run_uuid + "-"+ str(uuid.uuid1())[:6]
                model_data = f"{model_run_uuid}_p"
                model_type = f"{model_run_uuid}_t"
                modelname = e.__class__.__name__
                
                X = data
                X = X.drop(columns=['output', 'outputC'])
                fl_out = list(X.columns)
                
                if do_random:
                        idxx = rand.sample(range(1, len( X.columns ) ) , random_size)
                        features_list.append(idxx)    
                        fl= features_list[f]
                        X = data.iloc[:, fl]  
                        fl_out = fl                        
                elif feat_data != 'xxx':
                        X = X[feat_data]
                        fl_out = list(X.columns)

                y = data['output'].values
                
                if "Classifier" in modelname:       
                        y = data['outputC'].values                
                
              
                X_train, X_val, X_test, y_train, y_val, y_test = split_three_ways_full(X, y, run_test_size, 0.2, 42)
                #X_train, X_test, y_train, y_test = simple_split_and_scale(X, y, run_test_size, 42)
                
                modelname = modelname + "V2"
                
                if "Regressor" in modelname:

                        out_text, r_run_id, r_perf, r_predictions = train_regressor_model(modelname, X_train.columns, exp_name, True, e, X_train, 
                                                                                        y_train, X_val, y_val, X_test, y_test, save_to_mlflow)  
                        all_predict_data[model_run_uuid] = r_predictions
                        all_predict_data[model_data] = r_predictions * r_perf
                        all_predict_data[model_type] = "Regressor"
                        estimator_run_ids.append(model_run_uuid)
                        outputs = [ modelname, r_perf, fl_out, r_run_id ]        
                        estimator_perf.append(outputs)  
                        
                        output_text.append(out_text)
                
                if "Classifier" in modelname:

                        out_text, c_run_id, accuracy, c_predictions, pred_proba = train_classifier_model(modelname, X_train.columns, exp_name, True, e, X_train, 
                                                                                        y_train, X_val, y_val, X_test, y_test, save_to_mlflow)  
                        all_predict_data[model_run_uuid] = c_predictions
                        all_predict_data[model_data] = c_predictions 
                        all_predict_data[model_type] = "Classifier"
                        estimator_run_ids.append(model_run_uuid)
                        outputs = [ modelname, round(accuracy,4), fl_out, c_run_id ]        
                        estimator_perf.append(outputs) 
                        
                        output_text.append(out_text)

        all_predict_data["target"] = y_test
        e_perf = pd.DataFrame(estimator_perf)        
        e_perf.columns = ["Estimator", "Perf", "Features", "RUN_ID" ]
        
        print(f" ")
        print(f"Running Ensemble Calculations")
        ens_accuracy, ens_precision, ens_recall, win_p, loss_p, tn_p, tp_p, fn_p, fp_p, predictions, scaled_predictions, y_target = calc_ensemble_results(all_predict_data, estimator_run_ids)

        perf_data_t = [run_uuid, 0, e_perf.values.tolist(), features_list, ens_accuracy, ens_precision, ens_recall, win_p, loss_p, tn_p, tp_p, fn_p, fp_p ]
        return perf_data_t, output_text    




# -----------------------------------------------------
def train_classifier_model(model_name, features_used, experiment_id, nested, model, X_train, y_train, X_val, y_val, X_test, y_test, save_to_mlflow):
    
    run_id = 0
    print(" ")
    print(f"Training  {model_name}")            
    print(f"{model.get_params()}")            
    print(" ")              
    
    if "Cat" in model_name:
        val_pool = Pool(X_val, y_val)
        model.fit(X_train, y_train, eval_set=val_pool, early_stopping_rounds=10)
        
    if "LGB" in model_name :        
        model.fit(X_train, y_train, eval_set = [(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=10)] )
        
    if "XGB" in model_name :        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)


    y_pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)
    
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    accuracy = round(accuracy_score(y_pred, y_test),4)
    precision = round(precision_score(y_pred, y_test),4)
    recall = round(recall_score(y_pred, y_test),4)
    tot = TN + FP + FN + TP
            
    out_text = f"Total {tot}  Wins {TP+TN}  Acc: {accuracy}  Precision: {precision}  Recall: {recall}  TP {TP}  TN {TN}  FP {FP}  FN {FN}"        
            
    if save_to_mlflow :
        with mlflow.start_run(experiment_id = experiment_id, nested=nested):
            
            run_id = mlflow.active_run().info.run_id  
            mlflow.MlflowClient().set_tag(run_id, 'mlflow.note.content', features_used)
            
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

    return out_text, run_id, accuracy, y_pred, pred_proba   
    #return run_id, accuracy, precision, recall, TN/tot, FP/tot, FN/tot, TP/tot, tot, y_pred, pred_proba      



# -----------------------------------------------------
def train_regressor_model(model_name, features_used, experiment_id, nested, model, X_train, y_train, X_val, y_val, X_test, y_test, save_to_mlflow):
    
    run_id = 0
    
    print(" ")
    print(f"Training  {model_name}")            
    print(f"{model.get_params()}")            
    print(" ")              
    
    if "Cat" in model_name:
        val_pool = Pool(X_val, y_val)
        model.fit(X_train, y_train, eval_set=val_pool, early_stopping_rounds=10)
        
    if "LGB" in model_name :        
        model.fit(X_train, y_train, eval_set = [(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=10)] )
        
    if "XGB" in model_name :        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mse = rmse **2.0
    r2 =r2_score(y_test, y_pred)
    score = model.score(X_test, y_test)
    mae = float(mean_absolute_error(y_test,y_pred))                
    
    wins, perf, total, msex, rmsex, maex, r2x = gen_reg_stats_x(y_test, y_pred)        
    out_text = f"Total {total}  Wins {wins}  Perf {perf}  MSE {mse} RMSE {rmse}  MAE {mae}  R2 {r2} "
    
    if save_to_mlflow :
        with mlflow.start_run(experiment_id = experiment_id, nested=nested):
            
            run_id = mlflow.active_run().info.run_id
            mlflow.MlflowClient().set_tag(run_id, 'mlflow.note.content', features_used)
            
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
               
    
    return out_text, run_id, perf, y_pred        
    #return run_id, perf, total, mse, rmse, r2, score, mae, y_pred    


# -----------------------------------------------------
def save_reg_ens_data(ens_perf_df, exp_description=""):
    
    step = 0
    time_stamp = dte_time.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    exp_name = f"mixer_output_{time_stamp}"
    experiment_id = ""
    
    tags={'mlflow.note.content':exp_description}
    experiment_id = mlflow.create_experiment(exp_name, tags=tags)
    new_exp_name = f"mixer_output_{experiment_id}"

        
    mlflow.MlflowClient().rename_experiment(experiment_id, new_exp_name)

    for run_uuid, input_features, e_perf, features_list, ens_accuracy, ens_precision, ens_recall, win_p, loss_p, tn_p, tp_p, fn_p, fp_p  in ens_perf_df.values.tolist(): 
    
        with mlflow.start_run(experiment_id = experiment_id, nested=False): 
                        
            #mlflow.log_param('FeatureCount', input_features)
            mlflow.log_param('run_uuid', run_uuid)
            #mlflow.log_metric('FeatureCount', input_features, step)
            mlflow.log_metric('Accuracy', ens_accuracy)
            mlflow.log_metric('Precision', ens_precision)
            mlflow.log_metric('Recall', ens_recall)
            mlflow.log_metric('TrueNegPct', tn_p)
            mlflow.log_metric("TruePosPct", tp_p)                            
            mlflow.log_metric("FalsePosPct", fp_p)
            mlflow.log_metric("FalseNegPct", fn_p)
            
            mlflow.log_table(data=pd.DataFrame(e_perf), artifact_file="all_perf_data.json")        
            step += 1 

