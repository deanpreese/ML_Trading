import uuid
#import warnings
import mlflow
import pandas as pd
import random as rand

from ml_model.model_tracking import train_regressor_model, train_classifier_model
from ml_model.model_stats import calc_mse_rmse_mae, calc_ensemble_results
from ml_model.data_func import simple_split_and_scale
from sklearn.metrics import r2_score

import logging
logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)


def process_models(exp_name, data, models, run_test_size, save_to_mlflow, feat_data, do_random, random_size):
        
        run_uuid = str(uuid.uuid1())[:6]
        
        features_list = []
        all_predict_data = pd.DataFrame()
        estimator_perf = []
        estimator_run_ids = []     
                
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
                
              
                X_train, X_test, y_train, y_test = simple_split_and_scale(X, y, run_test_size, 42)
                
                modelname = modelname + "V2"
                
                if "Regressor" in modelname:
                        r_run_id, r_perf, r_predictions = train_regressor_model(modelname, X_train.columns, exp_name, True, e, X_train, 
                                                                                        y_train, X_test, y_test, save_to_mlflow)  
                        all_predict_data[model_run_uuid] = r_predictions
                        all_predict_data[model_data] = r_predictions * r_perf
                        all_predict_data[model_type] = "Regressor"
                        estimator_run_ids.append(model_run_uuid)
                        outputs = [ modelname, r_perf, fl_out, r_run_id ]        
                        estimator_perf.append(outputs)  
                
                if "Classifier" in modelname:           
                        c_run_id, accuracy, c_predictions, pred_proba = train_classifier_model(modelname, X_train.columns, exp_name, True, e, X_train, 
                                                                                        y_train, X_test, y_test, save_to_mlflow)  
                        all_predict_data[model_run_uuid] = c_predictions
                        all_predict_data[model_data] = c_predictions * accuracy
                        all_predict_data[model_type] = "Classifier"
                        estimator_run_ids.append(model_run_uuid)
                        outputs = [ modelname, round(accuracy,4), fl_out, c_run_id ]        
                        estimator_perf.append(outputs) 

        all_predict_data["target"] = y_test
        e_perf = pd.DataFrame(estimator_perf)        
        e_perf.columns = ["Estimator", "Perf", "Features", "RUN_ID" ]
        
        #correctX, correctY, correctP, totalX, cxp, cyp, cpp, r_predictions, r_y_target = calc_reg_ens_results(all_predict_data, estimator_run_ids)
        correctX, correctY, correctP, totalX, cxp, cyp, cpp, r_predictions, r_y_target = calc_ensemble_results(all_predict_data, estimator_run_ids)

        mse, rmse, mae = calc_mse_rmse_mae(r_y_target, r_predictions)
        r2 =r2_score(r_y_target, r_predictions)

        perf_data_t = [run_uuid, 0, e_perf.values.tolist(), features_list, 
                       correctX, correctY, correctP, totalX, cxp, cyp, cpp, mse, rmse, r2, mae]
        
        return perf_data_t    