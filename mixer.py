
import datetime as dte_time
import random as rand
import uuid
#import warnings
import mlflow
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import logging

from ml_model.model_tracking import track_regressor_model
from ml_model.model_stats import gen_reg_stats, calc_reg_ens_results
from ml_model.data_func import simple_split_and_scale
from ml_model.model_params import xgr_param_set, lbr_param_set, cbr_param_set, xgr_param_set2, xgb_rf_params

logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)

from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor
from lightgbm  import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

import ml_model.model_params as mp


def process_model(exp_name, data, models, run_test_size, feature_list_size):
        
        run_uuid = str(uuid.uuid1())[:6]
        
        features_list = []
        all_predict_data = pd.DataFrame()
        estimator_perf = []
        estimator_run_ids = []     
                
        for f, e in enumerate(models):
                
                fl_out = []
                
                model_run_uuid = run_uuid + "-"+ str(uuid.uuid1())[:6]
                modelname = e.__class__.__name__
                
                print(f"Running {modelname} with {feature_list_size} features")
                
                num_columns = len(data.columns)
                input_features = num_columns - 2
                X = data.iloc[:, 0:input_features]
                y = data['output'].values
                
                idxx = rand.sample(range(1, len(data.axes[1]) -2 ), feature_list_size)
                features_list.append(idxx)    
                fl= features_list[f]
                X = data.iloc[:, fl]  
                input_features = len(X.axes[1]) 
                fl_out = fl
              
                X_train, X_test, y_train, y_test = simple_split_and_scale(X, y, run_test_size, 42)
                
                run_id, perf, tot, mse, rmse, r2, score, mae, predictions = track_regressor_model(modelname, X_train.columns, exp_name, True, e, X_train, 
                                                                                  y_train, X_test, y_test, True)  
                all_predict_data[model_run_uuid] = predictions
                perf, tot = gen_reg_stats(y_test, predictions)
                
                mse = mean_squared_error(y_test, predictions)
                rmse =  rmse = mse**.5
                score = e.score(X_test, y_test)
                
                combined_prod_perf = predictions * perf
                nm = f"{model_run_uuid}_p"
                all_predict_data[nm] = combined_prod_perf
                
                estimator_run_ids.append(model_run_uuid)
                
                outputs = [ modelname, perf, tot, mse, rmse, score, fl_out, model_run_uuid, run_id ]        
                estimator_perf.append(outputs)  

        all_predict_data["target"] = y_test
        e_perf = pd.DataFrame(estimator_perf)        
        e_perf.columns = ["Estimator", "Perf", "Total", "MSE", "RMSE", "Score", "Features", "UUID", "RUN_ID" ]
        
        correctX, correctY, correctP, totalX, cxp, cyp, cpp, r_predictions, r_y_target = calc_reg_ens_results(all_predict_data, estimator_run_ids)

        mse = mean_squared_error(r_y_target, r_predictions, squared=True)
        rmse =mean_squared_error(r_y_target, r_predictions, squared=False)
        r2 =r2_score(r_y_target, r_predictions)
        score = r2
        mae = float(mean_absolute_error(r_y_target,r_predictions))                

        perf_data_t = [run_uuid, feature_list_size, e_perf.values.tolist(), features_list, 
                       correctX, correctY, correctP, totalX, cxp, cyp, cpp, mse, rmse, r2, mae]
        
        return perf_data_t    



def run_models(data, estimators, run_test_size, 
               min_features, max_features, step_features, total_cycles ):
        
        feature_list_size = min_features
        p_df = pd.DataFrame()
        
        if len(data.columns) < max_features:
                max_features = len(data.columns) - 3
        
        time_stamp = dte_time.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        exp_name = f"mixer_runs_{time_stamp}"
        
        try:
            experiment_id = mlflow.create_experiment(exp_name)
        except Exception as e:
            print(f"{e}")    
            experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id        
       
                
        perf_data = []
        
        for q in range(total_cycles):
                for f in range(min_features, max_features, step_features):
                        feature_list_size = f                        
                        perf_data_t = process_model(experiment_id, data, estimators, run_test_size, feature_list_size)
                        perf_data.append(perf_data_t)
                
                p_df = pd.DataFrame(perf_data)    
                p_df.columns = ["rid", "input_features", "e_perf", "features_list", "correctX", "correctY", 
                                "correctP", "totalX", "cxp", "cyp", "cpp", "mse", "rmse", "r2", "mae"]
                p_df.sort_values(by=['cpp'], ascending=False, inplace=True)

        step = 0
        time_stamp = dte_time.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        exp_name = f"mixer_output_{time_stamp}"
        
        try:
           experiment_id = mlflow.create_experiment(exp_name)
        except Exception as e:
            print(f"{e}")    
            experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id        
       
        for run_uuid, input_features, e_perf, features_list, correctX, correctY, correctP, totalX, cxp, cyp, cpp, mse, rmse, r2, mae in p_df.values.tolist() :
        
                with mlflow.start_run(experiment_id = experiment_id, nested=False): 
                                
                        mlflow.log_param('FeatureCount', input_features)
                        mlflow.log_param('run_uuid', run_uuid)
                        mlflow.log_metric('FeatureCount', input_features, step)
                        mlflow.log_metric('correctX', correctX, step)
                        mlflow.log_metric('correctP', correctP, step)
                        mlflow.log_metric('correctY', correctY, step)
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
                               
        return p_df, experiment_id        
                
# ---------------------------
#
# Run the models
#
# ---------------------------

est_list_base = [ 
        XGBRFRegressor(),
        XGBRFRegressor(**mp.xgbrf_t),
        XGBRFRegressor(**mp.xgbrf_F),
        XGBRFRegressor(**mp.xgbrf_D),
        XGBRFRegressor(**mp.xgbrf_set),
        XGBRegressor(),   
        XGBRegressor(**mp.xgb_3070),  
        XGBRegressor(**mp.xgbr_set),  
        XGBRegressor(**mp.xgb_params_F), 
        XGBRegressor(**mp.xgb_params_M), 
        CatBoostRegressor(),  
        CatBoostRegressor(**mp.cat_3070), 
        CatBoostRegressor(**mp.cbr_set),  
        CatBoostRegressor(**mp.cat_params_F), 
        CatBoostRegressor(**mp.cat_params_M),
        LGBMRegressor(), 
        LGBMRegressor(**mp.lgb_3070), 
        LGBMRegressor(**mp.lbr_set), 
        LGBMRegressor(**mp.lgb_params_F), 
        LGBMRegressor(**mp.lgb_params_M), 
]


datafile = [ 
        'data/buildSeqInd_Lucky13_5M_3070.csv',   #0
        'data/buildSeqInd_Lucky13_5M_ALL.csv',  #1
        'data/buildSeqInd_Lucky13_F.csv',  #2
        'data/buildSeqInd_Lucky13_D.csv',  #3
        'data/buildSeqInd_Lucky13_F_3070.csv',  #4
        'data/ndata_3070.csv', #5
    ]

dtx = pd.read_csv(datafile[5])

split_test_size_value = 0.7          
min_features_used = 6
max_features_used = 12
step_features_used = 1
total_cycles_used = 100

feat_ndata_3070 = ['RSI9', 'RSI7', 'ZH', 'ATR7', 'RSI14', 'ATR3', 'ZL', 'ROC14', 'ATR2', 'ZH9', 'VOLMA13', 'STOK714', 'RSI72', 'ROC9', 'output', 'outputC']
dtx = dtx[feat_ndata_3070]

#Lucky13
# SDLR310,SDBB91,SDKC91,SDKC9,ROC,ATR34,ATR32,ATR31,ATR3,ATR21,ATR2,RSI,STOK1,output,outputC

p_df, experiment_id_parent = run_models(dtx, est_list_base, 
                                        split_test_size_value, min_features_used, max_features_used, 
                                        step_features_used, total_cycles_used  )

print(" ")
print(p_df)                
print(" ")
