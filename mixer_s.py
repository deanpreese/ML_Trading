
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

def process_model(exp_name, data, models, run_test_size, save_to_mlflow):
        
        run_uuid = str(uuid.uuid1())[:6]
        
        features_list = []
        all_predict_data = pd.DataFrame()
        estimator_perf = []
        estimator_run_ids = []     
                
        for f, e in enumerate(models):
                
                fl_out = []
                
                model_run_uuid = run_uuid + "-"+ str(uuid.uuid1())[:6]
                modelname = e.__class__.__name__
                
                #num_columns = len(data.columns)
                #input_features = num_columns - 2
                
                X = data
                X = X.drop(columns=['output', 'outputC'])
                y = data['output'].values
                fl_out = list(X.columns)
              
                X_train, X_test, y_train, y_test = simple_split_and_scale(X, y, run_test_size, 42)
                
                run_id, perf, tot, mse, rmse, r2, score, mae, predictions = track_regressor_model(modelname, X_train.columns, exp_name, True, e, X_train, 
                                                                                  y_train, X_test, y_test, save_to_mlflow)  
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

        perf_data_t = [run_uuid, 0, e_perf.values.tolist(), features_list, 
                       correctX, correctY, correctP, totalX, cxp, cyp, cpp, mse, rmse, r2, mae]
        
        return perf_data_t    



def run_models(data, estimators, run_test_size, save_to_mlflow ):
        
        p_df = pd.DataFrame()
        
        time_stamp = dte_time.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        exp_name = f"mixer_runs_{time_stamp}"
        
        experiment_id = 0
        
        if save_to_mlflow :
                try:
                        experiment_id = mlflow.create_experiment(exp_name)
                except Exception as e:
                        print(f"{e}")    
                experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id        
                
        perf_data = []
                              
        perf_data_t = process_model(experiment_id, data, estimators, run_test_size, save_to_mlflow)
        perf_data.append(perf_data_t)
        
        p_df = pd.DataFrame(perf_data)    
        p_df.columns = ["rid", "input_features", "e_perf", "features_list", "correctX", "correctY", 
                        "correctP", "totalX", "cxp", "cyp", "cpp", "mse", "rmse", "r2", "mae"]
        p_df.sort_values(by=['cpp'], ascending=False, inplace=True)

        if save_to_mlflow :

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

datafile = [ 
        'data/buildSeqInd_Lucky13_5M_3070.csv',   #0
        'data/buildSeqInd_Lucky13_5M_ALL.csv',  #1
        'data/buildSeqInd_Lucky13_F.csv',  #2
        'data/buildSeqInd_Lucky13_D.csv',  #3
        'data/buildSeqInd_Lucky13_F_3070.csv',  #4
    ]


dtx = pd.read_csv(datafile[4])


xgr = xgr_param_set()
lbr = lbr_param_set()
cbr = cbr_param_set()
xg_rf = xgb_rf_params()

cat_params_F={'learning_rate': 0.0360944196001379, 'depth': 10, 
        'subsample': 0.3523958110464825, 'colsample_bylevel': 0.6176118972551982, 
                'min_data_in_leaf': 46 }

xgb_params_F={'learning_rate': 0.004023993590803149, 'max_depth': 9, 'subsample': 0.5061891892307074, 
'colsample_bytree': 0.6646068031525607, 'min_child_weight': 18}

lgb_params_F={'learning_rate': 0.006961479110933946, 'num_leaves': 762, 
'subsample': 0.5909033731294365, 'colsample_bytree': 0.8383929309109572, 
'min_data_in_leaf': 80}

xgb_params_M={'learning_rate': 0.00264122394857379, 'max_depth': 8, 
'subsample': 0.2772844546321145, 'colsample_bytree': 0.8118319429046319, 
'min_child_weight': 7}

lgb_params_M={'learning_rate': 0.004818774485749822, 'num_leaves': 9, 'subsample': 0.8313397546109982, 
'colsample_bytree': 0.6285174849150702, 'min_data_in_leaf': 68}

cat_params_M={'learning_rate': 0.012193433669679433, 'depth': 7, 'subsample': 0.8003609726402594, 
'colsample_bylevel': 0.9066114272514963, 'min_data_in_leaf': 34}

est_list = [ XGBRFRegressor(), XGBRFRegressor(),  
            XGBRegressor(**xgr),  CatBoostRegressor(**cbr) ,
             XGBRegressor(), LGBMRegressor(**lbr)  ]

est_list = [ XGBRFRegressor(), XGBRFRegressor(), 
            XGBRegressor(**xgr), XGBRegressor(),  
            CatBoostRegressor(**cbr) ,  CatBoostRegressor(), 
            LGBMRegressor(**lbr), LGBMRegressor() 
            ]

est_list = [ XGBRegressor(), XGBRegressor(**xgr), 
            XGBRFRegressor(), XGBRFRegressor(**xg_rf),  
            CatBoostRegressor(**cbr) ,  CatBoostRegressor(), 
            LGBMRegressor(**lbr), LGBMRegressor() 
            ]

est_list = [ XGBRegressor(**xgr), 
            XGBRFRegressor(**xg_rf),  
            CatBoostRegressor(**cbr),
            LGBMRegressor(**lbr) 
            ]

est_list = [ XGBRegressor(), 
            XGBRFRegressor(),  
            CatBoostRegressor(),
            LGBMRegressor() 
            ]


est_list = [ XGBRegressor(),  XGBRegressor(**xgr),  
             XGBRegressor(),  XGBRegressor(**xgr),
             XGBRegressor(),  XGBRegressor(**xgr)   
              ]

est_list = [  XGBRegressor(),  XGBRegressor(),  XGBRegressor() ,
              XGBRegressor(),  XGBRegressor(), XGBRegressor()  ]



#est_list = [ XGBRegressor(),  XGBRegressor(),  CatBoostRegressor() ,
#              XGBRegressor(),  XGBRegressor(), LGBMRegressor()  ]

#est_list = [ XGBRFRegressor(), XGBRFRegressor(),  
#            XGBRegressor(**xgr),  CatBoostRegressor(**cbr) ,
#             XGBRegressor(), LGBMRegressor(**lbr)  ]

est_list = [ CatBoostRegressor(**cbr), LGBMRegressor(**lbr),  
              XGBRegressor(**xgr), XGBRFRegressor(**xg_rf),
              CatBoostRegressor(), LGBMRegressor(),
              XGBRegressor(), XGBRFRegressor()
             ]



split_test_size_value = 0.7          
save_mlflow = False

p_df, experiment_id_parent = run_models(dtx, est_list, split_test_size_value, save_mlflow)

print("")
for run_uuid, input_features, e_perf, features_list, correctX, correctY, correctP, totalX, cxp, cyp, cpp, mse, rmse, r2, mae in p_df.values.tolist(): 
        print(f"{run_uuid}  {correctX}  {correctY}  {correctP}  {totalX}  {cxp}  {cyp}  {cpp}  {mse}  {rmse}  {r2}  {mae}")

print("")
