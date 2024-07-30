import datetime as dte_time
import random as rand
import uuid
#import warnings
import mlflow
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
import logging

logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)

from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor
from lightgbm  import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from ml_model.model_tracking import track_regressor_model
from ml_model.model_stats import gen_reg_stats, calc_reg_ens_results, calc_mse_rmse_mae
from ml_model.data_func import simple_split_and_scale
import ml_model.model_params as mp


def process_model(exp_name, data, models, run_test_size, save_to_mlflow, feat_data):
        
        run_uuid = str(uuid.uuid1())[:6]
        
        features_list = []
        all_predict_data = pd.DataFrame()
        estimator_perf = []
        estimator_run_ids = []     
                
        for f, e in enumerate(models):
                
                fl_out = []
                model_run_uuid = run_uuid + "-"+ str(uuid.uuid1())[:6]
                modelname = e.__class__.__name__
                
                X = data
                X = X.drop(columns=['output', 'outputC'])
                
                if feat_data != 'xxx':
                        X= X[feat_data]
                                
                y = data['output'].values
                fl_out = list(X.columns)
              
                X_train, X_test, y_train, y_test = simple_split_and_scale(X, y, run_test_size, 42)
                
                run_id, perf, tot, mse, rmse, r2, score, mae, predictions = track_regressor_model(modelname, X_train.columns, exp_name, True, e, X_train, 
                                                                                  y_train, X_test, y_test, save_to_mlflow)  
                all_predict_data[model_run_uuid] = predictions
                perf, total, mse, rmse, mae = gen_reg_stats(y_test, predictions)
                
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

        mse, rmse, mae = calc_mse_rmse_mae(r_y_target, r_predictions)
        r2 =r2_score(r_y_target, r_predictions)

        perf_data_t = [run_uuid, 0, e_perf.values.tolist(), features_list, 
                       correctX, correctY, correctP, totalX, cxp, cyp, cpp, mse, rmse, r2, mae]
        
        return perf_data_t    



def run_models(data, estimators, run_test_size, save_to_mlflow, feat_data ):
        
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
                              
        perf_data_t = process_model(experiment_id, data, estimators, run_test_size, save_to_mlflow, feat_data)
        perf_data.append(perf_data_t)
        
        p_df = pd.DataFrame(perf_data)    
        p_df.columns = ["rid", "input_features", "e_perf", "features_list", "correctX", "correctY", 
                        "correctP", "totalX", "cxp", "cyp", "cpp", "mse", "rmse", "r2", "mae"]
        p_df.sort_values(by=['cpp'], ascending=False, inplace=True)

        print(" ")
        
        for x in range(len(p_df["e_perf"][0])):
                print(f"{p_df['e_perf'][0][x][0]}  {p_df['e_perf'][0][x][1]}  {p_df['e_perf'][0][x][3]}  {p_df['e_perf'][0][x][4]}  {p_df['e_perf'][0][x][5]}" )     

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

def run():

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

        # 87_FI data        
        est_list_66 = [ 
                XGBRegressor(),XGBRegressor(mp.xgbr_set), 
                XGBRegressor(mp.xgb_params_F), XGBRegressor(mp.xgb_params_M), 
                CatBoostRegressor(), CatBoostRegressor(mp.cbr_set),
                CatBoostRegressor(mp.cat_params_F), CatBoostRegressor(mp.cat_params_M),
                LGBMRegressor(), LGBMRegressor(mp.lbr_set), 
                LGBMRegressor(mp.lgb_params_F), LGBMRegressor(mp.lgb_params_M), 
                XGBRFRegressor(),XGBRFRegressor(mp.xgbrf_set),
        ]

        est_list_xgbrf = [
                XGBRFRegressor(),
                XGBRFRegressor(**mp.xgbrf_t),
                XGBRFRegressor(**mp.xgbrf_F),
                XGBRFRegressor(**mp.xgbrf_D),
                XGBRFRegressor(**mp.xgbrf_set),                
        ]

        est_list_xgb = [ 
                XGBRegressor(**mp.xgb_params_M), 
                XGBRegressor(),   
                XGBRegressor(**mp.xgb_3070),  
                XGBRFRegressor(**mp.xgbrf_set),
                XGBRegressor(**mp.xgbr_set),  
        ]

        est_list_lgb = [ 
                LGBMRegressor(**mp.lgb_3070), 
                LGBMRegressor(), 
                LGBMRegressor(**mp.lbr_set), 
                LGBMRegressor(**mp.lgb_params_F), 
                LGBMRegressor(**mp.lgb_params_M), 
        ]


        est_list_cat = [ 
                CatBoostRegressor(**mp.cat_3070), 
                CatBoostRegressor(),  
                CatBoostRegressor(**mp.cbr_set),  
                CatBoostRegressor(**mp.cat_params_F), 
                CatBoostRegressor(**mp.cat_params_M),
        ]


        est_list = [ 
                LGBMRegressor(**mp.lgb_3070), 
                XGBRegressor(**mp.xgb_3070),  
                CatBoostRegressor(**mp.cat_3070), 
                XGBRFRegressor(**mp.xgbrf_set),
                
                #XGBRegressor(),   
                #XGBRegressor(**xgr),  
                #ßXGBRegressor(**xgb_params_F), 
                #XGBRegressor(**xgb_params_M), 
                #CatBoostRegressor(),  
                #CatBoostRegressor(**cbr),  
                #CatBoostRegressor(**cat_params_F), 
                #CatBoostRegressor(**cat_params_M),
                #LGBMRegressor(), 
                #LGBMRegressor(**lbr), 
                #LGBMRegressor(**lgb_params_F), 
                #LGBMRegressor(**lgb_params_M), 
                #XGBRFRegressor(),
        ]


        datafile = [ 
                'data/Lucky13_3070_oos.csv',   
                'data/Lucky13_3070.csv',  #1
                'data/ndata_diff_lucky13_3070_oos.csv', 
                'data/ndata_diff_lucky13_3070.csv', #3
                'data/ndata_lag_3070_oos.csv', 
                'data/ndata_lag_3070.csv', #5
        ]


        dtx = pd.read_csv(datafile[5])

        lucky13 = [
                #'SDLR310',
                #'SDBB91',
                #'SDKC91',
                #'SDKC9',
                'ROC',
                #'ATR34',
                #'ATR32',
                #'ATR31',
                'ATR3',
                #'ATR21',
                'ATR2',
                'RSI',
                'STOK1'
                ]

 
        #data/ndata_3070.csv', #6
        ndata_all_feat = [
                #'SDLR3102', 'SDLR3101', 
                # 'SDLR310', 
                #'VOLMA72', 'VOLMA71',
                #'VOLMA7', 
                #'VOLMA132', 'VOLMA131', 
                #'VOLMA13',
                #'ZH212', 'ZH211', 'ZH21', 'ZH92', 'ZH91', 
                #'ZH9', 
                #'ZL212', 'ZL211', 'ZL21', 'ZL92', 'ZL91', 
                #'ZL9', 
                #'ZC212', 'ZC211', 'ZC21', 'ZC92', 'ZC91', 
                #'ZC9', 
                #'SDBB92','SDBB91', 
                # 'SDBB9', 
                #'SDBB202', 'SDBB201', 
                # 'SDBB20',
                #'SDKC92', 
                #'SDKC91', 
                'SDKC9', 
                #'SDKC72', 'SDKC71', 
                'SDKC7', 
                #'ROC142', 'ROC141', 
                #'ROC14', 
                #'ROC132', 'ROC131',
                #'ROC13', 
                #'ROC92', 'ROC91', 
                 'ROC', 
                #'ATR144', 'ATR143', 'ATR142', 'ATR141', 
                 #'ATR14', 
                #'ATR74', 'ATR73', 'ATR72', 'ATR71', 
                'ATR7', 
                #'ATR54', 'ATR53', 'ATR52', 'ATR51', 
                'ATR5',
                'ATR34', 'ATR33', 'ATR32', 'ATR31', 
                'ATR3', 
                #'ATR24', 'ATR23', 'ATR22', 'ATR21', 
                'ATR2', 
                #'RSI142', 'RSI141', 
                #'RSI14', 
                
                #'RSI92', 'RSI91', 
                'RSI', 
                
                #'RSI72', 'RSI71', 
                'RSI7', 
                #'RSI52', 'RSI51', 
                #'RSI5', 
                #'RSI32', 'RSI31', 
                #'RSI3', 
               
                #'STOK7142', 'STOK7141', 
                'STOK714',
                
                #'STOK7212', 'STOK7211', 
                'STOK1', 
                
                #'STOK52', 'STOK51', 
                #'STOK5'
        ]




        feat_data = 'xxx'
        split_test_size_value = 0.7          
        save_mlflow = False
                
        p_df, experiment_id_parent = run_models(dtx, est_list_base, split_test_size_value, save_mlflow, feat_data)

        print("")
        for run_uuid, input_features, e_perf, features_list, correctX, correctY, correctP, totalX, cxp, cyp, cpp, mse, rmse, r2, mae in p_df.values.tolist(): 
                print(f"{run_uuid}  {cpp}  {mse}  {rmse} {mae} {r2}  ")

        print("")


if __name__ == "__main__":
    run()

