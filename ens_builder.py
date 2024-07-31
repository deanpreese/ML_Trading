import datetime as dte_time
import random as rand
import uuid
#import warnings
import mlflow
import pandas as pd
import logging

logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)

from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor
from lightgbm  import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from ml_model.model_tracking import save_reg_ens_data
import ml_model.model_params as mp
import ml_model.model_func as model_processing


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
        perf_data_t = model_processing.process_models(experiment_id, data, estimators, run_test_size, save_to_mlflow, feat_data, False, 0)
        
        perf_data.append(perf_data_t)
        
        p_df = pd.DataFrame(perf_data)    
        p_df.columns = ["rid", "input_features", "e_perf", "features_list", "correctX", "correctY", 
                        "correctP", "totalX", "cxp", "cyp", "cpp", "mse", "rmse", "r2", "mae"]
        p_df.sort_values(by=['cpp'], ascending=False, inplace=True)

        print(" ")
        
        for x in range(len(p_df["e_perf"][0])):
                print(f"{p_df['e_perf'][0][x][0]}  {p_df['e_perf'][0][x][1]}  {p_df['e_perf'][0][x][3]}  {p_df['e_perf'][0][x][4]}  {p_df['e_perf'][0][x][5]}" )     

        if save_to_mlflow :
            save_reg_ens_data(p_df)
                               
        return p_df, experiment_id        
                
# ---------------------------
#
# Run the models
#
# ---------------------------

def run():

        est_list_base = [ 
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
        est_list_lgb = [ 
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
                XGBRegressor(**mp.xgb_params_F), 
                XGBRegressor(**mp.xgb_params_M), 
                XGBRegressor(**mp.xgbr_set),   
                XGBRegressor(**mp.xgb_3070), 
                XGBRegressor(**mp.xgb_p),   
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
                #LGBMRegressor(**mp.lgb_3070), 
                LGBMRegressor(**mp.lbr_set), 
                CatBoostRegressor(**mp.cat_3070), 
                CatBoostRegressor(**mp.cbr_set),
                
                #XGBRegressor(**mp.xgb_params_F), 
                #XGBRegressor(**mp.xgb_params_M), 
                XGBRegressor(**mp.xgbr_set),   
                #XGBRegressor(**mp.xgb_3070), 
                XGBRegressor(**mp.xgb_p),   
                
                #XGBRFRegressor(),
                XGBRFRegressor(**mp.xgbrf_t),
                #XGBRFRegressor(**mp.xgbrf_F),
                #XGBRFRegressor(**mp.xgbrf_D),
                XGBRFRegressor(**mp.xgbrf_set),                       
                
        ]


        datafile = [ 
                'data/Lucky13_3070_oos.csv',   
                'data/Lucky13_3070.csv',  #1
                'data/ndata_diff_lucky13_3070_oos.csv', 
                'data/ndata_diff_lucky13_3070.csv', #3
                'data/ndata_lag_3070_oos.csv', 
                'data/ndata_lag_3070.csv', #5
        ]


        dtx = pd.read_csv(datafile[1])

        lucky13 = [
                #'SDLR310',
                #'SDBB91',
                #'SDKC91',
                'SDKC9',
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




        feat_data = lucky13
        #feat_data = 'xxx'
        split_test_size_value = 0.7          
        save_mlflow = False
                
        p_df, experiment_id_parent = run_models(dtx, est_list, split_test_size_value, save_mlflow, feat_data)

        print("")
        for run_uuid, input_features, e_perf, features_list, correctX, correctY, correctP, totalX, cxp, cyp, cpp, mse, rmse, r2, mae in p_df.values.tolist(): 
                print(f"{run_uuid}  {cpp}  {mse}  {rmse} {mae} {r2}  ")

        print("")


if __name__ == "__main__":
    run()

