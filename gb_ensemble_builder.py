import datetime as dte_time
import random as rand
import uuid
#import warnings
import mlflow
import pandas as pd
import logging

logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)

from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.svm import SVR

from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor
from lightgbm  import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from ml_model.model_process import save_reg_ens_data
import ml_model.model_params as mp
import ml_model.model_process as model_processing


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
        
        #["Estimator", "Perf", "Features", "RUN_ID" ]
        
        for x in range(len(p_df["e_perf"][0])):
                print(f"{p_df['e_perf'][0][x][0]}  {p_df['e_perf'][0][x][1]} " )     

        if save_to_mlflow :
            save_reg_ens_data(p_df)
                               
        return p_df, experiment_id        
                


def run():
        model_list = []

        monster = [
                LGBMRegressor(**mp.lgb_r_lucky13), 
                CatBoostRegressor(**mp.cat_r_lucky13),
                XGBRegressor(**mp.xgb_r_lucky13),   
                XGBRFRegressor(**mp.xgbrf_r_lucky13),                
                LGBMClassifier(**mp.lgb_c_lucky13),
                XGBClassifier(**mp.xgb_c_lucky13),
                CatBoostClassifier(**mp.cat_c_lucky13),
                XGBRFClassifier(**mp.xgbrf_c_lucky13),
                
                LGBMRegressor(**mp.lgb_r_L13EX), 
                CatBoostRegressor(**mp.cat_r_L13EX),
                XGBRegressor(**mp.xgb_r_L13EX),   
                XGBRFRegressor(**mp.xgbrf_r_L13EX),                
                LGBMClassifier(**mp.lgb_c_L13EX),
                XGBClassifier(**mp.xgb_c_L13EX),
                CatBoostClassifier(**mp.cat_c_L13EX),
                XGBRFClassifier(**mp.xgbrf_c_L13EX),
                
                LGBMRegressor(**mp.lgb_r_set), 
                CatBoostRegressor(**mp.cat_r_set),
                XGBRegressor(**mp.xgb_r_set),   
                XGBRFRegressor(**mp.xgbrf_r_set),                
                LGBMClassifier(**mp.lgb_c_set),
                XGBClassifier(**mp.xgb_c_set),
                CatBoostClassifier(**mp.cat_c_set),

                LGBMRegressor(**mp.lgb_r_t), 
                CatBoostRegressor(**mp.cat_r_t),
                XGBRegressor(**mp.xgb_r_t),   
                XGBRFRegressor(**mp.xgbrf_r_t),                
                LGBMClassifier(**mp.lgb_c_t),
                XGBClassifier(**mp.xgb_c_t),
                CatBoostClassifier(**mp.cat_c_t),
        ]       
        
        #model_list.append(monster)

        baseline = [
                LGBMRegressor(), 
                CatBoostRegressor(),
                XGBRegressor(),   
                XGBRFRegressor(),                
                LGBMClassifier(),
                XGBClassifier(),
                CatBoostClassifier(),
                XGBRFClassifier(),
        ]
       
        ens_r13 = [
                XGBRegressor(**mp.xgb_c_lucky13 ), 
                XGBRFRegressor(**mp.xgbrf_c_lucky13 ), 
                XGBRFRegressor(**mp.xgbrf_r_lucky13 ),
                CatBoostRegressor(**mp.cat_r_lucky13),
                LGBMRegressor(**mp.lgb_r_lucky13),               
        ]

        ens_c13 = [
                XGBClassifier(**mp.xgb_c_lucky13),
                XGBRFClassifier(**mp.xgbrf_c_lucky13),
                LGBMClassifier(**mp.lgb_c_lucky13),
                CatBoostClassifier(**mp.cat_c_lucky13),        
        ]


        ens_x3 = [
                XGBClassifier(**mp.xgb_c_t),
                XGBClassifier(**mp.xgb_c_lucky13),
                XGBRegressor(**mp.xgb_r_lucky13 ), 
                CatBoostClassifier(**mp.cat_c_lucky13),        
                CatBoostRegressor(**mp.cat_r_lucky13),
        ]

        ens_342 = [
                XGBClassifier(**mp.xgb_c_lucky13),
                XGBRegressor(**mp.xgb_r_lucky13 ), 
                CatBoostClassifier(**mp.cat_c_lucky13),        
                CatBoostRegressor(**mp.cat_r_lucky13),
                
        ]

        ens_341 = [
                CatBoostClassifier(),        
                CatBoostClassifier(**mp.cat_c_set),        
                CatBoostClassifier(**mp.cat_c_t),        
                CatBoostRegressor(),
        ]


        ens_290 = [
                CatBoostClassifier(**mp.cat_c_lucky13),        
                CatBoostClassifier(**mp.cat_c_set),        
                CatBoostClassifier(**mp.cat_c_t),        
        ]


     

        # ==========================================

        datafile = [ 
                'data/Lucky13_3070_oos.csv',   
                'data/Lucky13_3070.csv',  #1
                'data/ndata_diff_lucky13_3070_oos.csv', 
                'data/ndata_diff_lucky13_3070.csv', #3
                'data/ndata_lucky_13_lag_3070_oos.csv', 
                'data/ndata_lucky13_lag_3070.csv', #5
                'new_model_Z_lucky13_3070_oos.csv',
                'new_model_Z_lucky13_3070.csv' #7,
                'data/Lucky13_3070_oos_3.csv',   
                'data/Lucky13_3070_3.csv',  #8
                'data/Lucky13_3070_oos_5.csv',   
                'data/Lucky13_3070_5.csv',  #10
                'data/Lucky13_A_ALL_5M.csv',  
                'data/Lucky13_A_ALL_15.csv',  #12

        ]


        df = pd.read_csv(datafile[11])

        df = df[((df['RSI'] > 20) & (df['RSI'] < 30))|(df['RSI'] > 70) & (df['RSI'] < 80)] 
        

        f_87 =['RSI',
                'STOK1',
                'SDKC9',
                'SDLR310',
                'ATR2',
                'SDKC91',
                'SDBB91',
                'ATR5',
                'ATR21',]

        #feat_data = f_87
        #feat_data = lucky13
        feat_data = 'xxx'
        split_test_size_value = 0.7          
        save_mlflow = False
        
        p_df, experiment_id_parent = run_models(df, ens_342, split_test_size_value, save_mlflow, feat_data)
        
        print("")
        for run_uuid, input_features, e_perf, features_list, correctX, correctY, correctP, totalX, cxp, cyp, cpp, mse, rmse, r2, mae in p_df.values.tolist(): 
                print(f"{run_uuid}  {cxp}  {cyp}  {cpp}  {mse}  {rmse} {mae} {r2}  ")
        
        print("")


if __name__ == "__main__":
    run()

"""

baseline
LGBMRegressorV2  0.7603
CatBoostRegressorV2  0.7615
XGBRegressorV2  0.7539
XGBRFRegressorV2  0.7628
LGBMClassifierV2  0.7741
XGBClassifierV2  0.7636
CatBoostClassifierV2  0.7794
XGBRFClassifierV2  0.7834

19b97c  0.8187636590696222  0.8532625663440524  0.8353106462691227  0.0000  0.0001 0.0000 -6.436285709846608


"""