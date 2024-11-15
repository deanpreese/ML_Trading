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
                        new_exp_name = f"mixer_runs_{experiment_id}"
                        mlflow.MlflowClient().rename_experiment(experiment_id, new_exp_name)
                except Exception as e:
                        print(f"{e}")    
                        experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id        
                
        perf_data = []
        perf_data_t, output_text = model_processing.process_models(experiment_id, data, estimators, run_test_size, save_to_mlflow, feat_data, False, 0)
        
        for i in range(len(output_text)):
                print(output_text[i])
        
        
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

        ens_m = [
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
        

        ens_r = [
                LGBMRegressor(), 
                CatBoostRegressor(),
                #XGBRegressor(),   
                #LGBMRegressor(**mp.lgb_r_lucky13), 
                #CatBoostRegressor(**mp.cat_r_lucky13),
                XGBRegressor(**mp.xgb_r_lucky13),   
                #XGBRFRegressor(**mp.xgbrf_r_lucky13),                
                #LGBMRegressor(**mp.lgb_r_L13EX), 
                CatBoostRegressor(**mp.cat_r_L13EX),
                XGBRegressor(**mp.xgb_r_L13EX),   
                #XGBRFRegressor(**mp.xgbrf_r_L13EX),                
                LGBMRegressor(**mp.lgb_r_set), 
                CatBoostRegressor(**mp.cat_r_set),
                #XGBRegressor(**mp.xgb_r_set),   
                #XGBRFRegressor(**mp.xgbrf_r_set),                
                #LGBMRegressor(**mp.lgb_r_t), 
                CatBoostRegressor(**mp.cat_r_t),
                XGBRegressor(**mp.xgb_r_t),   
                #XGBRFRegressor(**mp.xgbrf_r_t),                
        ]    


        ens_c = [
                LGBMClassifier(),
                #XGBClassifier(),
                CatBoostClassifier(),
                #LGBMClassifier(**mp.lgb_c_lucky13),
                #XGBClassifier(**mp.xgb_c_lucky13),
                #CatBoostClassifier(**mp.cat_c_lucky13),
                #XGBRFClassifier(**mp.xgbrf_c_lucky13),
                LGBMClassifier(**mp.lgb_c_L13EX),
                #XGBClassifier(**mp.xgb_c_L13EX),
                #CatBoostClassifier(**mp.cat_c_L13EX),
                #XGBRFClassifier(**mp.xgbrf_c_L13EX),
                LGBMClassifier(**mp.lgb_c_set),
                XGBClassifier(**mp.xgb_c_set),
                CatBoostClassifier(**mp.cat_c_set),
                #LGBMClassifier(**mp.lgb_c_t),
                XGBClassifier(**mp.xgb_c_t),
                CatBoostClassifier(**mp.cat_c_t),
        ]       

        ens_cc =[
                LGBMClassifier(),
                #XGBClassifier(),
                CatBoostClassifier(),
                #LGBMClassifier(**mp.lgb_c_lucky13),
                #XGBClassifier(**mp.xgb_c_lucky13),
                #CatBoostClassifier(**mp.cat_c_lucky13),
                #XGBRFClassifier(**mp.xgbrf_c_lucky13),
                LGBMClassifier(**mp.lgb_c_L13EX),
                #XGBClassifier(**mp.xgb_c_L13EX),
                #CatBoostClassifier(**mp.cat_c_L13EX),
                #XGBRFClassifier(**mp.xgbrf_c_L13EX),
                LGBMClassifier(**mp.lgb_c_set),
                XGBClassifier(**mp.xgb_c_set),
                CatBoostClassifier(**mp.cat_c_set),
                #LGBMClassifier(**mp.lgb_c_t),
                XGBClassifier(**mp.xgb_c_t),
                CatBoostClassifier(**mp.cat_c_t),
                LGBMRegressor(), 
                CatBoostRegressor(),
                #XGBRegressor(),   
                #LGBMRegressor(**mp.lgb_r_lucky13), 
                #CatBoostRegressor(**mp.cat_r_lucky13),
                XGBRegressor(**mp.xgb_r_lucky13),   
                #XGBRFRegressor(**mp.xgbrf_r_lucky13),                
                #LGBMRegressor(**mp.lgb_r_L13EX), 
                CatBoostRegressor(**mp.cat_r_L13EX),
                XGBRegressor(**mp.xgb_r_L13EX),   
                #XGBRFRegressor(**mp.xgbrf_r_L13EX),                
                LGBMRegressor(**mp.lgb_r_set), 
                CatBoostRegressor(**mp.cat_r_set),
                #XGBRegressor(**mp.xgb_r_set),   
                #XGBRFRegressor(**mp.xgbrf_r_set),                
                #LGBMRegressor(**mp.lgb_r_t), 
                CatBoostRegressor(**mp.cat_r_t),
                XGBRegressor(**mp.xgb_r_t),   
                #XGBRFRegressor(**mp.xgbrf_r_t),                                  
        ]

        ens_newmodel = [ 
                LGBMRegressor(**mp.lgb_c_New3070), 
                CatBoostRegressor(**mp.cat_r_New3070),
                XGBRegressor(**mp.xgb_r_New3070), 
                #XGBClassifier(**mp.xgb_c_New3070),
                CatBoostClassifier(),
                LGBMRegressor(**mp.lgb_c_New3070), 
                ]


        # ==========================================
        datafile = [ 
                'data/NewModel_3070_oos.csv',   
                'data/NewModel_3070.csv',  #1
                'data/NewModel_ALL_oos.csv',   
                'data/NewModel_ALL.csv',  #3
                'data/NewModel_span3_3070_oos.csv',   
                'data/NewModel_span3_3070.csv',  #5
                
        ]   


        df = pd.read_csv(datafile[1])
        #df = df[((df['RSI'] > 20) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 80)]  
        #df = df[((df['RSI'] > 25) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 75)] 
        #df = df[(df['RSI'] > 60)]  
        #df = df[(df['RSI'] < 40)]  

        feat_data = 'xxx'
        
        lucky_13 = ['SDLR310','SDBB91','SDKC91','SDKC9','ROC','ATR54','ATR53','ATR52','ATR51','ATR5','ATR21','ATR2','RSI','STOK1']
        feat_data = lucky_13
        f_13x =['RSI','STOK1','SDLR310', 'ATR2', 'SDBB91','ATR5', 'ATR21']       
        #feat_data = f_13x
        
        comp_new = ['L1', 'RSI', 'O5', 'ROC', 'ROC14', 'ATR5', 'ATR21', 'TV41', 'RSI14X', 'TV31', 'ATR2', 
                'H2', 'ATR14Y1', 'L5', 'L2', 'STOK1', 'ROC1', 'SDBB91', 'ATR14Y', 'ROC141', 'SDKC9', 'SDLR310']
        #feat_data = comp
                
        comp_new_min = ['ATR2', 'RSI14X', 'SDLR310', 'ROC1', 'ATR5', 'SDKC9', 'TV31', 'RSI', 'O5', 'TV41', 'ATR21', 'STOK1']
        #feat_data = comp_new_min
                
        span3 = ['RSI', 'ATR5', 'STOK1', 'H1', 'SDBB9', 'ATR2', 'SDBB91', 'TV11', 'ROC141', 
             'ATR21', 'TV31', 'RSI14X', 'SDKC9', 'ROC1', 'SDLR310', 'ROC14', 'HourOfDay', 'TV41', 'ROC', 'SDKC91']
        #feat_data = span3                
                
        span3_min = ['ROC', 'TV31', 'ATR2', 'TV11', 'ROC141', 'RSI14X', 'TV41', 'RSI', 'STOK1', 'ATR5']                
        feat_data = span3_min                
                
        split_test_size_value = 0.7          
        save_mlflow = False
        
        p_df, experiment_id_parent = run_models(df, ens_newmodel, split_test_size_value, save_mlflow, feat_data)
        
        print("")
        for run_uuid, input_features, e_perf, features_list, correctX, correctY, correctP, totalX, cxp, cyp, cpp, mse, rmse, r2, mae in p_df.values.tolist(): 
                print(f"{run_uuid}  {cxp}  {cyp}  {cpp}  {mse}  {rmse} {mae} {r2}  ")
        
        print("")


if __name__ == "__main__":
    run()

