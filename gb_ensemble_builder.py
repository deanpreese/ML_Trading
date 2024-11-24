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
        
        
        #  perf_data_t = [run_uuid, 0, e_perf.values.tolist(), features_list, win_p, loss_p, tn_p, tp_p, fn_p, fp_p ]
        
        perf_data.append(perf_data_t)
        p_df = pd.DataFrame(perf_data)    
        p_df.columns = ["rid", "input_features", "e_perf", "features_list",  "ens_accuracy", "ens_precision", "ens_recall", "win_p", "loss_p", "tn_p", "tp_p", "fn_p", "fp_p" ]
        p_df.sort_values(by=['win_p'], ascending=False, inplace=True)

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
                LGBMRegressor(), 
                CatBoostRegressor(),
                XGBRegressor(),   
                LGBMClassifier(),
                XGBClassifier(),
                CatBoostClassifier(),

                LGBMRegressor(**mp.lgb_r_lucky13), 
                CatBoostRegressor(**mp.cat_r_lucky13),
                XGBRegressor(**mp.xgb_r_lucky13),   
                LGBMClassifier(**mp.lgb_c_lucky13),
                XGBClassifier(**mp.xgb_c_lucky13),
                CatBoostClassifier(**mp.cat_c_lucky13),
                
                LGBMRegressor(**mp.lgb_r_L13EX), 
                CatBoostRegressor(**mp.cat_r_L13EX),
                XGBRegressor(**mp.xgb_r_L13EX),   
                LGBMClassifier(**mp.lgb_c_L13EX),
                XGBClassifier(**mp.xgb_c_L13EX),
                CatBoostClassifier(**mp.cat_c_L13EX),
                
                LGBMRegressor(**mp.lgb_r_set), 
                CatBoostRegressor(**mp.cat_r_set),
                XGBRegressor(**mp.xgb_r_set),   
                LGBMClassifier(**mp.lgb_c_set),
                XGBClassifier(**mp.xgb_c_set),
                CatBoostClassifier(**mp.cat_c_set),

                LGBMRegressor(**mp.lgb_r_t), 
                CatBoostRegressor(**mp.cat_r_t),
                XGBRegressor(**mp.xgb_r_t),   
                LGBMClassifier(**mp.lgb_c_t),
                XGBClassifier(**mp.xgb_c_t),
                CatBoostClassifier(**mp.cat_c_t),
                
                LGBMRegressor(**mp.lgb_r_New3070), 
                CatBoostRegressor(**mp.cat_r_New3070),
                XGBRegressor(**mp.xgb_r_New3070), 
                LGBMClassifier(**mp.lgb_c_New3070),
                XGBClassifier(**mp.xgb_c_New3070),
                CatBoostClassifier(**mp.cat_c_New3070)
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
                #LGBMRegressor(), 
                #CatBoostRegressor(),
                #XGBRegressor(),   
                #LGBMClassifier(),
                #XGBClassifier(),
                CatBoostClassifier(),

                #LGBMRegressor(**mp.lgb_r_lucky13), 
                #CatBoostRegressor(**mp.cat_r_lucky13),
                #XGBRegressor(**mp.xgb_r_lucky13),   
                #LGBMClassifier(**mp.lgb_c_lucky13),
                #XGBClassifier(**mp.xgb_c_lucky13),
                CatBoostClassifier(**mp.cat_c_lucky13),
                
                #LGBMRegressor(**mp.lgb_r_L13EX), 
                #CatBoostRegressor(**mp.cat_r_L13EX),
                #XGBRegressor(**mp.xgb_r_L13EX),   
                #LGBMClassifier(**mp.lgb_c_L13EX),
                #XGBClassifier(**mp.xgb_c_L13EX),
                #CatBoostClassifier(**mp.cat_c_L13EX),
                
                #LGBMRegressor(**mp.lgb_r_set), 
                #CatBoostRegressor(**mp.cat_r_set),
                #XGBRegressor(**mp.xgb_r_set),   
                #LGBMClassifier(**mp.lgb_c_set),
                XGBClassifier(**mp.xgb_c_set),
                CatBoostClassifier(**mp.cat_c_set),

                #LGBMRegressor(**mp.lgb_r_t), 
                #CatBoostRegressor(**mp.cat_r_t),
                #XGBRegressor(**mp.xgb_r_t),   
                #LGBMClassifier(**mp.lgb_c_t),
                XGBClassifier(**mp.xgb_c_t),
                #CatBoostClassifier(**mp.cat_c_t),
                
                #LGBMRegressor(**mp.lgb_r_New3070), 
                #CatBoostRegressor(**mp.cat_r_New3070),
                #XGBRegressor(**mp.xgb_r_New3070), 
                #LGBMClassifier(**mp.lgb_c_New3070),
                #XGBClassifier(**mp.xgb_c_New3070),
                CatBoostClassifier(**mp.cat_c_New3070)
        ]


        # ==========================================
        datafile = [ 
                'data/NewModel_3070_oos.csv',   
                'data/NewModel_3070.csv',  #1
                'data/NewModel_ALL_oos.csv',   
                'data/NewModel_ALL.csv',  #3
                
        ]   


        df = pd.read_csv(datafile[1])
        #df = df[((df['RSI'] > 20) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 80)]  
        #df = df[((df['RSI'] > 25) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 75)] 
        #df = df[(df['RSI'] > 60)]  
        #df = df[(df['RSI'] < 40)]  
        

        feat_data = 'xxx'
                
        
        f_list_f = ['SDBB91', 'COMP2', 'COMP3', 'ATR5', 'TV3', 'HourOfDay', 'TV1', 'ZH79X', 'SDKC29C', 
                'ZL57X', 'COMP0', 'ATR2', 'TV6', 'RSI14', 'RSI9', 'ATR51' ]   
        f_list_r = ['RSI9', 'ATR2', 'ATR5', 'ATR51', 'RSI14', 'TV3', 'TV6', 'COMP2', 'SDKC29C', 'COMP3']
        f_list_c = ['RSI9', 'ATR2', 'RSI14', 'TV6', 'TV1', 'ZL57X', 'COMP0', 'HourOfDay', 'SDBB91', 'ZH79X']
                        
        f_list_uni = [
                'RSI9', 'RSI14', 'COMP2', 'COMP1', 'COMP0', 'TV5', 'TV6', 
                'RSI91', 'ROC9', 'COMP3', 'STOK5133', 'ROC7', 'STOK714Y', 
                'ROC14', 'RSI141', 'TV2', 'TV3', 'ROC141', 'ROC91', 'TV1']   
                        
                        
        #feat_data = f_list_f               
                
        split_test_size_value = 0.7          
        save_mlflow = False
        
        p_df, experiment_id_parent = run_models(df, ens_m, split_test_size_value, save_mlflow, feat_data)
        
        print("")
        for run_uuid, input_features, e_perf, features_list, ens_accuracy, ens_precision, ens_recall, win_p, loss_p, tn_p, tp_p, fn_p, fp_p  in p_df.values.tolist(): 
                print(f"{run_uuid} Acc: {ens_accuracy} Prec: {ens_precision} Recall: {ens_recall} --- Win%: {win_p}  Loss%: {loss_p}  TN: {tn_p}  TP: {tp_p}  FN: {fn_p} FP: {fp_p} ")
        
        print("")


if __name__ == "__main__":
    run()

