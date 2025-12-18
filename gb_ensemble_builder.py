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
import ml_model.feature_filter as feature_filter


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
                CatBoostClassifier(),
                CatBoostClassifier(**mp.cat_c_L13EX),
                CatBoostClassifier(**mp.cat_c_New3070),
                CatBoostClassifier(**mp.cat_c_lucky13),
                CatBoostClassifier(**mp.cat_c_set),
                CatBoostClassifier(**mp.cat_c_t),
                CatBoostRegressor(),
                CatBoostRegressor(**mp.cat_r_L13EX),
                CatBoostRegressor(**mp.cat_r_New3070),
                CatBoostRegressor(**mp.cat_r_lucky13),
                CatBoostRegressor(**mp.cat_r_set),
                CatBoostRegressor(**mp.cat_r_t),
                LGBMClassifier(),
                LGBMClassifier(**mp.lgb_c_L13EX),
                LGBMClassifier(**mp.lgb_c_New3070),
                LGBMClassifier(**mp.lgb_c_lucky13),
                LGBMClassifier(**mp.lgb_c_set),
                LGBMClassifier(**mp.lgb_c_t),
                LGBMRegressor(),
                LGBMRegressor(**mp.lgb_r_L13EX),
                LGBMRegressor(**mp.lgb_r_New3070),
                LGBMRegressor(**mp.lgb_r_lucky13),
                LGBMRegressor(**mp.lgb_r_set),
                LGBMRegressor(**mp.lgb_r_t),
                XGBClassifier(),
                XGBClassifier(**mp.xgb_c_L13EX),
                XGBClassifier(**mp.xgb_c_New3070),
                XGBClassifier(**mp.xgb_c_lucky13),
                XGBClassifier(**mp.xgb_c_set),
                XGBClassifier(**mp.xgb_c_t),
                XGBRegressor(),
                XGBRegressor(**mp.xgb_r_L13EX),
                XGBRegressor(**mp.xgb_r_New3070),
                XGBRegressor(**mp.xgb_r_lucky13),
                XGBRegressor(**mp.xgb_r_set),
                XGBRegressor(**mp.xgb_r_t)
                ]


        ens_r = [
                LGBMRegressor(), 
                CatBoostRegressor(),
                XGBRegressor(),   
                LGBMRegressor(**mp.lgb_r_lucky13), 
                CatBoostRegressor(**mp.cat_r_lucky13),
                XGBRegressor(**mp.xgb_r_lucky13),   
                LGBMRegressor(**mp.lgb_r_L13EX), 
                CatBoostRegressor(**mp.cat_r_L13EX),
                XGBRegressor(**mp.xgb_r_L13EX),   
                LGBMRegressor(**mp.lgb_r_set), 
                CatBoostRegressor(**mp.cat_r_set),
                XGBRegressor(**mp.xgb_r_set),   
                LGBMRegressor(**mp.lgb_r_t), 
                CatBoostRegressor(**mp.cat_r_t),
                XGBRegressor(**mp.xgb_r_t), 
        ]  
                


        ens_c = [
                LGBMClassifier(),
                XGBClassifier(),
                CatBoostClassifier(),
                LGBMClassifier(**mp.lgb_c_lucky13),
                XGBClassifier(**mp.xgb_c_lucky13),
                CatBoostClassifier(**mp.cat_c_lucky13),
                LGBMClassifier(**mp.lgb_c_L13EX),
                XGBClassifier(**mp.xgb_c_L13EX),
                CatBoostClassifier(**mp.cat_c_L13EX),
                LGBMClassifier(**mp.lgb_c_set),
                XGBClassifier(**mp.xgb_c_set),
                CatBoostClassifier(**mp.cat_c_set),
                LGBMClassifier(**mp.lgb_c_t),
                XGBClassifier(**mp.xgb_c_t),
                CatBoostClassifier(**mp.cat_c_t),
        ]       



        ens_13 = [
                CatBoostClassifier(),
                CatBoostClassifier(**mp.cat_c_L13EX),
                #CatBoostClassifier(**mp.cat_c_New3070),
                CatBoostClassifier(**mp.cat_c_set),
                LGBMClassifier(**mp.lgb_c_t),
                XGBClassifier(**mp.xgb_c_t),
                ]


        ens_13_s = [
                CatBoostClassifier(),
                CatBoostClassifier(**mp.cat_c_New3070),
                CatBoostClassifier(**mp.cat_c_set),
                LGBMClassifier(**mp.lgb_c_L13EX),
                LGBMClassifier(**mp.lgb_c_lucky13),
                LGBMClassifier(**mp.lgb_c_set),
                XGBClassifier(**mp.xgb_c_set),
                XGBRegressor(**mp.xgb_r_t)
                ]

        ens_x_1 = [
                CatBoostClassifier(),
                CatBoostClassifier(**mp.cat_c_New3070),
                CatBoostClassifier(**mp.cat_c_set),
                LGBMClassifier(),
                LGBMClassifier(**mp.lgb_c_L13EX),
                LGBMClassifier(**mp.lgb_c_set),
                XGBClassifier(**mp.xgb_c_set),
                XGBClassifier(**mp.xgb_c_t),
                #XGBRegressor(**mp.xgb_r_t)
                ]


        # ==========================================
        datafile = [ 
                'data/Lucky13_3070_oos.csv',   
                'data/Lucky13_3070.csv',  #1
                'data/Model_M_1_3070_oos.csv',  
                'data/Model_M_1_3070.csv',  #3
        ]   

        df = pd.read_csv(datafile[3])
        #df = df[((df['RSI'] > 0) & (df['RSI'] < 30))|(df['RSI'] > 70) & (df['RSI'] < 100)]  
        #df = df[((df['RSI'] > 25) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 75)] 
        
        #df = df[((df['output'] > -9.0) & (df['output'] < 9.0))]
        #df = df[((df['ATR2'] > 40) & (df['ATR2'] < 70))]
        #df = df[((df['ATR5'] > 36) & (df['ATR5'] < 80))]
        
        #col_filter = feature_filter.lucky13_all         
        #col_filter = feature_filter.lucky13_3070_comp
        #col_filter = feature_filter.model_m_1_all
        #col_filter = feature_filter.model_m_1_alt
        #col_filter = feature_filter.model_m_1_slim_x
        col_filter = feature_filter.model_m_1_r2
                
        split_test_size_value = 0.7          
        save_mlflow = False
        
        p_df, experiment_id_parent = run_models(df, ens_x_1, split_test_size_value, save_mlflow, col_filter)
        
        print("")
        print(df.shape)
        
        print("")
        for run_uuid, input_features, e_perf, features_list, ens_accuracy, ens_precision, ens_recall, win_p, loss_p, tn_p, tp_p, fn_p, fp_p  in p_df.values.tolist(): 
                print(f"{run_uuid} Acc: {ens_accuracy} Prec: {ens_precision} Recall: {ens_recall} --- Win%: {win_p}  Loss%: {loss_p}  TN: {tn_p}  TP: {tp_p}  FN: {fn_p} FP: {fp_p} ")
        
        print("")




if __name__ == "__main__":
    run()
