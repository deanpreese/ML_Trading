
import datetime as dte_time
import random as rand
import uuid
#import warnings
import mlflow
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import logging

from ml_model.model_process import save_reg_ens_data

logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)

from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor
from lightgbm  import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

import ml_model.model_params as mp
import ml_model.model_process as model_processing



def run_models(data, estimators, run_test_size, min_features, max_features, step_features, total_cycles ):
        
        p_df = pd.DataFrame()
        #if len(data.columns) < max_features:
        #        max_features = len(data.columns) - 2
        
        time_stamp = dte_time.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        exp_name = f"mixer_runs_{time_stamp}"
        
        experiment_id = ""
        
        try:
            experiment_id = mlflow.create_experiment(exp_name)
            new_exp_name = f"mixer_runs_{experiment_id}"
            mlflow.MlflowClient().rename_experiment(experiment_id, new_exp_name)
            
        except Exception as e:
            print(f"{e}")    
            experiment_id = mlflow.get_experiment(experiment_id).experiment_id        
       
        perf_data = []
        for q in range(total_cycles):
                for f in range(min_features, max_features, step_features):
                        perf_data_t = model_processing.process_models(experiment_id, data, estimators, run_test_size, True, 'xxx', True, f)
                        perf_data.append(perf_data_t)
                
                p_df = pd.DataFrame(perf_data)    
                p_df.columns = ["rid", "input_features", "e_perf", "features_list", "correctX", "correctY", 
                                "correctP", "totalX", "cxp", "cyp", "cpp", "mse", "rmse", "r2", "mae"]
                p_df.sort_values(by=['cpp'], ascending=False, inplace=True)


        markdown_content = f"### {len(estimators)} Models  --  Min Feat {min_features}  Max Feat {max_features}  Cycles {total_cycles} \n\n"                                
        
        for item in estimators:
                markdown_content += f"\n\n"
                markdown_content += f"### {item.__class__.__name__} \n"
                markdown_content += f"{item.get_params()}\n\n"

        markdown_content += f"\n\n"                
        save_reg_ens_data(p_df, markdown_content)
                               
        return p_df, experiment_id        
                
# ---------------------------
#
# Run the models
#
# ---------------------------
def run():
        
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

       

        ens_385 = [
                XGBClassifier(**mp.xgb_c_lucky13),
                CatBoostClassifier(**mp.cat_c_lucky13),        
                CatBoostClassifier(**mp.cat_c_set),        
                CatBoostClassifier(**mp.cat_c_t),        
                CatBoostRegressor(**mp.cat_r_lucky13),
        ]


        ens_387 = [
                XGBClassifier(**mp.xgb_c_set),
                CatBoostClassifier(**mp.cat_c_lucky13),        
                CatBoostClassifier(**mp.cat_c_set),        
                CatBoostClassifier(**mp.cat_c_t),        
                CatBoostRegressor(**mp.cat_r_set),
        ]

        ens_389 = [
                XGBRFClassifier(**mp.xgbrf_c_L13EX),
                #XGBRFRegressor(**mp.xgbrf_set ),
                #XGBClassifier(**mp.xgb_c_set),
                CatBoostClassifier(**mp.cat_c_lucky13),        
                CatBoostClassifier(**mp.cat_c_set),        
                CatBoostClassifier(**mp.cat_c_t),        
                CatBoostRegressor(**mp.cat_r_set),
        ]


        ens_393 = [
                XGBRFClassifier(**mp.xgbrf_c_L13EX),
                #XGBRFRegressor(**mp.xgbrf_set ),
                #XGBClassifier(**mp.xgb_c_set),
                CatBoostClassifier(**mp.cat_c_L13EX),        
                CatBoostClassifier(**mp.cat_c_set),        
                CatBoostClassifier(**mp.cat_c_t),        
                CatBoostRegressor(**mp.cat_r_L13EX),
        ]


        ens_xxx = [
                XGBClassifier(**mp.xgb_c_lucky13),
                CatBoostClassifier(**mp.cat_c_lucky13),        
                CatBoostClassifier(**mp.cat_c_set),        
                CatBoostClassifier(**mp.cat_c_t),        
                #CatBoostRegressor(**mp.cat_r_lucky13),
                #XGBRegressor(**mp.xgb_r_set ), 
                #XGBRFRegressor(**mp.xgbrf_r_L13EX ),
                XGBRFClassifier(**mp.xgbrf_c_lucky13),
        ]



        # ==================
        datafile = [ 
                'data/Lucky13_3070_oos.csv',   
                'data/Lucky13_3070.csv',  #1
                'data/Lucky13_3070_AUG_oos.csv',   
                'data/Lucky13_3070_AUG.csv',  #3
        ]


        df = pd.read_csv(datafile[1])
        df = df.drop(columns=['ATR51', 'ATR52','SDKC91'])

        #df = df[(df['RSI'] > 60) & (df['RSI'] < 80)]  
        #df = df[(df['RSI'] > 20) & (df['RSI'] < 40)]  
        
        #df = df[((df['RSI'] > 20) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 80)] 
        
        #df = df[((df['RSI'] > 25) & (df['RSI'] < 40))] 
        #df = df[( df['RSI'] > 60) & (df['RSI'] < 75 )] 

        
        max_avail = df.shape[1] - 2
        
        split_test_size_value = 0.7          
        min_features_used = 6
        max_features_used = max_avail
        step_features_used = 1
        total_cycles_used = 15
        
        p_df, experiment_id_parent = run_models(df, ens_xxx, 
                                                split_test_size_value, min_features_used, max_features_used, 
                                                step_features_used, total_cycles_used  )

        print(" ")
        print(p_df)                
        print(" ")




if __name__ == "__main__":
    run()

