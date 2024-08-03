
import datetime as dte_time
import random as rand
import uuid
#import warnings
import mlflow
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import logging

from ml_model.model_tracking import train_regressor_model, save_reg_ens_data

logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)

from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor
from lightgbm  import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

import ml_model.model_params as mp
import ml_model.model_process as model_processing



def run_models(data, estimators, run_test_size, min_features, max_features, step_features, total_cycles ):
        
        p_df = pd.DataFrame()
        if len(data.columns) < max_features:
                max_features = len(data.columns) - 3
        
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

        est_list = [ 
                LGBMRegressor(**mp.lbr_set), 
                CatBoostRegressor(**mp.cat_3070), 
                CatBoostRegressor(**mp.cbr_set),
                XGBRegressor(**mp.xgbr_set),   
                XGBRegressor(**mp.xgb_p),   
                XGBRFRegressor(**mp.xgbrf_t),
                XGBRFRegressor(**mp.xgbrf_set),                       
        ]


        est_t = [ 
                XGBClassifier(),
                XGBRegressor(),   
        ]

        est_comb = [
                LGBMRegressor(**mp.lbr_set), 
                CatBoostRegressor(**mp.cbr_set),
                XGBRegressor(**mp.xgbr_set),   
                XGBRFRegressor(**mp.xgbrf_set),                
                LGBMClassifier(**mp.lbc_set),
                XGBClassifier(),
                CatBoostClassifier(**mp.cbc_set),
                XGBRFClassifier(),
        ]

        est_comb_2 = [
                CatBoostRegressor(**mp.cbr_set),
                XGBRFRegressor(**mp.xgbrf_set),                
                LGBMClassifier(**mp.lbc_set),
                XGBClassifier(),
                CatBoostClassifier(**mp.cbc_set),
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

        split_test_size_value = 0.7          
        min_features_used = 6
        max_features_used = 8
        step_features_used = 1
        total_cycles_used = 1

        p_df, experiment_id_parent = run_models(dtx, est_comb_2, 
                                                split_test_size_value, min_features_used, max_features_used, 
                                                step_features_used, total_cycles_used  )

        print(" ")
        print(p_df)                
        print(" ")




if __name__ == "__main__":
    run()

