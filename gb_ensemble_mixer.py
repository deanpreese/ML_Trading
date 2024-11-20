
import datetime as dte_time
import random as rand
import uuid
import mlflow
import pandas as pd
import logging
import warnings

from ml_model.model_process import save_reg_ens_data
from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor
from lightgbm  import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)
logging.getLogger('lightgbm').setLevel(logging.ERROR)
logging.getLogger('[LightGBM]').setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
                        perf_data_t, output_text = model_processing.process_models(experiment_id, data, estimators, run_test_size, True, 'xxx', True, f)
                        perf_data.append(perf_data_t)
                
                p_df = pd.DataFrame(perf_data)  
        
                p_df.columns = ["rid", "input_features", "e_perf", "features_list",  "ens_accuracy", "ens_precision", "ens_recall", "win_p", "loss_p", "tn_p", "tp_p", "fn_p", "fp_p" ]
                p_df.sort_values(by=['win_p'], ascending=False, inplace=True)                
                  
                #p_df.columns = ["rid", "input_features", "e_perf", "features_list", "correctX", "correctY", 
                #                "correctP", "totalX", "cxp", "cyp", "cpp", "mse", "rmse", "r2", "mae"]
                #p_df.sort_values(by=['cpp'], ascending=False, inplace=True)


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

       



        # ==================
        datafile = [ 
                'data/NewModel_3070_oos.csv',   
                'data/NewModel_3070.csv',  #3
                'data/NewModel_ALL_oos.csv',   
                'data/NewModel_ALL.csv',  #3
        ]   

        for i in range(3):

                df = pd.read_csv(datafile[1])
                df = df.drop(columns=['TimeTicks','SeqClose'])

                #df = df[(df['RSI'] > 60)]  
                #df = df[(df['RSI'] < 40)]  
                #df = df[((df['RSI'] > 20) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 80)] 
                #df = df[((df['RSI'] > 25) & (df['RSI'] < 40))] 
                #df = df[( df['RSI'] > 60) & (df['RSI'] < 75 )] 
                
                #22 columns
                f_plus = ['RSI', 'RSI14', 'ATR2', 'SDBB91', 'STOK1', 'COMP2', 'RSI3', 'SDLR310', 'TV23', 'TV11', 'TV41', 'RSI7', 'STOK7143', 'TV21', 'TV13', 'TV22', 'TV43', 'TV42', 'ATR21', 'ROC', 'output', 'outputC']
                #df = df[f_plus]

                span3 = ['RSI', 'ATR5', 'STOK1', 'H1', 'SDBB9', 'ATR2', 'SDBB91', 'TV11', 'ROC141', 
                     'ATR21', 'TV31', 'RSI14X', 'SDKC9', 'ROC1', 'SDLR310', 'ROC14', 'HourOfDay', 'TV41', 'ROC', 'SDKC91', 'output', 'outputC']
                df = df[span3]

                
                max_avail = df.shape[1] - 3
        
                split_test_size_value = 0.7          
                min_features_used = 3
                max_features_used = 9
                step_features_used = 1
                total_cycles_used = 20

                p_df, experiment_id_parent = run_models(df, ens_389, 
                                                        split_test_size_value, min_features_used, max_features_used, 
                                                        step_features_used, total_cycles_used  )

                print(" ")
                print(p_df)                
                print(" ")




if __name__ == "__main__":
    run()

