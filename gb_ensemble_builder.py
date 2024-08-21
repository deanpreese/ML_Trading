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

from ml_model.model_tracking import save_reg_ens_data
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
                
                

def run_stack_vote(est_list):
                
        stacking_regressor = StackingRegressor(estimators=est_list, cv=4
        , verbose=True, final_estimator=CatBoostRegressor()
        )

        regressor = stacking_regressor
        #regressor = voting_regressor


        # Train the Voting Regressor on the training data
        regressor.fit(X_train, y_train)

        print(" ")
        #print("Saving and Reloading Model ")
        #pickle.dump(voting_regressor, open(model_filename, "wb"))
        #loaded_model = pickle.load(open(model_filename, "rb"))
        #predictions = loaded_model.predict(X_test)

        y_pred = regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred, squared=True)
        rmse =mean_squared_error(y_test, y_pred, squared=False)
        r2 =r2_score(y_test, y_pred)
        score = regressor.score(X_test, y_test)
        mae = float(mean_absolute_error(y_test,y_pred))                
        perf, tot = gen_reg_stats(y_test, y_pred)        

        print("Results ---")
        print(f"MSE  {mse}   RMSE {rmse}  R2 {r2}  Score {score}  MAE {mae}  Perf  {perf}  Total {tot}" )

        for m in regressor.named_estimators_:
                r_pred = regressor.named_estimators_[m].predict(X_test)

                mse = mean_squared_error(y_test, r_pred, squared=True)
                rmse =mean_squared_error(y_test, r_pred, squared=False)
                r2 =r2_score(y_test, r_pred)
                score = regressor.score(X_test, r_pred)
                mae = float(mean_absolute_error(y_test,r_pred))                
                perf, total, mse, rmse = gen_reg_stats(y_test, r_pred)        

                print(f"{m}  MSE  {mse}   RMSE {rmse}  R2 {r2}  Score {score}  MAE {mae}  Perf  {perf}  Total {tot}" )
                        
                
                
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

        baseline_r = [
                #XGBRegressor(), 
                XGBRegressor(**mp.xgbr_set ), 
                #XGBRFRegressor(), 
                XGBRFRegressor(**mp.xgbrf_set ),
                #CatBoostRegressor(),  
                CatBoostRegressor(**mp.cbr_set),
                #LGBMRegressor(), 
                LGBMRegressor(**mp.lbr_set),               
        ]

        est_list_1 = [ 
                XGBRegressor(), 
                #XGBRegressor(**mp.xgbr_set2 ), 
                #XGBRFRegressor(), 
                #XGBRFRegressor(**mp.xgbrf_set ),
                #CatBoostRegressor(),  
                #CatBoostRegressor(**mp.cbr_set),
                #LGBMRegressor(), 
                #LGBMRegressor(**mp.lbr_set), 
                #LGBMClassifier(),
                #LGBMClassifier(**mp.lbc_set),
                XGBClassifier(),
                XGBClassifier(**mp.lbc_set),
                CatBoostClassifier(),
                CatBoostClassifier(**mp.cbc_set),
                #XGBRFClassifier(),
        ]


        est_list_2 = [
                #XGBRegressor(), 
                XGBRegressor(**mp.xgbr_t),   
                XGBClassifier(),
                XGBRFClassifier(**mp.xgc_set),
                XGBClassifier(**mp.xgbc_t),
                
                #LGBMRegressor(**mp.lgbr_t), 
                LGBMClassifier(**mp.lbc_set),
                LGBMClassifier(**mp.lgbc_t),
                
                #CatBoostRegressor(**mp.catr_t),
                #CatBoostClassifier(),
                CatBoostClassifier(**mp.catc_t),
                CatBoostClassifier(**mp.cbc_set),
                
                #XGBRFRegressor(),                
                XGBRFClassifier(),
                #XGBRFRegressor(**mp.xgbrf_set ),
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

        ]



        dtx = pd.read_csv(datafile[1])


        f_87 =['RSI',
                'STOK1',
                'SDKC9',
                'SDLR310',
                'ATR2',
                'SDKC91',
                'SDBB91',
                'ATR5',
                'ATR21',]

        lucky13 = [
                #'SDLR310',
                #'SDBB91',
                #'SDKC91',
                'SDKC9',
                'ROC',
                #'ATR54',
                #'ATR52',
                #'ATR51',
                'ATR5',
                #'ATR21',
                'ATR2',
                'RSI',
                'STOK1'
                ]

 

        #feat_data = f_87
        #feat_data = lucky13
        feat_data = 'xxx'
        split_test_size_value = 0.7          
        save_mlflow = False
                
        p_df, experiment_id_parent = run_models(dtx, baseline_r, split_test_size_value, save_mlflow, feat_data)

        print("")
        for run_uuid, input_features, e_perf, features_list, correctX, correctY, correctP, totalX, cxp, cyp, cpp, mse, rmse, r2, mae in p_df.values.tolist(): 
                print(f"{run_uuid}  {cxp}  {cyp}  {cpp}  {mse}  {rmse} {mae} {r2}  ")
        
        print("")


if __name__ == "__main__":
    run()

