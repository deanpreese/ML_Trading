
import datetime as dte_time
import random as rand
import uuid
#import warnings
import mlflow

import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from common.common_func import calc_MSE, calc_reg_results, calc_reg_streaks, show_stats, simple_split_and_scale

from models.wrapped_models import TunableCatBoostRegressor, TunableLGBMRegressor, TunableXGBRegressor

import logging
logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)


def process_model(exp_name, data, models, run_test_size, feature_list_size):
        
        run_uuid = str(uuid.uuid1())[:6]
        
        features_list = []
        all_predict_data = pd.DataFrame()
        estimator_perf = []
        estimator_run_ids = []     
                
        for f, e in enumerate(models):
                
                fl_out = []
                
                model_run_uuid = run_uuid + "-"+ str(uuid.uuid1())[:6]
                modelname = e.__class__.__name__
                
                num_columns = len(data.columns)
                input_features = num_columns - 2
                X = data.iloc[:, 0:input_features]
                y = data['output'].values
                
                idxx = rand.sample(range(1, len(data.axes[1]) -2 ), feature_list_size)
                features_list.append(idxx)    
                fl= features_list[f]
                X = data.iloc[:, fl]  
                input_features = len(X.axes[1]) 
                fl_out = fl
              
                X_train, X_test, y_train, y_test = simple_split_and_scale(X, y, run_test_size, 0)
                
                #e.features_used = fl_out
                e.features_used = X_train.columns
                
                perf, tot, mse, rmse, r2, score, mae, predictions = e.track_model(exp_name, True, e, X_train, 
                                                                                  y_train, X_test, y_test)  
                all_predict_data[model_run_uuid] = predictions
                perf, tot = show_stats(False, y_test, predictions)
                mse, rmse  = calc_MSE(y_test, predictions, False)
                score = e.score(X_test, y_test)
                
                combined_prod_perf = predictions * perf
                nm = f"{model_run_uuid}_p"
                all_predict_data[nm] = combined_prod_perf
                
                estimator_run_ids.append(model_run_uuid)
                
                outputs = [ modelname, perf, tot, mse, rmse, score, fl_out, model_run_uuid, e.run_id ]        
                estimator_perf.append(outputs)  

        all_predict_data["target"] = y_test
        e_perf = pd.DataFrame(estimator_perf)        
        e_perf.columns = ["Estimator", "Perf", "Total", "MSE", "RMSE", "Score", "Features", "UUID", "RUN_ID" ]
        
        correctX, correctY, correctP, totalX, cxp, cyp, cpp, r_predictions, r_y_target = calc_reg_results(all_predict_data, estimator_run_ids)
        current_streak, longest_win_streak, longest_loss_streak, AveWinSt, AveLossSt, AveWinMiss, AveLossMiss, AveUpMiss, AveDownMiss = calc_reg_streaks(r_predictions, r_y_target, False, False, False)                

        mse = mean_squared_error(r_y_target, r_predictions, squared=True)
        rmse =mean_squared_error(r_y_target, r_predictions, squared=False)
        r2 =r2_score(r_y_target, r_predictions)
        score = r2
        mae = float(mean_absolute_error(r_y_target,r_predictions))                

        perf_data_t = [run_uuid, feature_list_size, e_perf.values.tolist(), features_list, 
                       correctX, correctY, correctP, totalX, cxp, cyp, cpp, 
                       current_streak, longest_win_streak, longest_loss_streak, AveWinSt, AveLossSt, 
                       AveWinMiss, AveLossMiss, AveUpMiss, AveDownMiss, mse, rmse, r2, mae]
        
        return perf_data_t    



def run_models(data, estimators, run_test_size, 
               min_features, max_features, step_features, total_cycles ):
        
        feature_list_size = min_features
        p_df = pd.DataFrame()
        
        if len(data.columns) < max_features:
                max_features = len(data.columns) - 3
        
        time_stamp = dte_time.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        exp_name = f"mixer_runs_{time_stamp}"
        
        try:
            experiment_id = mlflow.create_experiment(exp_name)
        except Exception as e:
            print(f"{e}")    
            experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id        
       
                
        perf_data = []
        
        for q in range(total_cycles):
                
                for f in range(min_features, max_features, step_features):
                
                        feature_list_size = f
                        
                        perf_data_t = process_model(experiment_id, data, 
                                                        estimators, run_test_size, feature_list_size)
                        perf_data.append(perf_data_t)
                
                p_df = pd.DataFrame(perf_data)    
                p_df.columns = ["rid", "input_features", "e_perf", "features_list", "correctX", "correctY", 
                                "correctP", "totalX", "cxp", "cyp", "cpp", "current_streak", "longest_win_streak", "longest_loss_streak"
                                , "AveWinSt", "AveLossSt", "AveWinMiss", "AveLossMiss", "AveUpMiss", "AveDownMiss", "mse", "rmse", "r2", "mae"]
                
                p_df.sort_values(by=['cpp'], ascending=False, inplace=True)
                                        
                        

        step = 0
        time_stamp = dte_time.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        exp_name = f"mixer_output_{time_stamp}"
        
        try:
           experiment_id = mlflow.create_experiment(exp_name)
        except Exception as e:
            print(f"{e}")    
            experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id        
       
        for run_uuid, input_features, e_perf, features_list, correctX, correctY, correctP, totalX, cxp, cyp, cpp, current_streak, longest_win_streak, longest_loss_streak, AveWinSt, AveLossSt, AveWinMiss, AveLossMiss, AveUpMiss, AveDownMiss, mse, rmse, r2, mae in p_df.values.tolist() :
        
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

                        mlflow.log_metric('Longest Win Streak', longest_win_streak, step)
                        mlflow.log_metric('Longest Loss Streak', longest_loss_streak, step)
                        mlflow.log_metric('Ave Win Streak', AveWinSt, step)
                        mlflow.log_metric("Ave Loss Streak", AveLossSt, step)
                        mlflow.log_metric('Ave Win Miss', AveWinMiss, step)
                        mlflow.log_metric("Ave Loss Miss", AveLossMiss, step)
                        mlflow.log_metric("Ave Up Miss", AveUpMiss, step)
                        mlflow.log_metric("Ave Dwn Miss", AveDownMiss, step)        
                        
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

datafile = [ 
        'data/buildSeqInd_13X_5M_ALL.csv',   #0
        'data/buildSeqIndX_5M_ALL.csv',   #1
        'data/buildSeqInd_Lucky13_5M_3070.csv',   #2
        'data/markov.csv',  #3
        'data/buildSeqInd_Lucky13_5M_ALL.csv',  #4
        'data/buildSeqInd_13X_5M_3070.csv',  #5
        'data/buildSeqIndX_5M_3070.csv',   #6
        'data/Fractal_ALL_5M.csv', #7
        'data/buildSeqInd_Lucky11_5M_3070.csv',   #8
        'data/buildSeqInd_Lucky13_F.csv',  #9
        'data/buildSeqInd_Lucky13_D.csv',  #10
    ]


dtx = pd.read_csv(datafile[2])

"""
est_list = [ TunableXGBRegressor(),  TunableXGBRegressor(), TunableXGBRegressor(), 
             TunableXGBRegressor(),  TunableXGBRegressor(), TunableXGBRegressor(), 
            TunableXGBRegressor(),  TunableXGBRegressor(), TunableXGBRegressor(),
            TunableXGBRegressor(),  TunableXGBRegressor(), TunableXGBRegressor(),
            TunableXGBRegressor(),  TunableXGBRegressor(), TunableXGBRegressor() ,
            TunableXGBRegressor(),  TunableXGBRegressor(), TunableXGBRegressor()   
                ]
est_list = [ TunableXGBRegressor(),  TunableXGBRegressor(), TunableXGBRegressor(),  TunableXGBRegressor(),
             TunableXGBRegressor(),  TunableXGBRegressor(), TunableXGBRegressor(),  TunableXGBRegressor(), 
             TunableXGBRegressor()]


est_list = [ TunableCatBoostRegressor(), TunableXGBRegressor(),  TunableLGBMRegressor(),
             TunableCatBoostRegressor(), TunableXGBRegressor(),  TunableLGBMRegressor(),  
             TunableCatBoostRegressor(), TunableXGBRegressor(),  TunableLGBMRegressor() ]

"""
est_list = [  TunableXGBRegressor(),  TunableXGBRegressor(),  TunableCatBoostRegressor() ,
              TunableXGBRegressor(),  TunableXGBRegressor(), TunableLGBMRegressor()  ]


split_test_size_value = 0.7          

min_features_used = 1
max_features_used = 4
step_features_used = 1
total_cycles_used = 100


p_df, experiment_id_parent = run_models(dtx, est_list, 
                                        split_test_size_value, min_features_used, max_features_used, 
                                        step_features_used, total_cycles_used  )

print(" ")
print(p_df)                
print(" ")
