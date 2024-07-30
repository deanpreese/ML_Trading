
import datetime as dte_time
import random as rand
import uuid
#import warnings
import mlflow
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import logging

import scipy.stats

from ml_model.model_tracking import track_regressor_model
from ml_model.model_stats import gen_reg_stats, calc_reg_ens_results
from ml_model.data_func import simple_split_and_scale
from ml_model.model_params import xgr_param_set, lbr_param_set, cbr_param_set, xgr_param_set2, xgb_rf_params


logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)

from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor
from lightgbm  import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor




def process_model(exp_name, data, models, run_test_size, save_to_mlflow, feat_data):
        
        run_uuid = str(uuid.uuid1())[:6]
        
        features_list = []
        all_predict_data = pd.DataFrame()
        estimator_perf = []
        estimator_run_ids = []     
                
        for f, e in enumerate(models):
                
                fl_out = []
                
                model_run_uuid = run_uuid + "-"+ str(uuid.uuid1())[:6]
                modelname = e.__class__.__name__
                
                X = data
                X = X.drop(columns=['output', 'outputC'])
                X= X[feat_data]
                                
                y = data['output'].values
                fl_out = list(X.columns)
              
                X_train, X_test, y_train, y_test = simple_split_and_scale(X, y, run_test_size, 42)
                
                run_id, perf, tot, mse, rmse, r2, score, mae, predictions = track_regressor_model(modelname, X_train.columns, exp_name, True, e, X_train, 
                                                                                  y_train, X_test, y_test, save_to_mlflow)  
                all_predict_data[model_run_uuid] = predictions
                perf, tot = gen_reg_stats(y_test, predictions)
                
                mse = mean_squared_error(y_test, predictions)
                rmse =  rmse = mse**.5
                
                score = e.score(X_test, y_test)
                
                combined_prod_perf = predictions * perf
                nm = f"{model_run_uuid}_p"
                all_predict_data[nm] = combined_prod_perf
                
                estimator_run_ids.append(model_run_uuid)
                
                outputs = [ modelname, perf, tot, mse, rmse, score, fl_out, model_run_uuid, run_id ]        
                estimator_perf.append(outputs)  

        all_predict_data["target"] = y_test
        e_perf = pd.DataFrame(estimator_perf)        
        e_perf.columns = ["Estimator", "Perf", "Total", "MSE", "RMSE", "Score", "Features", "UUID", "RUN_ID" ]
        
        correctX, correctY, correctP, totalX, cxp, cyp, cpp, r_predictions, r_y_target = calc_reg_ens_results(all_predict_data, estimator_run_ids)

        mse = mean_squared_error(r_y_target, r_predictions, squared=True)
        rmse =mean_squared_error(r_y_target, r_predictions, squared=False)
        r2 =r2_score(r_y_target, r_predictions)
        
        pear = scipy.stats.pearsonr(r_y_target, r_predictions) 
        r2 = pear
        
        mae = float(mean_absolute_error(r_y_target,r_predictions))                

        perf_data_t = [run_uuid, 0, e_perf.values.tolist(), features_list, 
                       correctX, correctY, correctP, totalX, cxp, cyp, cpp, mse, rmse, r2, mae]
        
        return perf_data_t    



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
                              
        perf_data_t = process_model(experiment_id, data, estimators, run_test_size, save_to_mlflow, feat_data)
        perf_data.append(perf_data_t)
        
        p_df = pd.DataFrame(perf_data)    
        p_df.columns = ["rid", "input_features", "e_perf", "features_list", "correctX", "correctY", 
                        "correctP", "totalX", "cxp", "cyp", "cpp", "mse", "rmse", "r2", "mae"]
        p_df.sort_values(by=['cpp'], ascending=False, inplace=True)

        print(" ")
        
        for x in range(len(p_df["e_perf"][0])):
                print(f"{p_df['e_perf'][0][x][0]}  {p_df['e_perf'][0][x][1]}  {p_df['e_perf'][0][x][5]}" )     

        if save_to_mlflow :

                step = 0
                time_stamp = dte_time.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
                exp_name = f"mixer_output_{time_stamp}"
                
                try:
                        experiment_id = mlflow.create_experiment(exp_name)
                except Exception as e:
                        print(f"{e}")    
                        
                experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id        
        
                for run_uuid, input_features, e_perf, features_list, correctX, correctY, correctP, totalX, cxp, cyp, cpp, mse, rmse, r2, mae in p_df.values.tolist() :
                
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

def run():

       

        xgb_3070 = {'learning_rate': 0.003170080749254201, 'max_depth': 32, 'subsample': 0.2957816844532192, 
        'colsample_bytree': 0.6594664699872866, 'min_child_weight': 8}

        lgb_3070 = {'learning_rate': 0.00297158669016989, 'num_leaves': 32, 'subsample': 0.5457131060645429, 
                        'colsample_bytree': 0.6206074333400939, 'min_data_in_leaf': 31,  'verbosity':-1 }

        cat_3070 = {'learning_rate': 0.012872913108877197, 'depth': 5, 'subsample': 0.9491103714261131, 
                'colsample_bylevel': 0.9771468169920741, 'min_data_in_leaf': 21}

        xgr = xgr_param_set()
        lbr = lbr_param_set()
        cbr = cbr_param_set()
        xg_rf = xgb_rf_params()

        cat_params_F={'learning_rate': 0.0360944196001379, 'depth': 10, 
                'subsample': 0.3523958110464825, 'colsample_bylevel': 0.6176118972551982, 
                        'min_data_in_leaf': 46 }

        xgb_params_F={'learning_rate': 0.004023993590803149, 'max_depth': 9, 'subsample': 0.5061891892307074, 
        'colsample_bytree': 0.6646068031525607, 'min_child_weight': 18}

        lgb_params_F={'learning_rate': 0.006961479110933946, 'num_leaves': 762, 
        'subsample': 0.5909033731294365, 'colsample_bytree': 0.8383929309109572, 
        'min_data_in_leaf': 80, 'verbosity':-1 }

        xgb_params_M={'learning_rate': 0.00264122394857379, 'max_depth': 8, 
        'subsample': 0.2772844546321145, 'colsample_bytree': 0.8118319429046319, 
        'min_child_weight': 7}

        lgb_params_M={'learning_rate': 0.004818774485749822, 'num_leaves': 9, 'subsample': 0.8313397546109982, 
        'colsample_bytree': 0.6285174849150702, 'min_data_in_leaf': 68, 'verbosity': -1 }

        cat_params_M={'learning_rate': 0.012193433669679433, 'depth': 7, 'subsample': 0.8003609726402594, 
        'colsample_bylevel': 0.9066114272514963, 'min_data_in_leaf': 34}

        xgb_rf_t={'learning_rate': 0.09992558454567729, 'max_depth': 4, 'subsample': 0.6295085012732937, 
                        'colsample_bytree': 0.507405257238443, 'min_child_weight': 12}

        xgb_rf_F={'learning_rate': 0.09947887382378602, 'max_depth': 6, 'subsample': 0.4532548971709517, 
                'colsample_bytree': 0.26550838751481926, 'min_child_weight': 10}

        xgb_rf_D={'learning_rate': 0.09959861108872929, 'max_depth': 8, 'subsample': 0.22688490349547857, 
                'colsample_bytree': 0.4775583435702645, 'min_child_weight': 15}


        est_list = [ 
                        XGBRegressor(),   
                        XGBRegressor(**xgb_3070),  
                        XGBRegressor(**xgr),  
                        XGBRegressor(**xgb_params_F), 
                        XGBRegressor(**xgb_params_M), 
                        CatBoostRegressor(),  
                        CatBoostRegressor(**cat_3070), 
                        CatBoostRegressor(**cbr),  
                        CatBoostRegressor(**cat_params_F), 
                        CatBoostRegressor(**cat_params_M),
                        LGBMRegressor(**lbr), 
                        LGBMRegressor(**lgb_3070), 
                        LGBMRegressor(**lbr), 
                        LGBMRegressor(**lgb_params_F), 
                        LGBMRegressor(**lgb_params_M), 
                        XGBRFRegressor(**xg_rf),
                        XGBRFRegressor()  
                ]


        # for 'data/buildSeqInd_Lucky13_5M_3070.csv',   #0
        features_87_FI = [
                'RSI',
                'STOK1',
                'SDKC9',
                'SDLR310',
                'ATR2',
                'SDKC91',
                'SDBB91',
                'ATR3',
                'ATR21',
        ]

        feat_777 = ['RSI', 'ATR21', 'ATR2', 'ATR3', 'SDKC91']
        

        datafile = [ 
                        'data/buildSeqInd_Lucky13_5M_3070.csv',   #0
                        'data/buildSeqInd_Lucky13_5M_ALL.csv',  #1
                        'data/buildSeqInd_Lucky13_F.csv',  #2
                        'data/buildSeqInd_Lucky13_D.csv',  #3
                        'data/buildSeqInd_Lucky13_F_3070.csv',  #4
                        'data/Expanded_Lucky13_3070.csv',  #5
                        'data/ndata_3070.csv', #6
                ]


        dtx = pd.read_csv(datafile[6])

        lucky13 = ['SDLR310','SDBB91','SDKC91','SDKC9','ROC','ATR34','ATR32','ATR31','ATR3','ATR21','ATR2','RSI','STOK1']

        all_feat = [
                ['SDLR3102', 'SDLR3101', 'SDLR310', 'VOLMA72'], 
                ['VOLMA71', 'VOLMA7', 'VOLMA132', 'VOLMA131', 'VOLMA13'],
                ['ZH212', 'ZH211', 'ZH21', 'ZH92', 'ZH91', 'ZH9'], 
                ['ZL212', 'ZL211', 'ZL21', 'ZL92', 'ZL91', 'ZL9'], 
                ['ZC212', 'ZC211', 'ZC21', 'ZC92', 'ZC91', 'ZC9'], 
                ['SDBB92', 'SDBB91', 'SDBB9'],
                ['SDBB202', 'SDBB201', 'SDBB20'],
                ['SDKC92', 'SDKC91', 'SDKC9'], 
                ['SDKC72', 'SDKC71', 'SDKC7'], 
                ['ROC142', 'ROC141', 'ROC14'], 
                ['ROC132', 'ROC131','ROC13'],
                ['ROC92', 'ROC91', 'ROC'],
                ['ATR144', 'ATR143', 'ATR142', 'ATR141', 'ATR14'], 
                ['ATR74', 'ATR73', 'ATR72', 'ATR71', 'ATR7'], 
                ['ATR34', 'ATR33', 'ATR32', 'ATR31', 'ATR3'],
                ['ATR54', 'ATR53', 'ATR52', 'ATR51', 'ATR5'],
                ['ATR24', 'ATR23', 'ATR22', 'ATR21', 'ATR2'], 
                ['RSI142', 'RSI141', 'RSI14'], 
                ['RSI92', 'RSI91', 'RSI'], 
                ['RSI72', 'RSI71', 'RSI7'], 
                ['RSI52', 'RSI51', 'RSI5'], 
                ['RSI32', 'RSI31', 'RSI3'], 
                ['STOK7142', 'STOK7141', 'STOK714'],
                ['STOK7212', 'STOK7211', 'STOK1'], 
                ['STOK52', 'STOK51', 'STOK5'],
        ]

        df_list = []

        for i in range(len(all_feat)):

            feat_data = all_feat[i]
            print(feat_data)
            
            split_test_size_value = 0.7          
            save_mlflow = False

            p_df, experiment_id_parent = run_models(dtx, est_list, split_test_size_value, save_mlflow, feat_data)
            df_list.append(p_df.values.tolist())


        d_list = []        
        for x in range(len(df_list)):
            
            for run_uuid, input_features, e_perf, features_list, correctX, correctY, correctP, totalX, cxp, cyp, cpp, mse, rmse, r2, mae in df_list[x]: 
                
                df_t = f"{run_uuid}  {correctX}  {correctY}  {correctP}  {totalX}  {cxp}  {cyp}  {cpp}  {mse}  {rmse}  {r2}  {mae} {all_feat[x]}"
                d_list.append(df_t)
                

        print(d_list)
        data_df = pd.DataFrame(d_list)
        data_df.to_csv("df.csv", index=False)        

if __name__ == "__main__":
    run()
