from common_func import *
from models.wrapped_models import *

import warnings
warnings.filterwarnings("ignore")

def process_model(exp_name, data,  randomize, models, run_test_size, feature_list_size):
        
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
                
                if randomize:
                        idxx = random.sample(range(1, len(data.axes[1]) -2 ), feature_list_size)
                        features_list.append(idxx)    
                        fl= features_list[f]
                        X = data.iloc[:, fl]  
                        input_features = len(X.axes[1]) 
                        fl_out = fl
              

                X_train, X_test, y_train, y_test = simple_split_and_scale(X, y, run_test_size, 0)
                
                perf, tot, mse, rmse, r2, score, mae, predictions = e.TrackInMLFlow(exp_name, True, e, X_train, y_train, X_test, y_test)  
                all_predict_data[model_run_uuid] = predictions
                perf, tot = show_stats(False, y_test, predictions)
                mse, rmse  = calc_MSE(y_test, predictions, False)
                score = e.score(X_test, y_test)

                combined_prod_perf = predictions * perf
                nm = f"{model_run_uuid}_p"
                all_predict_data[nm] = combined_prod_perf
                
                estimator_run_ids.append(model_run_uuid)
                
                outputs = [ modelname, perf, tot, mse, rmse, score, fl_out, model_run_uuid ]        
                estimator_perf.append(outputs)  

        all_predict_data["target"] = y_test
        e_perf = pd.DataFrame(estimator_perf)        
        e_perf.columns = ["Estimator", "Perf", "Total", "MSE", "RMSE", "Score", "Features", "UUID" ]
        
        correctX, correctY, correctP, totalX, cxp, cyp, cpp = calc_reg_results(all_predict_data, estimator_run_ids)


        perf_data_t = [run_uuid, feature_list_size, e_perf.values.tolist(), features_list, correctX, correctY, correctP, totalX, cxp, cyp, cpp]
        
        return perf_data_t    



def GenerateResults(perf_data):
        p_df = pd.DataFrame(perf_data)    
        p_df.columns = ["rid", "input_features", "e_perf", "features_list", "correctX", "correctY", "correctP", "totalX", "cxp", "cyp", "cpp" ]
        p_df.sort_values(by=['cpp'], ascending=False, inplace=True)
        return p_df




def run_models(data, write_to_file, random_features, file_out, estimators, run_test_size, min_features, max_features, step_features, total_cycles ):
        
        feature_list_size = min_features
        p_df = pd.DataFrame()
        
        if len(data.columns) < max_features:
                max_features = len(data.columns) - 3
        
        
        time_stamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        exp_name = f"meta_runs_{time_stamp}"
        
        try:
            experiment_id = mlflow.create_experiment(exp_name)
        except:
            experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id        
            
        with mlflow.start_run(experiment_id = experiment_id):
        
                if random_features:
                        
                        perf_data = []
                        e_df_list = []
                        
                        for q in range(total_cycles):
                        
                                with mlflow.start_run(experiment_id = experiment_id, nested=True):
                        
                                        for f in range(min_features, max_features, step_features):
                                        
                                                random_features = True
                                                feature_list_size = f
                                                
                                                perf_data_t = process_model(experiment_id, data,  random_features, estimators, run_test_size, feature_list_size)
                                                perf_data.append(perf_data_t)
                                        
                                        p_df = GenerateResults(perf_data)
                                        
                        if write_to_file:
                                p_df.to_csv(file_out)
                        
                                                
                else:   
                        perf_data = []
                        
                        perf_data_t =process_model(exp_name, data,  random_features, estimators, run_test_size, feature_list_size)
                        perf_data.append(perf_data_t)
                        p_df = GenerateResults(perf_data)
                        
        return p_df, experiment_id
                

def LogFinalResults(p_df, experiment_id_parent):
        
        step = 0

        time_stamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        exp_name = f"meta_output_{time_stamp}"
                
        
        try:
            experiment_id = mlflow.create_experiment(exp_name)
        except:
            experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id        
                            
                
        with mlflow.start_run(experiment_id = experiment_id):                

                
                mlflow.log_table(data=p_df, artifact_file="all_results.json")                
                        
                for run_uuid, input_features, e_perf, features_list, correctX, correctY, correctP, totalX, cxp, cyp, cpp in p_df.values.tolist() :
                
                        with mlflow.start_run(experiment_id = experiment_id, nested=True): 
                                       
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
                                
                                mlflow.log_table(data=pd.DataFrame(e_perf), artifact_file="perf_data.json")        
                                mlflow.log_table(data=pd.DataFrame(features_list), artifact_file="features_list.json")        
                                
                                step += 1        

                
# ---------------------------
#
# Run the models
#
# ---------------------------

datafile = [ 
        'data/ReFried_5M_ALL.csv',
        'data/buildSeqInd_13X_5M_ALL.csv',
        'data/buildSeqIndX_5M_ALL.csv',  # 93 cols
        'data/buildSeqInd_Lucky13_5M_3070.csv',
        'data/markov.csv',
        'data/ReFried_5M_3070.csv',
        'data/buildSeqInd_Lucky13_5M_ALL.csv',
        'data/buildSeqInd_13X_5M_3070.csv',
        'data/buildSeqIndX_5M_3070.csv',
    ]


dtx = pd.read_csv(datafile[2])
# 19 Cols available
#  3-4-5-6 make up 80% of top 100
#f_out = "output/13x_ALL_3M_mm_cbr_lgb.csv"
#est_list = [ CBRModel(), LGBModel(), ]


# 93 cols available
#  3-4-5-6 make up 75% of top 100
f_out = "output/mm_cbr_LGB_IndX_ALL_DIFF_15_BIG.csv"
#est_list = [ TunableCatBoostRegressor(), TunableLGBMRegressor(), TunableXGBRegressor() ]
est_list = [ TunableCatBoostRegressor(), TunableCatBoostRegressor(), TunableCatBoostRegressor() ]


to_file = False
#to_file = True
#randomize_features = False
randomize_features = True

split_test_size_value = 0.8          

min_features_used = 10
max_features_used = 50
step_features_used = 1
total_cycles_used = 5



p_df, experiment_id_parent = run_models(dtx, to_file, randomize_features, f_out, est_list, split_test_size_value, min_features_used, max_features_used, step_features_used, total_cycles_used  )
LogFinalResults(p_df, experiment_id_parent )

print(" ")
print(p_df)                
print(" ")
