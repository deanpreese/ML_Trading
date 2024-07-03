from common_func import * 
from models.wrapped_models import *

from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import matplotlib as mpl 
import seaborn as sns

import mlflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:8888")

import warnings
warnings.filterwarnings("ignore")

def run_data_files(experiment_id, datafiles, estimators, min_features):

    output_data = []
    min_output_data = []

    for filename in datafiles:

        data = pd.read_csv(filename)   
        run_test_size = 0.8

        num_columns = len(data.axes[1]) 
        input_features =  num_columns -2
        X = data.iloc[:, 0:input_features]  
        y = data['output'].values

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=run_test_size, random_state=0)
        
        with mlflow.start_run(experiment_id = experiment_id, nested=True):

            for r in estimators:
            
                #r.SetFeatureList(list(X_train.columns))           
                #perf_orig, tot_orig, mse_orig, rmse_orig, r2_orig, score_orig, mae_orig, y_pred  = r.TrackInMLFlow(experiment_id,True, r ,#X_train, y_train, X_test, y_test)        
                #fi_orig = r.feature_importances_
                #col_orig = X.columns
               
                selector = TunableRFECV( 
                            estimator=r,
                            min_features_to_select=min_features,
                            step=2,
                            n_jobs=4,
                            #scoring='neg_mean_squared_error',
                            cv=3,
                            verbose=1,
                    )
                
                selector.SetRFECVExperiment(experiment_id)
                    
                selector = selector.fit(X_train, y_train)
                cv_mean = selector.cv_results_["mean_test_score"]
                cv_std = selector.cv_results_["std_test_score"] 
                
                y_pred = selector.predict(X_test)

                
    return output_data

  
datafile = [ 
        #'data/ReFried_5M_ALL.csv',
        #'data/buildSeqInd_13X_5M_ALL.csv',
        #'data/buildSeqIndX_5M_ALL.csv',
        #'data/buildSeqInd_Lucky13_5M_3070.csv',
        #'data/markov.csv',
        #'data/ReFried_5M_3070.csv',
        'data/buildSeqInd_Lucky13_5M_ALL.csv',
        #'data/buildSeqInd_13X_5M_3070.csv',
        #'data/buildSeqIndX_5M_3070.csv',
    ]

min_features = 5


time_stamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
exp_name = f"features_exp_{time_stamp}"

try:
        experiment_id = mlflow.create_experiment(exp_name)
except:
        
        experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id        
        mlflow.set_experiment(experiment_id=experiment_id)


#estimators = [ TunableRandomForestRegressor() ]  
#estimators = [ TunableLGBMRegressor() ]  

estimators = [ GradientBoostingRegressor() ]  

#estimators = [ TunableCatBoostRegressor() ]  

final_data = run_data_files(experiment_id, datafile, estimators, min_features)
print(" ")
df = pd.DataFrame(final_data)
print(df)


#full_output_tuple = (input_features,perf_orig, total_orig, mse_orig, rmse_orig, fi_orig, col_orig, len(selected_features_data), sel_perf, sel_tot, sel_mse , sel_rmse, cols_idxs, fi_sel, col_sel, cv_mean, cv_std )                       
#output_tuple = (perf_orig, total_orig, mse_orig, rmse_orig, len(selected_features_data), sel_perf, sel_tot, sel_mse , sel_rmse )                                   
            
#e_i = 0

#df = pd.DataFrame(min_final_data)
#print(df)

#
#for model_name, input_features, perf_orig, total_orig, mse_orig, rmse_orig, fi_orig, col_orig, sel_feat, sel_perf, sel_tot, sel_mse , sel_rmse, cols_idxs, fi_sel, col_sel, cv_mean, cv_std in final_data:

#       cl = estimators[e_i].model.__class__.__name__

#        txt = f"{cl}, {input_features},  {perf_orig},  {total_orig},  {mse_orig},  {rmse_orig},  {sel_feat},  {sel_perf},  {sel_tot},  {sel_mse},  {sel_rmse}" 
#        txt2 = f"{cl},   {sel_feat},   {sel_perf},   {sel_tot},   {sel_mse},   {sel_rmse},   {cols_idxs}   {col_sel}" 
        
#        e_i += 1
        
#        print(f" {txt2} ")
    