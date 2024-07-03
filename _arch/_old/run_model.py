from datetime import datetime

import mlflow
import pandas as pd
from common.common_func import full_split_and_scale
from common.other_wrapped_models import TunableAdaBoostRegressor, TunableGradientBoostingRegressor
from common.other_wrapped_models import TunableBaggingRegressor, TunableRandomForestRegressor
from common.wrapped_models import TunableCatBoostClassifier, TunableCatBoostRegressor, TunableLGBMClassifier, TunableLGBMRegressor, TunableXGBClassifier, TunableXGBRegressor


run_test_size = 0.8
    
def run_final_classifier(aggregate_predictions,exp_id):
    
    cls_perf = [] 
        
    X_train, X_test, y_train, y_test, input_features = full_split_and_scale(aggregate_predictions, 1, 0.2, 0, 'BTarget')  
    cb_c = TunableCatBoostClassifier()
    
    accuracy, precision, recall, TN, FP, FN, TP, tot, y_pred, pred_proba = cb_c.track_model(exp_id, True, cb_c, X_train, y_train, X_test, y_test)
    cls_perf_data = [cb_c, accuracy, precision, recall, TN, FP, FN, TP, tot]
    cls_perf.append(cls_perf_data)
        
    lg_c = TunableLGBMClassifier()
    accuracy, precision, recall, TN, FP, FN, TP, tot, y_pred, pred_proba = lg_c.track_model(exp_id, True, lg_c, X_train, y_train, X_test, y_test)
    lg_perf_data = [lg_c, accuracy, precision, recall, TN, FP, FN, TP, tot]
    cls_perf.append(lg_perf_data)
        
    xg_c = TunableXGBClassifier()
    accuracy, precision, recall, TN, FP, FN, TP, tot, y_pred, pred_proba = xg_c.track_model(exp_id, True, xg_c, X_train, y_train, X_test, y_test)
    xg_perf_data = [xg_c, accuracy, precision, recall, TN, FP, FN, TP, tot]
    cls_perf.append(xg_perf_data)
        
    return cls_perf        
    
    
def run_models(files, estimators):

    time_stamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    exp_name = f"run_model_{time_stamp}"
        
    experiment_id = mlflow.create_experiment(exp_name) if mlflow.get_experiment_by_name(exp_name) is None else mlflow.get_experiment_by_name(exp_name).experiment_id   

    estimator_perf = [] 
    classified_aggregate_predictions = pd.DataFrame()
    
    for filename in files:
    
        data = pd.read_csv(filename) 
        X_train, X_test, y_train, y_test, input_features = full_split_and_scale(data, 2, 0.8, 0, 'output')  

        for key in estimators.keys():
            
            estimators[key].features_used = list(X_train.columns)          
            
            perf, tot, mse, rmse, r2, score, mae, y_pred = estimators[key].track_model(experiment_id, True, estimators[key], X_train, y_train, X_test, y_test)      
            
            outputs = [ filename, key, perf, tot, mse, rmse, score, r2, mae ]        
            estimator_perf.append(outputs)  
            classified_aggregate_predictions[key] = y_pred
            
        classified_aggregate_predictions['FTarget'] = y_test 
        bin_out = classified_aggregate_predictions['FTarget'].apply(lambda x: 1 if x > 0 else 0)
        classified_aggregate_predictions['BTarget'] = bin_out 
        classified_aggregate_predictions = classified_aggregate_predictions.drop(['FTarget'], axis=1)
                
                
        return estimator_perf, classified_aggregate_predictions, experiment_id
    

datafile = [ 
        #'data/buildSeqInd_13X_5M_ALL.csv',   #0
        #'data/buildSeqIndX_5M_ALL.csv',   #1
        #'data/buildSeqInd_Lucky13_5M_3070.csv',   #2
        #'data/markov.csv',  #3
        'data/buildSeqInd_Lucky13_5M_ALL.csv',  #4
        #'data/buildSeqInd_13X_5M_3070.csv',  #5
        #'data/buildSeqIndX_5M_3070.csv',   #6
    ]


gp1 = {'n_jobs': 4,'verbose' : 2, }
gp2 = {'verbose' : 2, }

models = {}
models['cat'] = TunableCatBoostRegressor(**TunableCatBoostRegressor().param_set())
models['xgb'] = TunableXGBRegressor(**TunableXGBRegressor().param_set())
models['lgb'] = TunableLGBMRegressor(**TunableLGBMRegressor().param_set())
models['rf'] = TunableRandomForestRegressor(**gp2)
models['br'] = TunableBaggingRegressor(**gp2)
models['gb']  = TunableGradientBoostingRegressor(**gp2)
models['ada'] = TunableAdaBoostRegressor()


estimator_perf, classified_aggregate_predictions, experiment_id = run_models(datafile, models)
#cls_perf = run_final_classifier(classified_aggregate_predictions,experiment_id)
        
e_df = pd.DataFrame(estimator_perf)        
e_df.columns = ["File", "Estimator", "Perf", "Total", "MSE", "RMSE", "Score", "R2", "MAE"]    
print(" ")
print(e_df)    
print(" ")

#p_df = pd.DataFrame(cls_perf)    
#p_df.columns = ["Model", "Accuracy", "Precision", "Recall", "True Neg", "False Pos", "False Neg", "True Pos", "Total"]
#p_df.sort_values(by=['Accuracy'], ascending=False, inplace=True)
#print(" ")
#print(p_df)
#print(" ")
 