
import datetime
import mlflow
import pandas as pd

from common_func import full_split_and_scale
from models.wrapped_models import TunableCatBoostClassifier, TunableCatBoostRegressor, TunableLGBMRegressor, TunableXGBRegressor

datafile = [ 
        #'data/ReFried_5M_ALL.csv',
        #'data/buildSeqInd_13X_5M_ALL.csv',
        'data/buildSeqIndX_5M_ALL.csv',
        #'data/buildSeqInd_Lucky13_5M_3070.csv',
        #'data/markov.csv',
        #'data/ReFried_5M_3070.csv',
        #'data/buildSeqInd_Lucky13_5M_ALL.csv',
        #'data/buildSeqInd_13X_5M_3070.csv',
        #'data/buildSeqIndX_5M_3070.csv',
    ]

data = pd.read_csv(datafile[0])
   
X_train, X_test, y_train, y_test, input_features = full_split_and_scale(data, 2, 0.8, 0, 'output')  

time_stamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
exp_name = f"3-2-1_{time_stamp}"

try:
        experiment_id = mlflow.create_experiment(exp_name)
except Exception as e:
        print(f"Exception loading experiment_id {e}")
        experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id        
        mlflow.set_experiment(experiment_id=experiment_id)


with mlflow.start_run(experiment_id = experiment_id, nested=False):

    models = {}
    
    aggregated_predictions = pd.DataFrame()
    
    models['cat'] = TunableCatBoostRegressor()
    models['xgb'] = TunableXGBRegressor()
    models['lgbm'] = TunableLGBMRegressor()
    
    
    estimator_perf = []
    cls_perf = [] 
    
    for key in models.keys():
        perf, tot, mse, rmse, r2, score, mae, y_pred = models[key].TrackInMLFlow(experiment_id, True, models[key], X_train, y_train, X_test, y_test)

        outputs = [key,  perf, tot, mse, rmse, score, r2, mae ]        
        estimator_perf.append(outputs)  
        aggregated_predictions[key] = y_pred
        
    aggregated_predictions['FTarget'] = y_test 
    bin_out = aggregated_predictions['FTarget'].apply(lambda x: 1 if x > 0 else 0)
    aggregated_predictions['BTarget'] = bin_out 
    aggregated_predictions = aggregated_predictions.drop(['FTarget'], axis=1)
    
    X_train, X_test, y_train, y_test, input_features = full_split_and_scale(aggregated_predictions, 1, 0.8, 0, 'BTarget')  
    cb_c = TunableCatBoostClassifier()
    accuracy, precision, recall, TN, FP, FN, TP, tot, y_pred, pred_proba = cb_c.track_model(experiment_id, True, cb_c, X_train, y_train, X_test, y_test)
    cls_perf_data = [cb_c, accuracy, precision, recall, TN, FP, FN, TP, tot]
    cls_perf.append(cls_perf_data)
    
    
e_df = pd.DataFrame(estimator_perf)        
e_df.columns = ["Estimator", "Perf", "Total", "MSE", "RMSE", "Score", "R2", "MAE"]        
print(" ")
print(e_df)
print(" ")

p_df = pd.DataFrame(cls_perf)    
p_df.columns = ["Model", "Accuracy", "Precision", "Recall", "True Neg", "False Pos", "False Neg", "True Pos", "Total"]
p_df.sort_values(by=['Accuracy'], ascending=False, inplace=True)
print(" ")
print(p_df)
print(" ")
 