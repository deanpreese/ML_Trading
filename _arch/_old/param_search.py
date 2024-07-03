from datetime import datetime
import mlflow
import pandas as pd

from common.common_func import full_split_and_scale
from common.wrapped_models import TunableCatBoostClassifier

datafile = [ 
        'data/ReFried_5M_ALL.csv',   #0
        'data/buildSeqInd_13X_5M_ALL.csv',  #1
        'data/buildSeqIndX_5M_ALL.csv',  #2
        'data/buildSeqInd_Lucky13_5M_3070.csv',  #3
        'data/markov.csv',  #4
        'data/ReFried_5M_3070.csv',  #5
        'data/buildSeqInd_Lucky13_5M_ALL.csv',  #6
        'data/buildSeqInd_13X_5M_3070.csv',  #7
        'data/buildSeqIndX_5M_3070.csv',  #8
        'data/Fractal_ALL_5M.csv', #9 Classifier Only  
        'data/Fractal_ALL_5M_30_70.csv', #9 Classifier Only
    ]

filename = datafile[3]
data = pd.read_csv(filename)

time_stamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
exp_name = f"param_search_{time_stamp}"
    
experiment_id = mlflow.create_experiment(exp_name) if mlflow.get_experiment_by_name(exp_name) is None else mlflow.get_experiment_by_name(exp_name).experiment_id   
        
with mlflow.start_run(experiment_id = experiment_id):                

    estimator_perf = [] 
    X_train, X_test, y_train, y_test, input_features = full_split_and_scale(data, 2, 0.8, 0, 'outputC')  

    grid = TunableCatBoostClassifier().param_grid()  
    
    for p in grid:

        print(p) 
        model = TunableCatBoostClassifier(**p)
        model.track_model(experiment_id, True, model, X_train, y_train, X_test, y_test)
        