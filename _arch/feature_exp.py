import datetime
import warnings
import mlflow

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel

from common.common_func import full_split_and_scale, full_split_and_scale_with_filter
from common.other_wrapped_models import TunableAdaBoostRegressor, TunableBaggingRegressor, TunableGradientBoostingRegressor, TunableRandomForestRegressor  # noqa: F401
from common.wrapped_models import TunableCatBoostRegressor, TunableLGBMRegressor, TunableXGBRegressor  # noqa: F401
warnings.filterwarnings("ignore")


def sel_from(file, estimator, experiment_id):
    
    data = pd.read_csv(file) 
    X_train, X_test, y_train, y_test, input_features = full_split_and_scale(data, 2, 0.8, 0, 'output')
    features_used = X_train.columns       
    
    params_x=   {'n_estimators': 100, 
                     'n_jobs' : 4,   
                     'verbose' : 2
                }
    
    model = ExtraTreesRegressor(**params_x)
    
    for feature_count in (range(100, 5, -2)):
        
        mx_feat = int((len(features_used)-2) * (feature_count/100))
        
        #select = SelectFromModel(model, threshold=0.03, max_features=mx_feat)
        select = SelectFromModel(model, max_features=mx_feat)

        select.fit_transform(X_train, y_train)
        cols_idxs = select.get_support(indices=True)

        col_names = []
        #col_index_map = {col_name: i for i, col_name in enumerate(X_train.columns)}
        for c in cols_idxs:
            col_names.append(X_train.columns[c])
        
        X_train, X_test, y_train, y_test, input_features, cols_used = full_split_and_scale_with_filter(data, 2, 0.8, 0, 'output',cols_idxs)
        
        if X_train.shape[1] < 1 :
            print("No Features selected")
            break
        else:        
            features_used = cols_idxs 
            estimator.features_used = col_names                                                  
            estimator.track_model(experiment_id, True, estimator, X_train, y_train, X_test, y_test)         


def seq_select(file, feat_min, feat_step, estimator, experiment_id, reverse ):
    
    data = pd.read_csv(file) 
    X_train, X_test, y_train, y_test, input_features = full_split_and_scale(data, 2, 0.8, 0, 'output')
    features_used = X_train.columns
    
    for targetCount in (range(len(features_used), feat_min, feat_step*-1)):
                
        if targetCount < abs(feat_step):
                break
            
        print( "Doing Feature Extraction" )
        
        #col_index_map = {col_name: i for i, col_name in enumerate(X_train.columns)}                
                
        estimator.features_used = features_used                         
        estimator.track_model(experiment_id, True, estimator, X_train, y_train, X_test, y_test)                              

        feature_importance = list(zip(data.columns[:len(features_used)], estimator.feature_importances_))
        sorted_feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=reverse)

        features_used = []

        fcnt = 0
        for feature, weight in sorted_feature_importance:
                #txt = f"Feature: {feature}, Weight: {weight}"
                if fcnt <= targetCount:
                        #print(f"{txt}")
                        features_used.append( f'{feature}'  )
                fcnt = fcnt +1
        
        features_used.append("output")
        
        data = pd.read_csv(file) 
        data = data[features_used]
        X_train, X_test, y_train, y_test, input_features = full_split_and_scale(data, 2, 0.8, 0, 'output')
        print( f"{estimator}  {len(features_used)}  Features Selected" )        
        print(" ")    


def runFile(file, estimators, min_cols, col_step):

       
        time_stamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        exp_name = f"feature_exp_{time_stamp}"
            
     
        experiment_id = mlflow.create_experiment(exp_name) if mlflow.get_experiment_by_name(exp_name) is None else mlflow.get_experiment_by_name(exp_name).experiment_id   

        for e in estimators.keys():  
            seq_select(file, min_cols, col_step, estimators[e], experiment_id, False  )
            sel_from(file, estimators[e], experiment_id)
        
        

datafile = [ 
        'data/buildSeqInd_13X_5M_ALL.csv',   #0
        'data/buildSeqIndX_5M_ALL.csv',   #1
        'data/buildSeqInd_Lucky13_5M_3070.csv',   #2
        'data/markov.csv',  #3
        'data/buildSeqInd_Lucky13_5M_ALL.csv',  #4
        'data/buildSeqInd_13X_5M_3070.csv',  #5
        'data/buildSeqIndX_5M_3070.csv',   #6
    ]


gp2 = {'verbose' : 2, }

models = {}
models['lgb'] = TunableLGBMRegressor(**TunableLGBMRegressor().param_set())
models['xgb'] = TunableXGBRegressor(**TunableXGBRegressor().param_set())
models['cat'] = TunableCatBoostRegressor(**TunableCatBoostRegressor().param_set())
#models['rf'] = TunableRandomForestRegressor(**gp2)
#models['br'] = TunableBaggingRegressor(**gp2)
#models['gb']  = TunableGradientBoostingRegressor(**gp2)
#models['ada'] = TunableAdaBoostRegressor()



#runFile(file, estimator, min_cols, col_step):
runFile(datafile[4], models, 2, 1)

        
        

               