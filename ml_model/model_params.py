from itertools import product


def create_param_list(grid):
    param_combinations = list(product(*grid.values()))
    par_list = []
    for params in param_combinations:
        param_set = dict(zip(grid.keys(), params))
        par_list.append(param_set)
    return par_list


def cbc_param_set():    
    params = {
        'learning_rate' : 0.009,
        'depth' : 3,
        'l2_leaf_reg' :  3.0,
        'min_child_samples' : 32,
        'iterations' : 1000,
        #'random_state' : [0],
        #'thread_count' : [-1],
    }
    return params

def cbc_param_grid():
    params = {
        'learning_rate' : [ 0.009, 0.01],
        #'depth' : [3,5,7,9,11],
        'depth' : [3,7,9,11],
        #'l2_leaf_reg' : [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0 ],
        'l2_leaf_reg' : [ 3.0, 4.0, 5.0 ],
        #'min_child_samples' : [1, 4, 8, 16, 32],
        'min_child_samples' : [ 8, 16, 32],
        #'grow_policy' : ['Depthwise'],
        'iterations' : [1000],
        #'eval_metric' : ['RMSE'],
        #'random_state' : [0],
        #'boosting_type' : ['Ordered', 'Plain'],
        #'thread_count' : [-1],
    }
    return create_param_list(params)
    
def lbc_param_set():    
    params = {
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'n_estimators': 10,
        'objective': None,
        'min_child_samples': 20,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'random_state': 0,
        'n_jobs': -1
    }
    return params
    
def lbc_param_grid():
    params = {
        'boosting_type': 'gbdt',
        #'class_weight': None,
        #'colsample_bytree': 1.0,
        #'importance_type': 'split',
        'learning_rate': 0.1,
        'max_depth': -1,
        'min_child_samples': 20,
        #'min_child_weight': 0.001,
        #'min_split_gain': 0.0,
        'n_estimators': 100,
        'num_leaves': 31,
        #'objective': None,
        #'random_state': None,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'subsample': 1.0,
        'subsample_for_bin': 200000,
        'subsample_freq': 0,            
    }
    return create_param_list(params)
        
def xgc_param_set():
    params = {
        'colsample_bytree': 0.6655392754230048, 
        'gamma': 4.198875359789924, 
        'max_depth': 17.0, 
        'min_child_weight': 1.0, 
        'reg_alpha': 57.0, 
        'reg_lambda': 0.896332305739873
        }
    return params
    
def xgc_param_grid():
    params = { 
        'max_depth': [ 3, 18, 1],
        'gamma': [1,9],
        'reg_alpha' : [40,180,1],
        'reg_lambda' : [ 0,1],
        'colsample_bytree' : [0.5,1],
        'min_child_weight' : [0, 10],
        'n_estimators': [180],
        'seed': [0]
    }
    return create_param_list(params)        

# =============================================================================

def cbr_param_set():
    params = {
        "iterations": 800,
        "learning_rate": 0.01,
        "depth": 7,
        #"subsample": [0.05, 0.07, 1.0],
        #"colsample_bylevel": [ 0.05, 0.07,  1.0],
        #"min_data_in_leaf": [ 1, 5, 25, 50, 100],

    }
    return params
    
def cbr_param_grid():
    params = {
        "iterations": [600,700,800,900 ],
        "learning_rate": [0.01, 0.3, 0.7, 0.1 ],
        "depth": [ 1, 3, 5, 7, 10],
        #"subsample": [0.05, 0.07, 1.0],
        #"colsample_bylevel": [ 0.05, 0.07,  1.0],
        #"min_data_in_leaf": [ 1, 5, 25, 50, 100],
    } 
    return create_param_list(params)

def lbr_param_set():
    params= {
        'n_estimators' : 150,
        'objective': 'regression',
        'min_child_samples' : 7,
        'subsample' : 1,
        'num_leaves': 35,
        'colsample_bytree' : 1,
        'random_state' : 0,
        'n_jobs' : -1,
        'learning_rate': 0.01,
        'verbose': 1,
        }
    return params        

def lbr_param_grid():
    params = {
        'n_estimators' : [150],
        #'boosting_type': ['gbdt', 'rf', 'dart'],
        #'objective': ['regression'],
        'min_child_samples' : [5,7,9],
        #'subsample' : [1],
        'num_leaves': [19,21,23,25,30,35],
        #'colsample_bytree' : [1,2,3],
        'random_state' : [0],
        'n_jobs' : [-1],
        #'learning_rate': [0.01, 0.02, 0.03, 0.04 ],
        'learning_rate': [0.01],
        'verbose': [1],
    }    
    return create_param_list(params)


def xgr_param_set():
    params = {
        'max_depth': 3,
        'booster' : 'dart', 
        'learning_rate': 0.1,
        'n_estimators':  50, 
        #'gamma': [0, 20], 
        #'subsample': [0.8,1], 
        #'colsample_bytree': [0.8,1], 
        #'lambda': [0, 0.1, 1],
        'tree_method': "hist",
        'eval_metric': "mae",
        "verbosity" : 2
    }
    return params

def xgr_param_set2():
    params = {
        #'max_depth': 3,
        'booster' : 'gbtree', 
        #'learning_rate': 0.1,
        'n_estimators':  100, 
        #'gamma': [0, 20], 
        #'subsample': [0.8,1], 
        #'colsample_bytree': [0.8,1], 
        #'lambda': [0, 0.1, 1],
        'tree_method': "hist",
        #'eval_metric': "mae",
        "verbosity" : 2
    }
    return params
        
def xgr_param_grid():
    params = {
        'max_depth': [2, 3, 4, 5, 6], 
        'learning_rate': [0.1, 0.2, 0.3],
        'n_estimators': [25, 50, 100, 150], 
        #'gamma': [0, 20], 
        #'subsample': [0.8,1], 
        #'colsample_bytree': [0.8,1], 
        #'lambda': [0, 0.1, 1],
        'tree_method': ["hist"],
        'eval_metric': ["mae"],
        "verbosity" : [2]
    }
    return create_param_list(params)      



def xgb_rf_params():
    params = {
    "colsample_bynode": 0.8,
    "learning_rate": 1,
    "max_depth": 5,
    "num_parallel_tree": 100,
    "objective": "reg:squarederror",
    "subsample": 0.8,
    "tree_method": "hist",
    #"device": "cuda",
    }
    return params    