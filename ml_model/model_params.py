cbc_set = {
    'learning_rate': 0.009, 
    'depth': 3, 
    'l2_leaf_reg': 3.0, 
    'min_child_samples': 32, 
    'iterations': 1000, 
    #'random_state': [0], 
    #'thread_count': [-1],
}

lbc_set = { 
    'boosting_type': 'gbdt', 
    'num_leaves': 31, 
    'learning_rate': 0.1, 
    'n_estimators': 10,
    'objective': None, 
    'min_child_samples': 20, 
    'subsample': 1.0, 
    'colsample_bytree': 1.0, 
    'random_state': 0, 
    'n_jobs': -1,
}

xgc_set = {
    'colsample_bytree': 0.6655392754230048, 
    'gamma': 4.198875359789924, 
    'max_depth': 17.0, 
    'min_child_weight': 1.0,  
    'reg_alpha': 57.0,  
    'reg_lambda': 0.896332305739873,
}

# =============================================================================

xgb_p ={'lambda': 7.0571738749147706, 'alpha': 7.55600857346679, 'eta': 0.7, 'gamma': 20, 
        'learning_rate': 0.01, 'colsample_bytree': 0.6, 'colsample_bynode': 0.9, 'n_estimators': 876, 
        'min_child_weight': 102, 'max_depth': 5, 'subsample': 1.0
        }


xgbr_set = { 
    #'max_depth': 3,
    'booster': 'dart', 
    'learning_rate': 0.1, 
    'n_estimators': 100, 
    #'gamma': [0, 20],  
    #'subsample': [0.8,1], 
    #'colsample_bytree': [0.8,1], 
    #'lambda': [0, 0.1, 1],
    'tree_method': "hist",
    #'eval_metric': "reg:squarederror",
    "objective": "reg:squarederror",
    "verbosity": 2,
}

xgbr_set2 = {
    #'max_depth': 3,
    'booster': 'gbtree', 
    'learning_rate': 0.1,
    'n_estimators': 100, 
    #'gamma': [0, 20], 
    #'subsample': [0.8,1], 
    #'colsample_bytree': [0.8,1], 
    #'lambda': [0, 0.1, 1],
    'tree_method': "hist",
    #'eval_metric': "mae",
    "objective": "reg:squarederror",
    "verbosity": 2,
}
        

xgb_params_F = {
    'learning_rate': 0.004023993590803149, 
    'max_depth': 9, 
    'subsample': 0.5061891892307074, 
    'colsample_bytree': 0.6646068031525607, 
    'min_child_weight': 18,
    "objective": "reg:squarederror",
}

xgb_3070 = {
    'learning_rate': 0.003170080749254201, 
    'max_depth': 32, 
    'subsample': 0.2957816844532192, 
    'colsample_bytree': 0.6594664699872866, 
    'min_child_weight': 8,
    "objective": "reg:squarederror",
}



xgb_params_M = {
    'learning_rate': 0.00264122394857379, 
    'max_depth': 8, 
    'subsample': 0.2772844546321145, 
    'colsample_bytree': 0.8118319429046319, 
    'min_child_weight': 7,
    "objective": "reg:squarederror",
}

# ==============

lgb_x = {'reg_alpha': 0.005952046197492608, 'reg_lambda': 0.08307352697498646, 
         'colsample_bytree': 0.8, 'subsample': 0.4, 'learning_rate': 0.006, 'max_depth': 20, 
         'num_leaves': 10, 'min_child_samples': 131, 'min_data_per_groups': 76}


lgb_params_F = {
    'learning_rate': 0.006961479110933946, 
    'num_leaves': 762, 
    'subsample': 0.5909033731294365, 
    'colsample_bytree': 0.8383929309109572, 
    'min_data_in_leaf': 80, 
    'verbosity': -1,
}

lbr_set = { 
    'n_estimators': 150, 
    'objective': 'regression', 
    'min_child_samples': 7, 
    'subsample': 1,
    'num_leaves': 35, 
    'colsample_bytree': 1, 
    'random_state': 0, 
    'n_jobs': -1, 
    'learning_rate': 0.01, 
    'verbose': 1,
}


lgb_3070 = {
    'learning_rate': 0.00297158669016989, 
    'num_leaves': 32, 
    'subsample': 0.5457131060645429, 
    'colsample_bytree': 0.6206074333400939, 
    'min_data_in_leaf': 31,  
    'verbosity': -1,
}


lgb_params_M = {
    'learning_rate': 0.004818774485749822, 
    'num_leaves': 9, 
    'subsample': 0.8313397546109982, 
    'colsample_bytree': 0.6285174849150702, 
    'min_data_in_leaf': 68, 
    'verbosity': -1,
}

# ==============

cat_params_M = {
    'learning_rate': 0.012193433669679433, 
    'depth': 7, 
    'subsample': 0.8003609726402594, 
    'colsample_bylevel': 0.9066114272514963, 
    'min_data_in_leaf': 34,
}


cbr_set = {
    "iterations": 800, 
    "learning_rate": 0.01, 
    "depth": 7,
}

cat_3070 = {
    'learning_rate': 0.012872913108877197, 
    'depth': 5, 
    'subsample': 0.9491103714261131, 
    'colsample_bylevel': 0.9771468169920741, 
    'min_data_in_leaf': 21,
}

cat_params_F = {
    'learning_rate': 0.0360944196001379, 
    'depth': 10, 
    'subsample': 0.3523958110464825, 
    'colsample_bylevel': 0.6176118972551982, 
    'min_data_in_leaf': 46,
}


# ==============

xgbrf_t = {
    'learning_rate': 0.09992558454567729, 
    'max_depth': 4, 
    'subsample': 0.6295085012732937, 
    'colsample_bytree': 0.507405257238443, 
    'min_child_weight': 12,
}

xgbrf_F = {
    'learning_rate': 0.09947887382378602, 
    'max_depth': 6, 
    'subsample': 0.4532548971709517, 
    'colsample_bytree': 0.26550838751481926, 
    'min_child_weight': 10,
}

xgbrf_D = {
    'learning_rate': 0.09959861108872929, 
    'max_depth': 8, 
    'subsample': 0.22688490349547857, 
    'colsample_bytree': 0.4775583435702645, 
    'min_child_weight': 15,
}

xgbrf_set = {
    "colsample_bynode": 0.8,
    "learning_rate": 0.1,
    "max_depth": 5,
    "num_parallel_tree": 100,
    "objective": "reg:squarederror",
    "subsample": 0.8,
    "tree_method": "hist",
    #"device": "cuda",
}