# --- 
# params 10/10/2024


catc_lucky13 = {'iterations': 162, 'learning_rate': 0.010463258885379521, 'objective': 'Logloss', 'colsample_bylevel': 0.01161504658137422, 'depth': 2, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 0.7414129195847714}
xgbc_lucky13 = {'booster': 'dart', 'lambda': 2.8409026532087042e-06, 'alpha': 0.6635352958565828, 'subsample': 0.28996448676651215, 'colsample_bytree': 0.43832008451333576, 'max_depth': 3, 'min_child_weight': 4, 'eta': 1.579079759343244e-07, 'gamma': 9.282358728683162e-06, 'grow_policy': 'lossguide', 'sample_type': 'uniform', 'normalize_type': 'tree', 'rate_drop': 1.1429793349532624e-06, 'skip_drop': 0.0003874670877712408}
lgbc_lucky13 = {'lambda_l1': 0.7939061182280855, 'lambda_l2': 2.7785406431968906e-06, 'num_leaves': 142, 'feature_fraction': 0.9315397361540054, 'bagging_fraction': 0.8717606696848038, 'bagging_freq': 4, 'min_child_samples': 53}

xgbr_lucky13 = {'lambda': 8.545299921698266, 'alpha': 16.993322455483007, 'eta': 0.7, 'gamma': 25, 'learning_rate': 0.012, 'colsample_bytree': 0.9, 'colsample_bynode': 1.0, 'n_estimators': 739, 'min_child_weight': 105, 'max_depth': 7, 'subsample': 1.0}
catr_lucky13 = {'learning_rate': 0.003583339802324007, 'depth': 10, 'subsample': 0.4529600813178114, 'colsample_bylevel': 0.7439504747396157, 'min_data_in_leaf': 43}
lgbr_lucky13 = {'reg_alpha': 0.008016904246004472, 'reg_lambda': 0.009356818652445557, 'colsample_bytree': 0.8, 'subsample': 0.8, 'learning_rate': 0.006, 'max_depth': 100, 'num_leaves': 134, 'min_child_samples': 102, 'min_data_per_groups': 72}


# ------

catc_t = {'objective': 'Logloss', 'colsample_bylevel': 0.015411304224424851, 'depth': 3, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 8.820225862779308}
xgbc_t = {'max_depth': 11, 'subsample': 0.8, 'n_estimators': 9600, 'eta': 0.09999999999999999, 'reg_alpha': 2, 'reg_lambda': 7, 'min_child_weight': 8, 'colsample_bytree': 0.41751610958110463}
lgbc_t = {'lambda_l1': 3.448361997410009e-06, 'lambda_l2': 0.0184646060486577, 'num_leaves': 163, 'feature_fraction': 0.8823675284869981, 'bagging_fraction': 0.6100669161739707, 'bagging_freq': 5, 'min_child_samples': 13}

cbc_set = {'learning_rate': 0.009, 'depth': 3, 'l2_leaf_reg': 3.0, 'min_child_samples': 32, 'iterations': 1000,}

lbc_set = { 'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.1, 'n_estimators': 10, 'min_child_samples': 20, 
    'subsample': 1.0, 'colsample_bytree': 1.0, 'random_state': 0, 'n_jobs': -1,}

xgc_set = {'colsample_bytree': 0.6655392754230048, 'gamma': 4.198875359789924, 'max_depth': 17, 'min_child_weight': 1,  
    'reg_alpha': 57, 'reg_lambda': 0.896332305739873,}

# =============================================================================

xgbr_t = {'lambda': 7.686003702107571, 'alpha': 8.6004068234192, 'eta': 0.3, 'gamma': 20, 'learning_rate': 0.012, 'colsample_bytree': 0.9, 'colsample_bynode': 0.8, 'n_estimators': 882, 'min_child_weight': 103, 'max_depth': 4, 'subsample': 0.7}

xgbr_set = { 'booster': 'dart', 'learning_rate': 0.1, 'n_estimators': 100, 'tree_method': "hist", "objective": "reg:squarederror", "verbosity": 2,}

xgb_3070 = {'learning_rate': 0.003170080749254201, 'max_depth': 32, 'subsample': 0.2957816844532192, 'colsample_bytree': 0.6594664699872866, 
    'min_child_weight': 8, "objective": "reg:squarederror",}

lgbr_t = {'reg_alpha': 0.01816277967596359, 'reg_lambda': 5.297070757607957, 'colsample_bytree': 0.9, 'subsample': 0.5, 'learning_rate': 0.006, 'max_depth': 10, 'num_leaves': 245, 'min_child_samples': 55, 'min_data_per_groups': 43}

lbr_set = { 'n_estimators': 150, 'objective': 'regression', 'min_child_samples': 7, 'subsample': 1,'num_leaves': 35, 
    'colsample_bytree': 1, 'random_state': 0, 'n_jobs': -1, 'learning_rate': 0.01, 'verbose': 1,}

lgb_3070 = { 'learning_rate': 0.00297158669016989, 'num_leaves': 32, 'subsample': 0.5457131060645429, 'colsample_bytree': 0.6206074333400939, 
    'min_data_in_leaf': 31,  'verbosity': -1, }

catr_t = {'learning_rate': 0.004384466366976951, 'depth': 11, 'subsample': 0.30959203889098214, 'colsample_bylevel': 0.8258819366074175, 'min_data_in_leaf': 32}

cbr_set = { "iterations": 800, "learning_rate": 0.01, "depth": 7,}

cat_3070 = { 'learning_rate': 0.012872913108877197, 'depth': 5, 'subsample': 0.9491103714261131, 
            'colsample_bylevel': 0.9771468169920741, 'min_data_in_leaf': 21,}

xgbrf_t = {'learning_rate': 0.09992558454567729, 'max_depth': 4, 'subsample': 0.6295085012732937, 'colsample_bytree': 0.507405257238443, 'min_child_weight': 12,}

xgbrf_set = {"colsample_bynode": 0.8, "learning_rate": 0.1, "max_depth": 5, "num_parallel_tree": 100,
    "objective": "reg:squarederror", "subsample": 0.8, "tree_method": "hist", }

