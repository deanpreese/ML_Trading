import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, accuracy_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from ml_model.data_func import simple_split_and_scale

from ml_model.model_tracking import track_regressor_model
from ml_model.model_stats import gen_reg_stats, calc_reg_ens_results
from ml_model.data_func import simple_split_and_scale
from ml_model.model_params import xgr_param_set, lbr_param_set, cbr_param_set, xgr_param_set2, xgb_rf_params

from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor
from lightgbm  import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

model_filename = "Stacking.ml"
num_epocs = 5
run_batch_size = 16
run_test_size = 0.8

dtx = pd.read_csv('data/buildSeqInd_Lucky13_F_3070.csv')
X = dtx
X = X.drop(columns=['output', 'outputC'])
y = dtx['output'].values
fl_out = list(X.columns)
X_train, X_test, y_train, y_test = simple_split_and_scale(X, y, 0.7, 42)

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


est_list_1 = [('cbr', CatBoostRegressor(**cbr)), 
              ('lgb', LGBMRegressor(**lbr)),  
              ('xgb', XGBRegressor(**xgr)), 
              ('xgbrf', XGBRFRegressor(**xg_rf)),
              ('cbr2', CatBoostRegressor()), 
              ('lgb2',LGBMRegressor()),
              ('xgb2', XGBRegressor()), 
              ('xgbrf2',XGBRFRegressor()),
             ]


est_list_2 = [ 
                ('xgb_p',XGBRegressor()),   
                #('xgb',XGBRegressor(**xgb_3070)),  
                ('xgbr',XGBRegressor(**xgr)),  
                #('xgbf',XGBRegressor(**xgb_params_F)), 
                #('xgbm',XGBRegressor(**xgb_params_M)), 
                ('cbr_p',CatBoostRegressor()),  
                ('cb',CatBoostRegressor(**cat_3070)), 
                ('cbr',CatBoostRegressor(**cbr)),  
                ('cbrf',CatBoostRegressor(**cat_params_F)), 
                ('cbrm',CatBoostRegressor(**cat_params_M)),
                ('lgb_p',LGBMRegressor(**lbr)), 
                #('lgb',LGBMRegressor(**lgb_3070)), 
                ('lgbr',LGBMRegressor(**lbr)), 
                ('lgbrf',LGBMRegressor(**lgb_params_F)), 
                #('lgbrm',LGBMRegressor(**lgb_params_M)), 
                ('xgbrfrf',XGBRFRegressor(**xg_rf)),
                ('xgbrf',XGBRFRegressor()),
                ('mlp',MLPRegressor()),
                #('abr', AdaBoostRegressor()),
                ('rfr', RandomForestRegressor())                
          ]

est_list_3 = [ ('xgb',XGBRegressor()),  
              ('xgb2', XGBRegressor(**xgr)),  
              ('xgb3', XGBRegressor(**xgb_params_F)), 
              ('xgb4',XGBRegressor(**xgb_params_M)),                
              ]


# Create a Voting Regressor that combines the individual regressors
stacking_regressor = StackingRegressor(estimators=est_list_2
    , cv=4
    , verbose=True,
    final_estimator=CatBoostRegressor()
 )

regressor = stacking_regressor
#regressor = voting_regressor


# Train the Voting Regressor on the training data
regressor.fit(X_train, y_train)

print(" ")
#print("Saving and Reloading Model ")
#pickle.dump(voting_regressor, open(model_filename, "wb"))
#loaded_model = pickle.load(open(model_filename, "rb"))
#predictions = loaded_model.predict(X_test)

y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred, squared=True)
rmse =mean_squared_error(y_test, y_pred, squared=False)
r2 =r2_score(y_test, y_pred)
score = regressor.score(X_test, y_test)
mae = float(mean_absolute_error(y_test,y_pred))                
perf, tot = gen_reg_stats(y_test, y_pred)        

print("Results ---")
print(f"MSE  {mse}   RMSE {rmse}  R2 {r2}  Score {score}  MAE {mae}  Perf  {perf}  Total {tot}" )

for m in regressor.named_estimators_:
        r_pred = regressor.named_estimators_[m].predict(X_test)

        mse = mean_squared_error(y_test, r_pred, squared=True)
        rmse =mean_squared_error(y_test, r_pred, squared=False)
        r2 =r2_score(y_test, r_pred)
        score = regressor.score(X_test, r_pred)
        mae = float(mean_absolute_error(y_test,r_pred))                
        perf, tot = gen_reg_stats(y_test, r_pred)        

        print(f"{m}  MSE  {mse}   RMSE {rmse}  R2 {r2}  Score {score}  MAE {mae}  Perf  {perf}  Total {tot}" )


