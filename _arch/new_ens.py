import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, accuracy_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import StackingRegressor, StackingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from sklearn.model_selection import train_test_split

from ml_model.data_func import simple_split_and_scale
from ml_model.model_tracking import train_regressor_model
from ml_model.model_stats import gen_reg_stats, gen_class_stats
import ml_model.model_params as mp


datafile = [ 
        'data/Lucky13_3070_oos.csv',   
        'data/Lucky13_3070.csv',  #1
        'data/ndata_diff_lucky13_3070_oos.csv', 
        'data/ndata_diff_lucky13_3070.csv', #3
        'data/ndata_lucky_13_lag_3070_oos.csv', 
        'data/ndata_lucky13_lag_3070.csv', #5
        'new_model_Z_lucky13_3070_oos.csv',
        'new_model_Z_lucky13_3070.csv' #7,
        'data/Lucky13_3070_oos_3.csv',   
        'data/Lucky13_3070_3.csv',  #8
        'data/Lucky13_3070_oos_5.csv',   
        'data/Lucky13_3070_5.csv',  #10

]

dtx = pd.read_csv(datafile[1])

X = dtx
X = X.drop(columns=['output', 'outputC'])
y = dtx['output'].values
yC = dtx['outputC'].values

fl_out = list(X.columns)
X_train, X_test, y_train, y_test, yC_train, yC_test = train_test_split(
    X, y, yC, test_size=0.2, random_state=42)


# Define base learners
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
lgb_model = lgb.LGBMRegressor(objective='regression', n_estimators=100)
cb_model = cb.CatBoostRegressor(loss_function='RMSE', iterations=100, verbose=0)

xgb_clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100)
lgb_clf = lgb.LGBMClassifier(objective='binary', n_estimators=100)
cb_clf = cb.CatBoostClassifier(loss_function='Logloss', iterations=100, verbose=0)

stack_reg = StackingRegressor(
    estimators=[
        ('xgbr', xgb_model),
        ('lgbr', lgb_model),
        (' cbr', cb_model)
    ],
    final_estimator=Lasso(alpha=0.1),
    cv=KFold(n_splits=5, shuffle=True, random_state=42)
)

stack_clf = StackingClassifier(
    estimators=[
        ('xgbc', xgb_clf),
        ('lgbc', lgb_clf),
        (' cbc', cb_clf)
    ],
    final_estimator=LogisticRegression(),
    cv=KFold(n_splits=5, shuffle=True, random_state=42)
)


# Train the model
stack_reg.fit(X_train, y_train)
y_pred = stack_reg.predict(X_test)
r2 = round(r2_score(y_test, y_pred),4)
score = round(stack_reg.score(X_test, y_pred),4)
perf, tot, mse, rmse, mae = gen_reg_stats(y_test, y_pred)    

kcv = KFold(n_splits=5, shuffle=True, random_state=42)
k_pred = cross_val_predict(stack_reg, X_test, y_test, cv=kcv)
kr2 = round(r2_score(y_test, k_pred),4)
kscore = round(stack_reg.score(X_test, k_pred),4)
kperf, ktot, kmse, krmse, kmae = gen_reg_stats(y_test, k_pred)    


stack_clf.fit(X_train, yC_train)
clf_y_pred = stack_clf.predict(X_test)
clf_r2 = round(r2_score(yC_test, y_pred),4)
clf_score = round(stack_reg.score(X_test, clf_y_pred),4)
clf_perf, clf_total, clf_tn, clf_fp, clf_fn, clf_tp = gen_class_stats(yC_test, clf_y_pred)    

kclf_pred = cross_val_predict(stack_clf, X_test, yC_test, cv=KFold(n_splits=5, shuffle=True, random_state=42))
kclf_r2 =round(r2_score(yC_test, kclf_pred),4)
kclf_score = round(stack_reg.score(X_test, kclf_pred),4)
kclf_perf, kclf_total, kclf_tn, kclf_fp, kclf_fn, kclf_tp = gen_class_stats(yC_test, kclf_pred)    


print(" ")
for m in stack_reg.named_estimators_:
        r_pred = stack_reg.named_estimators_[m].predict(X_test)
        rr2 = f"{round(r2_score(y_test, r_pred),4):.4f}"
        rscore = f"{round(stack_reg.score(X_test, r_pred),4):.4f}"
        rperf, rtotal, rmse, rrmse, rmae = gen_reg_stats(y_test, r_pred)        
        print(f"{m}  R2 {rr2}     Score {rscore}   Perf  {rperf}   MAE {rmae} MSE {rmse}   RMSE {rrmse}" )


for m in stack_clf.named_estimators_:
        c_pred = stack_clf.named_estimators_[m].predict(X_test)
        xr2 = f"{round(r2_score(yC_test, c_pred),4):.4f}"
        xscore = f"{round(stack_clf.score(X_test, c_pred),4):.4f}"
        xclf_perf, xclf_total, xclf_tn, xclf_fp, xclf_fn, xclf_tp = gen_class_stats(yC_test, c_pred)  
        print(f"{m}  R2 {xr2}     Score {xscore}   Perf  {xclf_perf}   TN {xclf_tn}  TP {xclf_tp}  FN {xclf_fn} FP {xclf_fp}" )


# Combine predictions
combined_preds = np.column_stack((y_pred, k_pred))

# Train final model
final_model = xgb.XGBRFRegressor()
final_model.fit(combined_preds, y_pred)

# Evaluate final model
final_predictions = final_model.predict(combined_preds)
mse = mean_squared_error(y, final_predictions)
print(f'Mean Squared Error of Final Model: {mse}')


print(" ")
print(f"Total Predicts {tot}")
print(f"Stacked  Reg:   Perf  {perf}   MAE {mae}   MSE {mse}   RMSE {rmse} " )
print(f"KFOLD Reg       Perf  {kperf}   MAE {kmae}   MSE  {kmse}   RMSE {krmse}" )
print(f"Stacked  Class  Perf  {clf_perf}   TN {clf_tn}   TP {clf_tp}   FN {clf_fn}   FP {clf_fp} " )
print(f"KFOLD Class     Perf  {kclf_perf}   TN {kclf_tn}   TP {kclf_tp}   FN {kclf_fn}   FP {kclf_fp} " )
print(" ")

print(" ")
print(f'Mean Squared Error of Final Model: {mse}')
print(" ")
