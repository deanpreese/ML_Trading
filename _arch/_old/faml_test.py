from flaml import AutoML
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from flaml.automl.ml import sklearn_metric_loss_score

from models.wrapped_models import TunableXGBRegressor

from xgboost import XGBRegressor

datafile = [ 
        'data/buildSeqInd_13X_5M_ALL.csv',   #0
        'data/buildSeqIndX_5M_ALL.csv',   #1
        'data/buildSeqInd_Lucky13_5M_3070.csv',   #2
        'data/markov.csv',  #3
        'data/buildSeqInd_Lucky13_5M_ALL.csv',  #4
        'data/buildSeqInd_13X_5M_3070.csv',  #5
        'data/buildSeqIndX_5M_3070.csv',   #6
        'data/Fractal_ALL_5M.csv', #7
    ]


data = pd.read_csv(datafile[4])
run_test_size = 0.8



automl = AutoML()
settings = {
    "time_budget": 60,  # total running time in seconds
    "metric": 'r2',  # primary metrics for regression can be chosen from: ['mae','mse','r2']
    "estimator_list": ['xgboost'],  # list of ML learners; we tune XGBoost in this example
    "task": 'regression',  # task type
    "log_file_name": 'faml_experiment.log',  # flaml log file
    "seed": 7654321,  # random seed
}


num_columns = len(data.axes[1]) 
input_features =  num_columns -1
X = data.iloc[:, 0:input_features]  
y = data['output'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=run_test_size, random_state=0)

automl.fit(X_train=X_train, y_train=y_train, **settings)

print('Best hyperparmeter config:', automl.best_config)
print('Best r2 on validation data: {0:.4g}'.format(1-automl.best_loss))
print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))
print(automl.model.estimator)


plt.barh(automl.feature_names_in_, automl.feature_importances_)

print("  ")

y_pred = automl.predict(X_test)
print('Predicted labels', y_pred)


print('r2', '=', 1 - sklearn_metric_loss_score('r2', y_pred, y_test))
print('mse', '=', sklearn_metric_loss_score('mse', y_pred, y_test))
print('mae', '=', sklearn_metric_loss_score('mae', y_pred, y_test))

print("  ")

xgb = TunableXGBRegressor()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

print('default xgboost r2', '=', 1 - sklearn_metric_loss_score('r2', y_pred, y_test))

print("  ")