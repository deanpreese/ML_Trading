from ipaddress import v4_int_to_packed
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor, StackingRegressor, ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

import xgboost as xgb
import catboost as cb

from common_func import gen_importances, show_stats, calc_MSE

model_filename = "Stacking.ml"
num_epocs = 5
run_batch_size = 16
run_test_size = 0.8

data = pd.read_csv("data/Seq_buildSeqInd_R3070_DIFF_X.csv")

num_columns = len(data.axes[1]) 
input_features =  num_columns -1
X = data.iloc[:, 0:input_features]  

#  GBR set
#idxs =  [47, 52, 61, 62, 67]

# xtra
idxs = [47, 52, 61, 62, 77]

#cbr set
#idxs =[ 9 17 21 22 24 28 32 35 36 37 42 43 44 46 47 48 51 52 53 54 55 56 57 60  61 62 75 78 79]


#Rft Set
#idxs = [47, 49, 50, 51, 52, 56, 57, 60, 61, 62, 67]

#idxs = [47, 49, 50, 51, 52, 56, 57, 60, 61, 62, 67]

#X = X.iloc[:, idxs]
#input_features = len(X.axes[1])

y = data['output'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=run_test_size, random_state=0)

print("  ")
print(f"TRAIN: {X_train.shape[0]}   {X_train.shape[1]}")
print(f"TEST: {X_test.shape[0]}  {X_test.shape[1]} ")
print(" ")


# Scale the features to the range [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

params_xtra = {
    'n_estimators': 100,
    "max_depth": 50,
    "min_samples_split": 5,
    #"learning_rate": 0.01,
    'criterion': 'squared_error',
    #"min_samples_leaf" : 1,
    'random_state' : 0,
    'max_features': 'sqrt', 
    'n_jobs' : 6,   
    'verbose' : 2
 }

params_mlpr = { 'hidden_layer_sizes' :  [100,50],
    #'activation' : 'relu',
    #'solver' : 'adam',
    #'alpha' : 0.0,
    'batch_size' : 32,
    'random_state' : 0,
    #'tol' : 0.000001,
    #'nesterovs_momentum' : False,
    #'learning_rate' : 'invscaling',
    #'learning_rate' : 'constant',
    #'learning_rate_init' : 0.001,
    #'learning_rate_init' : 0.00088,
    'max_iter' : 250,
    #'shuffle' : False,
    'early_stopping' : True,
    'n_iter_no_change' : 25,
    'verbose' : True }

params_gbr = {
    'n_estimators': 700,
    "max_depth": 40,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    'loss': 'squared_error',
    "min_samples_leaf" : 20,
    'random_state' : 0,
    'verbose' : 1,
    'max_features': 'log2',
    'n_iter_no_change' : 25
    }

params_rfa = {
    'n_estimators': 100,
    #"max_depth": 20,
    #"min_samples_split": 10,
    #"learning_rate": 0.01,
    'criterion': 'squared_error',
    #"min_samples_leaf" : 5,
    'random_state' : 0,
    'max_features': 'sqrt', 
    'n_jobs' : 4,      
    'verbose' : 2
    
 }

params = {
    'criterion': 'squared_error',
    'random_state' : 0,
    'n_jobs' : 4,   
    'verbose' : 2
 }

xgbr = xgb.XGBRegressor(
    #learning_rate=0.75,
    n_estimators=200,
    #max_depth=5,
    #subsample=0.9,
    #colsample_bytree=0.8,
    #colsample_bylevel=0.8,
    #gamma=0,
    #min_child_weight=1
)


# Create individual regressors

cbr = cb.CatBoostRegressor(loss_function='RMSE')
KNN = KNeighborsRegressor()
SVR = SVR()
LReg = LinearRegression()
mlp_regressor = MLPRegressor(**params_mlpr)
random_forest_regressor = RandomForestRegressor(**params_rfa)
rfr = RandomForestRegressor(**params)
gradient_boosting_regressor = GradientBoostingRegressor(**params_gbr)
gbr = GradientBoostingRegressor()
extra_trees = ExtraTreesRegressor(**params_xtra)
ext = ExtraTreesRegressor(**params)
abr = AdaBoostRegressor(n_estimators=100, random_state=0)



# Create a Voting Regressor that combines the individual regressors
stacking_regressor = StackingRegressor(estimators=[
    #('KNN', KNN),
    #('ABR', abr),
    #('RFR', rfr),
    #('EXT', ext),
    #('SVR', SVR),
    #('LReg', LReg),    
    #('RandomForest', random_forest_regressor),
    #('GradientBoosting', gradient_boosting_regressor),
    #('MLP', mlp_regressor),
    #('ExtraTrees', extra_trees),
    #('GBR', gbr),
    ('XBGR', xgbr),
    ('CBR', cbr),
    ]
    , cv=2
    , verbose=True
 )



regressor = stacking_regressor

# Train the Voting Regressor on the training data
regressor.fit(X_train, y_train)

print(" ")
#print("Saving and Reloading Model ")
#pickle.dump(voting_regressor, open(model_filename, "wb"))
#loaded_model = pickle.load(open(model_filename, "rb"))
#predictions = loaded_model.predict(X_test)

predictions = regressor.predict(X_test)
temp = pd.DataFrame(y_test)
temp['prediction'] = regressor.predict(X_test)

# The stacked models predictions, which should perform the best
#temp['stacking_prediction'] = stacking_regressor.predict(X_test)
show_stats(True, y_test, predictions)
calc_MSE(y_test, predictions, True)

# Get each model in the stacked model to see how they individually perform
for m in regressor.named_estimators_:
        temp[m] = regressor.named_estimators_[m].predict(X_test)

print("Correlations with target column")
print(temp.corr()["prediction"])

# See what our meta-learner is thinking (the linear regression)
for coef in zip(regressor.named_estimators_, regressor.final_estimator_.coef_):
    print(coef)


