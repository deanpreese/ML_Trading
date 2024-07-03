import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from common_func import show_stats, calc_MSE

from RFTModel import RFTModel
from XtraModel import XtraModel
from GBRModel import GBRModel
from MLPRModel import MLPRModel
from LTSMModel import LTSMModel
from MLPRTFModel import MLPRTFModel
from CBRModel import CBRModel
from XGBModel import XGBModel


#data = pd.read_csv("data/Seq_buildSeq30_R3070_TAG.csv")   
#data = pd.read_csv("data/Seq_buildSeqInd_R3070_TAG.csv")
#data = pd.read_csv("data/Seq_CL_buildSeqInd_R3070_TAG.csv")
#data = pd.read_csv("data/Seq_GC_buildSeqInd_R3070_TAG.csv")
#data = pd.read_csv("data/Seq_buildSeqInd_R3070_TAG_X.csv")
data = pd.read_csv("data/Seq_buildSeqInd_R3070_DIFF_X.csv")
#data = pd.read_csv("data/Seq_buildSeqInd_R2080_TAG.csv")


num_columns = len(data.axes[1]) 
input_features =  num_columns -1
X = data.iloc[:, 0:input_features]  
y = data['output'].values
run_test_size = 0.8
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=run_test_size, random_state=0)

import xgboost as xgb

xgb_e = xgb.XGBRegressor().fit(X_train,y_train)

xbg_params = {
    "subsample":[0.5, 0.75, 1],
    "colsample_bytree":[0.5, 0.75, 1],
    "max_depth":[2, 6, 12],
    "min_child_weight":[1,5,15],
    "learning_rate":[0.3, 0.1, 0.03],
    "n_estimators":[100]
    }


gv_out = RandomizedSearchCV(estimator=xgb_e, param_distributions=xbg_params,
                              n_iter = 100, scoring='neg_mean_absolute_error', 
                              cv = 3, verbose=2, random_state=0, n_jobs=4,
                              return_train_score=True)

# Fit the random search model
gv_out.fit(X_train, y_train)
print("   ")
print(f" {gv_out.best_params_} " )
print("   ")
predicted_values = gv_out.predict(X_test)
# ShowImportances(data.columns[:self.input_features],self.model.feature_importances_, False)

print("   ")
show_stats(True, y_test, predicted_values)
print("   ")
print("   ")


