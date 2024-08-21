import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # For saving scaler and IsolationForest

from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor
from lightgbm  import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from ml_model.model_stats import gen_reg_stats_x 
from models.ensemble_models import Anomaly_Ensemble 

tf.config.set_visible_devices([], 'GPU')

   
def main():
    datafile = [ 
        'data/Lucky13_3070_oos.csv',   
        'data/Lucky13_3070.csv',  #1
        'data/ndata_diff_lucky13_3070_oos.csv', 
        'data/ndata_diff_lucky13_3070.csv', #3
        'data/ndata_lucky_13_lag_3070_oos.csv', 
        'data/ndata_lucky13_lag_3070.csv', #5
        'new_model_Z_lucky13_3070_oos.csv',
        'new_model_Z_lucky13_3070.csv', #7,
        'data/Lucky13_3070_oos_3.csv',   
        'data/Lucky13_3070_3.csv',  #8
        'data/Lucky13_3070_oos_5.csv',   
        'data/Lucky13_3070_5.csv',  #10
    ]

    filepath = datafile[1]
    ae = Anomaly_Ensemble(epochs=75, batch_size=32)
    
    train = False
    test = False
    single_item = True
    
    if train:
        #Batch Training 
        ae.training_setup(filepath)
        ae.build_ensemble_models()
        ae.train_ensemble()
        ae.train_baseline()
        ae.anomaly_baseline()

    if test:
        # Batch Testing
        ae.test_setup(filepath)
        ae.load_saved_ensemble()
        ae.anomaly_baseline()    

    if single_item:
        ae.load_saved_ensemble()
        data = pd.read_csv(filepath)                   
        X = data
        X = X.drop(columns=['output', 'outputC'])
        y = data['output'].values
                
        ae.X_test = X.values
        ae.y_test = y
        
        y_pred = []
        
        yn = False
        count = 0
        ycount = 0
        for i in range(len(y)):

            yn = ae.detect_anomalies_single(X.iloc[i])
            count += 1
            
            if yn == True:
                ycount += 1
                print(yn)
                
        print(f"Count {count}  {ycount} ")    
    
    
if __name__ == "__main__":
    main()