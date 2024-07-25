from time import time
import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

# =============================================================================

def run():

    datafile = [ 
        '../data/buildSeqInd_Lucky13_5M_3070.csv',   #0
        '../data/buildSeqInd_Lucky13_5M_ALL.csv',  #1
        '../data/buildSeqInd_Lucky13_F.csv',  #2
        '../data/buildSeqInd_Lucky13_D.csv',  #3
        '../data/buildSeqInd_Lucky13_F_3070.csv',  #4
        '../data/Expanded_Lucky13_3070.csv',  #5
        '../data/alt_ex13.csv', #6
        
        
    ]

    file_loaded = pd.read_csv(datafile[6])
    X = file_loaded
    X = X.drop(columns=['output', 'outputC'])
    y = file_loaded['outputC'].values
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, verbose=2)
    #rf = LGBMClassifier(n_jobs=-1,verbose=2)
        
    tic_fwd = time()        
    sfs_forward = SequentialFeatureSelector(
        rf, n_features_to_select=9, direction="forward", n_jobs=-1
    ).fit(X, y)
    toc_fwd = time()

    tic_bwd = time()
    sfs_backward = SequentialFeatureSelector(
        rf, n_features_to_select=9, direction="backward", n_jobs=-1
    ).fit(X, y)
    toc_bwd = time()

        
  
    print(
        "Features selected by forward sequential selection: "
        f"{feature_names[sfs_forward.get_support()]}"
    )
    print(f"Done in {toc_fwd - tic_fwd:.3f}s")
    print(
        "Features selected by backward sequential selection: "
        f"{feature_names[sfs_backward.get_support()]}"
    )
    print(f"Done in {toc_bwd - tic_bwd:.3f}s")        
        

if __name__ == "__main__":
    run()






