import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, mean_squared_error
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

from ml_model.model_stats import gen_reg_stats
from ml_model.data_func import simple_split_and_scale


def calc_importances_and_baseline(models, X_train, y_train, X_test, y_test, features):

    importance_df = pd.DataFrame(features, columns=['Feature'])
    baseline_data =[]
    
    # Train models, calculate importances, and store results
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_r2 = r2_score(y_test,y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        if "Reg" in name:
            perf, total, mse, rmse, mae = gen_reg_stats(y_test, y_pred)
            baseline_performance = perf
        else:           
            baseline_performance = accuracy_score(y_test, y_pred)
            
        baseline_data.append({'Model': name, 'Baseline': baseline_performance, 'R2': model_r2, 'MSE':mse })
        print(f"Baseline {baseline_performance}")        
        print(f"Calculating Importances ... ")        
        
        perm_importances = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
        perm_means = perm_importances.importances_mean
        perm_stds = perm_importances.importances_std
        
        # Calculate standard feature importance from the model
        if 'Cat' in name:
            standard_importances = model.get_feature_importance()
        else:
            standard_importances = model.feature_importances_
        
        # Store results in DataFrame
        importance_df[f'{name}_PI_Mean'] = perm_means
        importance_df[f'{name}_PI_Std'] = perm_stds
        importance_df[f'{name}_Std_Imp'] = standard_importances

    # Compute composite importance
    importance_df['Comp_PI_Mean'] = importance_df[
        [f'{name}_PI_Mean' for name in models.keys()]
    ].mean(axis=1)

    importance_df['Comp_Std_Imp'] = importance_df[
        [f'{name}_Std_Imp' for name in models.keys()]
    ].mean(axis=1)

    importance_df_sorted = importance_df.sort_values(by='Comp_Std_Imp', ascending=False)
    
    baseline_df = pd.DataFrame(baseline_data)
    print(baseline_df)
    
    return importance_df_sorted, baseline_df
    

def select_features(importance_df_sorted, threshold_v):    

    # Define thresholds for feature selection
    threshold_mean = np.percentile(importance_df_sorted['Comp_PI_Mean'], threshold_v)  # top 25% permutation importance mean
    threshold_standard = np.percentile(importance_df_sorted['Comp_Std_Imp'], threshold_v)  # top 25% standard importance

    # Select features that are consistently important
    important_features = importance_df_sorted[
        (importance_df_sorted['Comp_PI_Mean'] > threshold_mean) & 
        (importance_df_sorted['Comp_Std_Imp'] > threshold_standard)
    ]['Feature'].tolist()
    return important_features
    
def retrain_models(models, important_features, X_train, X_test, y_train, y_test, baseline_df):    
    
    composite_results = []
    for name, model in models.items():
        X_train_selected = X_train[important_features]
        X_test_selected = X_test[important_features]

        model.fit(X_train_selected, y_train)
        
        # Evaluate the new model
        y_pred_selected = model.predict(X_test_selected)
        sel_mse = mean_squared_error(y_test, y_pred_selected)
        sel_model_r2 = r2_score(y_test,y_pred_selected)
        
        if "Reg" in name:
            perf, total, mse, rmse, mae = gen_reg_stats(y_test, y_pred_selected)
            selected_performance = perf
        else:           
            selected_performance = accuracy_score(y_test, y_pred_selected)
        
        baseline_perf = baseline_df[baseline_df['Model'] == name]['Baseline'].values[0]
        base_r2 = baseline_df[baseline_df['Model'] == name]['R2'].values[0]
        base_mse = baseline_df[baseline_df['Model'] == name]['MSE'].values[0]
        
        composite_results.append({'Model': name, 'Perf': baseline_perf, 'Sel_Perf': selected_performance,  'R2': base_r2, 'Sel_R2': sel_model_r2, 'MSE': base_mse, 'Sel_MSE': sel_mse})
        #print(f'{name} Selected Features Performance: {selected_performance:.4f}')

    # Convert results to DataFrame and plot performance comparison
    performance_df = pd.DataFrame(composite_results) 
    return performance_df   




def gen_results(models, X_train, y_train, X_test, y_test, columns, threshold):
    
    importance_df_sorted, baseline_df = calc_importances_and_baseline(models, X_train, y_train, X_test, y_test, columns)
    important_features = select_features(importance_df_sorted, threshold)    
    performance_df = retrain_models(models, important_features, X_train, X_test, y_train, y_test, baseline_df)   
   
    return importance_df_sorted, important_features, performance_df 
    
    
# --------------
def run():

    datafile = [ 
            'data/NewModel_3070_oos.csv',   
            'data/NewModel_3070.csv',  #1
            'data/NewModel_ALL_oos.csv',   
            'data/NewModel_ALL.csv',  #3
    ]   

    data = pd.read_csv(datafile[1])
    df = data.drop(columns=['TimeTicks','SeqClose'])
    #df = df[((df['RSI'] > 0) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 100)]  
    
    #big list
    big_list = ['STOK5133', 'HourOfDay', 'SDBB291C', 'ATR51', 'TV4', 'ZC911X', 'ZH79X', 'CRSI14', 'ZC79X', 'COMP2', 'ROC91', 
     'SDLR310', 'TV6', 'COMP0', 'SDKC9C', 'ATR2', 'SDKC29C', 'SDBB91C', 'RSI9', 'COMP1', 'RSI91', 'STOK714Y1', 
     'COMP3', 'ROC7', 'ZL79X', 'SDBB29C', 'TV1', 'ROC14', 'TV3', 'ATR5', 'ZL57X', 'RSI141', 'SDBB91', 'ROC71', 
     'STOK714Y', 'ZH57X', 'TV2', 'ROC141', 'ROC9', 'RSI14', 'STOK51331', 'TV5', 'CATR3']
    #X = df[big_list]

    f_list = ['SDBB91', 'COMP2', 'COMP3', 'ATR5', 'TV3', 'HourOfDay', 'TV1', 'ZH79X', 'SDKC29C', 
                    'ZL57X', 'COMP0', 'ATR2', 'TV6', 'RSI14', 'RSI9', 'ATR51']
    X = df[f_list]


    span3 = ['RSI', 'ATR5', 'STOK1', 'H1', 'SDBB9', 'ATR2', 'SDBB91', 'TV11', 'ROC141', 
             'ATR21', 'TV31', 'RSI14X', 'SDKC9', 'ROC1', 'SDLR310', 'ROC14', 'HourOfDay', 'TV41', 'ROC', 'SDKC91']
    #X = df[span3]

    #X = df
    #X = X.drop(columns=['output', 'outputC'])
    
    y = df['outputC'].values
    y2 = df['output'].values
    
    threshold = 50


    #X_train_c, X_test_c, y_train_c, y_test_c = simple_split_and_scale(X, y, 0.2, 42)
    #X_train_r, X_test_r, y_train_r, y_test_r = simple_split_and_scale(X, y2, 0.2, 42)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y2, test_size=0.2, random_state=42)
    
    models_c = {
        #'RandomForestClassifier' :RandomForestClassifier(random_state=42, verbose=2, n_jobs=-1),
        'LGBCls': LGBMClassifier(random_state=42, verbose=2, n_jobs=-1),
        'XGBCls': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=2),
        'CatCls': CatBoostClassifier(random_state=42, verbose=2)
    }

    models_r = {
        #'RandomForestRegressor' :RandomForestRegressor(random_state=42, verbose=2, n_jobs=-1),
        'LGBReg': LGBMRegressor(random_state=42, verbose=2),
        'XGBReg': XGBRegressor(random_state=42, use_label_encoder=False, verbosity=2),
        'CatReg': CatBoostRegressor(random_state=42, verbose=2)
    }

    use_class = True
    use_reg = True
    if use_class:
        importance_df_sorted_c, important_features_c, performance_df_c  = gen_results(models_c, X_train_c, y_train_c, X_test_c, y_test_c, X.columns, threshold)    

    if use_reg:
        importance_df_sorted_r, important_features_r, performance_df_r  = gen_results(models_r, X_train_r, y_train_r, X_test_r, y_test_r, X.columns, threshold)            

    if use_class:
        print("Classifier Sorted Importance")
        print(" ")
        print(importance_df_sorted_c)
        print(" ")
        print(f"Features Selected based on threshold of {threshold} percent")
        print("Classifier Selected Features")
        print(important_features_c)
        print("Perf Results")
        print(performance_df_c)
        print(" ")

        
    if use_reg:
        print("------------------------------------------------------------")
        print(" ")
        print("Regressor Sorted Importance")
        print(" ")
        print(importance_df_sorted_r)
        print(" ")
        print(f"Features Selected based on threshold of {threshold} percent")
        print("Regressor Selected Features")
        print(important_features_r)
        print("Perf Results")
        print(performance_df_r)
        print(" ")



if __name__ == "__main__":
    run()

    #'data/NewModel_ALL.csv',  #3
    
    
    """
    Features Selected based on threshold of 50 percent
    Classifier Selected Features
    ['TV1', 'ZL79X', 'COMP0', 'ZL57X', 'HourOfDay', 'ATR2', 'SDLR310', 'CATR3', 'TV3', 'ROC91', 'ROC71', 'ZC79X', 'SDBB29C', 'SDBB91', 'CRSI14', 'SDKC29C', 'ZH79X', 'SDKC9C', 'TV6']
    Perf Results
        Model      Perf  Sel_Perf        R2    Sel_R2       MSE   Sel_MSE
    0  LGBCls  0.526376  0.525564 -0.899122 -0.902378  0.473624  0.474436
    1  XGBCls  0.521174  0.518104 -0.919979 -0.932290  0.478826  0.481896
    2  CatCls  0.524980  0.525107 -0.904718 -0.904209  0.475020  0.474893

    Regressor Selected Features
    ['ZH57X', 'ZL79X', 'TV3', 'ZH79X', 'ATR2', 'ZC79X', 'COMP3', 'RSI141', 'ROC14', 'ROC141', 'ROC9', 'ATR51', 'STOK51331', 'ZC911X', 'ATR5', 'SDBB291C', 'SDBB91C', 'ROC71']
    Perf Results
        Model    Perf  Sel_Perf        R2    Sel_R2      MSE    Sel_MSE
    0  LGBReg  0.4727    0.4729 -0.008631 -0.013306  11.0985  11.149955
    1  XGBReg  0.4685    0.4706 -0.058814 -0.041507  11.6507  11.460267
    2  CatReg  0.4707    0.4717 -0.027692 -0.025023  11.3083  11.278880
    
    Features Selected based on threshold of 50 percent
    Classifier Selected Features
    ['RSI9', 'TV6', 'TV1', 'COMP0', 'RSI14', 'ATR2', 'HourOfDay', 'ZC79X', 'CATR2', 'ZH911X', 'ZH57X', 'ZC57X', 'ZL911X', 'Day', 'CRSI3', 'ZH79X', 'RSI141', 'STOK5133', 'ROC14', 'ROC141', 'SDLR310', 'SDKC9C', 'ROC7']
    Perf Results
        Model      Perf  Sel_Perf        R2    Sel_R2       MSE   Sel_MSE
    0  LGBCls  0.781103  0.782177  0.124166  0.128462  0.218897  0.217823
    1  XGBCls  0.771977  0.769695  0.087651  0.078522  0.228023  0.230305
    2  CatCls  0.780298  0.782177  0.120944  0.128462  0.219702  0.217823
        
    Features Selected based on threshold of 50 percent
    Regressor Selected Features
    ['RSI9', 'ATR2', 'RSI14', 'COMP2', 'ZL57X', 'ATR5', 'ZH57X', 'ATR21', 'COMP0', 'STOK51331', 'TV5', 'TV6', 'COMP3', 'SDKC29C', 'ATR14', 'SDLR310V', 'ROC141', 'SDKC9', 'TV2', 'SDKC291C', 'ROC14', 'SDBB9', 'ATR51']
    Perf Results
        Model    Perf  Sel_Perf        R2    Sel_R2      MSE    Sel_MSE
    0  LGBReg  0.7556    0.7556  0.375378  0.394523  10.7818  10.451287
    1  XGBReg  0.7512    0.7579  0.313055  0.375975  11.8575  10.771447
    2  CatReg  0.7517    0.7543  0.360360  0.393261  11.0410  10.473082
    
    -------------------------------------------------------------------
    
    
    Features Selected based on threshold of 70 percent
    Regressor Selected Features
    ['RSI9', 'ATR2', 'RSI14', 'ZL57X', 'ATR5', 'ZH57X', 'TV5', 'TV6', 'COMP3', 'SDKC29C', 'ATR14']
    Perf Results
        Model    Perf  Sel_Perf        R2    Sel_R2      MSE    Sel_MSE
    0  LGBReg  0.7556    0.7564  0.375378  0.411651  10.7818  10.155651
    1  XGBReg  0.7512    0.7527  0.313055  0.362738  11.8575  10.999935
    2  CatReg  0.7517    0.7533  0.360360  0.383957  11.0410  10.633673
    
    Classifier Selected Features
    ['RSI9', 'TV6', 'TV1', 'COMP0', 'RSI14', 'ATR2', 'CATR2', 'ZH911X', 'ZC57X', 'ZL911X', 'Day', 'CRSI3']
    Perf Results
        Model      Perf  Sel_Perf        R2    Sel_R2       MSE   Sel_MSE
    0  LGBCls  0.781103  0.781237  0.124166  0.124703  0.218897  0.218763
    1  XGBCls  0.771977  0.774527  0.087651  0.097854  0.228023  0.225473
    2  CatCls  0.780298  0.782982  0.120944  0.131684  0.219702  0.217018
    
    
    
    Features Selected based on threshold of 70 percent
    Regressor Selected Features
    ['ZH57X', 'ATR2', 'ROC14', 'ROC9', 'ATR51']
    Perf Results
        Model    Perf  Sel_Perf        R2    Sel_R2      MSE    Sel_MSE
    0  LGBReg  0.4727    0.4724 -0.008631 -0.017648  11.0985  11.197734
    1  XGBReg  0.4685    0.4685 -0.058814 -0.045961  11.6507  11.509273
    2  CatReg  0.4707    0.4690 -0.027692 -0.028171  11.3083  11.313523
    
    ['TV1', 'COMP0', 'HourOfDay', 'ROC91', 'ROC71', 'ZC79X']
    Perf Results
        Model      Perf  Sel_Perf        R2    Sel_R2       MSE   Sel_MSE
    0  LGBCls  0.526376  0.525056 -0.899122 -0.904413  0.473624  0.474944
    1  XGBCls  0.521174  0.515592 -0.919979 -0.942362  0.478826  0.484408
    2  CatCls  0.524980  0.520185 -0.904718 -0.923947  0.475020  0.479815
    

    
    Features Selected based on threshold of 50 percent
    Classifier Selected Features
    ['TV1', 'ZL79X', 'COMP0', 'ZL57X', 'HourOfDay', 'ATR2', 'SDLR310', 'CATR3', 'TV3', 'ROC91', 'ROC71', 'ZC79X', 'SDBB29C', 'SDBB91', 'CRSI14', 'SDKC29C', 'ZH79X', 'SDKC9C', 'TV6']
    Perf Results
    Model      Perf  Sel_Perf        R2    Sel_R2       MSE   Sel_MSE
    0  LGBCls  0.526376  0.525564 -0.899122 -0.902378  0.473624  0.474436
    1  XGBCls  0.521174  0.518104 -0.919979 -0.932290  0.478826  0.481896
    2  CatCls  0.524980  0.525107 -0.904718 -0.904209  0.475020  0.474893


    Features Selected based on threshold of 50 percent
    Regressor Selected Features
    ['ZH57X', 'ZL79X', 'TV3', 'ZH79X', 'ATR2', 'ZC79X', 'COMP3', 'RSI141', 'ROC14', 'ROC141', 'ROC9', 'ATR51', 'STOK51331', 'ZC911X', 'ATR5', 'SDBB291C', 'SDBB91C', 'ROC71']
    Perf Results
        Model    Perf  Sel_Perf        R2    Sel_R2      MSE    Sel_MSE
    0  LGBReg  0.4727    0.4729 -0.008631 -0.013306  11.0985  11.149955
    1  XGBReg  0.4685    0.4706 -0.058814 -0.041507  11.6507  11.460267
    2  CatReg  0.4707    0.4717 -0.027692 -0.025023  11.3083  11.278880
    
    
    
    Features Selected based on threshold of 50 percent
    Classifier Selected Features
    ['RSI9', 'ATR2', 'RSI14', 'TV6', 'TV1', 'ZL57X', 'COMP0', 'HourOfDay', 'SDBB91', 'ZH79X']
    Perf Results
        Model      Perf  Sel_Perf        R2    Sel_R2       MSE   Sel_MSE
    0  LGBCls  0.775466  0.777345  0.101613  0.109131  0.224534  0.222655
    1  XGBCls  0.769964  0.775064  0.079596  0.100002  0.230036  0.224936
    2  CatCls  0.776406  0.779090  0.105372  0.116111  0.223594  0.220910
        
    
    Features Selected based on threshold of 50 percent
    Regressor Selected Features
    ['RSI9', 'ATR2', 'ATR5', 'ATR51', 'RSI14', 'TV3', 'TV6', 'COMP2', 'SDKC29C', 'COMP3']
    Perf Results
        Model    Perf  Sel_Perf        R2    Sel_R2      MSE    Sel_MSE
    0  LGBReg  0.7536    0.7549  0.379641  0.407515  10.7082  10.227037
    1  XGBReg  0.7517    0.7539  0.273167  0.374111  12.5460  10.803638
    2  CatReg  0.7537    0.7569  0.343594  0.380305  11.3304  10.696709
    
    
    """