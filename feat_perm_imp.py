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


def calc_importances_and_baseline(models, X_train, y_train, X_test, y_test, features):

    importance_df = pd.DataFrame(features, columns=['Feature'])
    baseline_data =[]
    
    # Train models, calculate importances, and store results
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_r2 = r2_score(y_test,y_pred)
        mse = mean_squared_error(y_test, y_pred, squared=True)
        
        if "Regressor" in name:
            perf, total = gen_reg_stats(y_test, y_pred)
            baseline_performance = perf
        else:           
            baseline_performance = accuracy_score(y_test, y_pred)
            
        baseline_data.append({'Model': name, 'Baseline Performance': baseline_performance, 'R2': model_r2, 'MSE':mse })
        print(f"Baseline {baseline_performance}")        
        print(f"Calculating Importances ... ")        
        
        perm_importances = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
        perm_means = perm_importances.importances_mean
        perm_stds = perm_importances.importances_std
        
        # Calculate standard feature importance from the model
        if name == 'CatBoost':
            standard_importances = model.get_feature_importance()
        else:
            standard_importances = model.feature_importances_
        
        # Store results in DataFrame
        importance_df[f'{name} Permutation Importance Mean'] = perm_means
        importance_df[f'{name} Permutation Importance Std'] = perm_stds
        importance_df[f'{name} Standard Importance'] = standard_importances

    # Compute composite importance
    importance_df['Composite Permutation Importance Mean'] = importance_df[
        [f'{name} Permutation Importance Mean' for name in models.keys()]
    ].mean(axis=1)

    importance_df['Composite Standard Importance'] = importance_df[
        [f'{name} Standard Importance' for name in models.keys()]
    ].mean(axis=1)

    importance_df_sorted = importance_df.sort_values(by='Composite Permutation Importance Mean', ascending=False)
    
    baseline_df = pd.DataFrame(baseline_data)
    print(baseline_df)
    
    return importance_df_sorted, baseline_df
    
    
def plot_importances(models, importance_df_sorted):
    
    # Plot the importances
    fig, ax = plt.subplots(2, 2, figsize=(20, 16), sharey=True)
    for i, (name, _) in enumerate(models.items()):
        ax[i // 2, i % 2].barh(importance_df_sorted['Feature'], importance_df_sorted[f'{name} Permutation Importance Mean'], 
                            xerr=importance_df_sorted[f'{name} Permutation Importance Std'], color='skyblue')
        ax[i // 2, i % 2].set_xlabel('Permutation Importance')
        ax[i // 2, i % 2].set_title(f'{name} Permutation Importance')

    # Composite Importance plot
    ax[1, 1].barh(importance_df_sorted['Feature'], importance_df_sorted['Composite Permutation Importance Mean'], color='lightgreen')
    ax[1, 1].set_xlabel('Composite Permutation Importance')
    ax[1, 1].set_title('Composite Permutation Importance')
    plt.tight_layout()
    plt.show()    
    

def select_features(importance_df_sorted, threshold_v):    

    # Define thresholds for feature selection
    threshold_mean = np.percentile(importance_df_sorted['Composite Permutation Importance Mean'], threshold_v)  # top 25% permutation importance mean
    threshold_standard = np.percentile(importance_df_sorted['Composite Standard Importance'], threshold_v)  # top 25% standard importance

    # Select features that are consistently important
    important_features = importance_df_sorted[
        (importance_df_sorted['Composite Permutation Importance Mean'] > threshold_mean) & 
        (importance_df_sorted['Composite Standard Importance'] > threshold_standard)
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
        sel_mse = mean_squared_error(y_test, y_pred_selected, squared=True)
        sel_model_r2 = r2_score(y_test,y_pred_selected)
        
        if "Regressor" in name:
            perf, total = gen_reg_stats(y_test, y_pred_selected)
            selected_performance = perf
        else:           
            selected_performance = accuracy_score(y_test, y_pred_selected)
        
        baseline_perf = baseline_df[baseline_df['Model'] == name]['Baseline Performance'].values[0]
        base_r2 = baseline_df[baseline_df['Model'] == name]['R2'].values[0]
        base_mse = baseline_df[baseline_df['Model'] == name]['MSE'].values[0]
        
        composite_results.append({'Model': name, 'Base Perf': baseline_perf, 'Sel Feat Perf': selected_performance,  'Base R2': base_r2, 'Sel R2': sel_model_r2, 'Base MSE': base_mse, 'Sel MSE': sel_mse})
        #print(f'{name} Selected Features Performance: {selected_performance:.4f}')

    # Convert results to DataFrame and plot performance comparison
    performance_df = pd.DataFrame(composite_results) 
    return performance_df   
    
def plot_new_results(performance_df):
    # Set the figure size
    plt.figure(figsize=(10, 6))
    
    # Set the bar width
    bar_width = 0.35
    
    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(performance_df))
    r2 = [x + bar_width for x in r1]
    
    # Create the bars for baseline performance
    plt.bar(r1, performance_df['Baseline Performance'], color='skyblue', width=bar_width, edgecolor='grey', label='Baseline Performance')
    
    # Create the bars for selected features performance
    plt.bar(r2, performance_df['Selected Features Performance'], color='lightgreen', width=bar_width, edgecolor='grey', label='Selected Features Performance')
    
    # Add labels to the x-axis
    plt.xlabel('Models', fontweight='bold')
    plt.xticks([r + bar_width/2 for r in range(len(performance_df))], performance_df['Model'])
    
    # Add the labels, title, and legend
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.legend()
    
    # Show the plot
    plt.tight_layout()
    plt.show()


def gen_results(models, X_train, y_train, X_test, y_test, columns, threshold):
    
    importance_df_sorted, baseline_df = calc_importances_and_baseline(models, X_train, y_train, X_test, y_test, columns)
    #plot_importances(models, importance_df_sorted)
    important_features = select_features(importance_df_sorted, threshold)    
    performance_df = retrain_models(models, important_features, X_train, X_test, y_train, y_test, baseline_df)   
    #plot_new_results(performance_df)
    
    return importance_df_sorted, important_features, performance_df 
    
    
# --------------
def run():

    datafile = [ 
            'data/Lucky13_3070_oos.csv',   
            'data/Lucky13_3070.csv',  #1
            'data/ndata_diff_lucky13_3070_oos.csv', 
            'data/ndata_diff_lucky13_3070.csv', #3
            'data/ndata_lag_3070_oos.csv', 
            'data/ndata_lag_3070.csv', #5
    ]


    file_loaded = pd.read_csv(datafile[1])
    X = file_loaded
    X = X.drop(columns=['output', 'outputC'])
    y = file_loaded['outputC'].values
    y2 = file_loaded['output'].values
    
    threshold = 80

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y2, test_size=0.2, random_state=42)
    
    models_c = {
        'RandomForestClassifier' :RandomForestClassifier(random_state=42, verbose=2, n_jobs=-1),
        'LightGBM': LGBMClassifier(random_state=42, verbose=2, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=2),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=2)
    }

    models_r = {
        'RandomForestRegressor' :RandomForestRegressor(random_state=42, verbose=2, n_jobs=-1),
        'LightGBMRegressor': LGBMRegressor(random_state=42, verbose=2, n_jobs=-1),
        'XGBoostRegressor': XGBRegressor(random_state=42, use_label_encoder=False, verbosity=2),
        'CatBoostRegressor': CatBoostRegressor(random_state=42, verbose=2)
    }

    importance_df_sorted_r, important_features_r, performance_df_r  = gen_results(models_r, X_train_r, y_train_r, X_test_r, y_test_r, X.columns, threshold)
    importance_df_sorted_c, important_features_c, performance_df_c  = gen_results(models_c, X_train_c, y_train_c, X_test_c, y_test_c, X.columns, threshold)
       
    print("------------------------------------------------------------")
    print(" ")
    print("Regressor Sorted Importance")
    print(" ")
    print(importance_df_sorted_r)
    print(" ")
    print("Classifier Sorted Importance")
    print(" ")
    print(importance_df_sorted_c)
    print(" ")
     
    print(" ")
    print(f"Features Selected based on threshold of {threshold} percent")
    print(" ")
    print("Regressor Selected Features")
    print(important_features_r)
    print("Perf Results")
    print(performance_df_r)
    print(" ")
    print("Classifier Selected Features")
    print(important_features_c)
    print("Perf Results")
    print(performance_df_c)
    print(" ")


if __name__ == "__main__":
    run()

"""
'data/ndata_3070.csv', #6
Features Selected based on threshold of 80 percent

Regressor Selected Features
['RSI', 'RSI14', 'RSI7', 'ZH21', 'ATR5', 'ATR3', 'ZL21', 'ROC142']
Perf Results
                   Model  Base Perf  Sel Feat Perf   Base R2    Sel R2   Base MSE    Sel MSE
0  RandomForestRegressor     0.7575         0.7577  0.298154  0.267068  12.841684  13.410465
1      LightGBMRegressor     0.7580         0.7579  0.354981  0.354198  11.801921  11.816252
2       XGBoostRegressor     0.7558         0.7581  0.198856  0.279380  14.658546  13.185196
3      CatBoostRegressor     0.7585         0.7593  0.315338  0.329355  12.527262  12.270795

Classifier Selected Features
['RSI', 'RSI14', 'ATR2', 'RSI5', 'RSI3']
Perf Results
                    Model  Base Perf  Sel Feat Perf   Base R2    Sel R2  Base MSE   Sel MSE
0  RandomForestClassifier   0.777062       0.758107  0.108143  0.032311  0.222938  0.241893
1                LightGBM   0.777062       0.776466  0.108143  0.105758  0.222938  0.223534
2                 XGBoost   0.771221       0.769790  0.084773  0.079050  0.228779  0.230210
3                CatBoost   0.773724       0.772771  0.094789  0.090973  0.226276  0.227229



'data/ndata_3070_alt.csv', #7
Features Selected based on threshold of 80 percent

Regressor Selected Features
['STOK714', 'RSI9X', 'RSI7', 'ZH21', 'ROC14', 'RSI14', 'ZL9', 'ZH9', 'ATR5', 'ATR2X', 'ATR7', 'ATR3X']
Perf Results
                   Model  Base Perf  Sel Feat Perf   Base R2    Sel R2   Base MSE    Sel MSE
0  RandomForestRegressor     0.7678         0.7648  0.311004  0.282914  12.032978  12.523554
1      LightGBMRegressor     0.7663         0.7666  0.373942  0.380523  10.933788  10.818863
2       XGBoostRegressor     0.7622         0.7639  0.277903  0.277994  12.611076  12.609479
3      CatBoostRegressor     0.7685         0.7635  0.357480  0.368677  11.221289  11.025749

Classifier Selected Features
['RSI9X', 'ZH21', 'ATR2X', 'STOKX721']
Perf Results
                    Model  Base Perf  Sel Feat Perf   Base R2    Sel R2  Base MSE   Sel MSE
0  RandomForestClassifier   0.779485       0.772099  0.117826  0.088277  0.220515  0.227901
1                LightGBM   0.782225       0.783655  0.128788  0.134507  0.217775  0.216345
2                 XGBoost   0.770550       0.780558  0.082082  0.122115  0.229450  0.219442
3                CatBoost   0.783059       0.780438  0.132124  0.121639  0.216941  0.219562



'data/buildSeqInd_Lucky13_5M_3070.csv',   #0
Regressor Selected Features
['RSI']
Perf Results
                   Model  Base Perf  Sel Feat Perf   Base R2    Sel R2  Base MSE    Sel MSE
0  RandomForestRegressor     0.7541         0.7260  0.437924 -0.006855  9.593983  17.185827
1      LightGBMRegressor     0.7584         0.7581  0.467815  0.337973  9.083780  11.300027
2       XGBoostRegressor     0.7561         0.7559  0.424432  0.284433  9.824265  12.213895
3      CatBoostRegressor     0.7550         0.7577  0.447159  0.333775  9.436339  11.371682

Classifier Selected Features
['RSI', 'ATR2']
Perf Results
                    Model  Base Perf  Sel Feat Perf   Base R2    Sel R2  Base MSE   Sel MSE
0  RandomForestClassifier   0.777000       0.761286  0.107925  0.045063  0.223000  0.238714
1                LightGBM   0.780857       0.779429  0.123355  0.117641  0.219143  0.220571
2                 XGBoost   0.777000       0.777714  0.107925  0.110783  0.223000  0.222286
3                CatBoost   0.778429       0.776714  0.113640  0.106782  0.221571  0.223286


'data/ym_ndata_3070_alt.csv', #8
Features Selected based on threshold of 80 percent

Regressor Selected Features
['RSI9X', 'STOK714', 'RSI7', 'ZC21', 'RSI14', 'ATR3X', 'ATR2X', 'RSI3', 'ZC9']
Perf Results
                   Model  Base Perf  Sel Feat Perf   Base R2    Sel R2    Base MSE     Sel MSE
0  RandomForestRegressor     0.7746         0.7709  0.422498  0.405263  673.230376  693.321742
1      LightGBMRegressor     0.7794         0.7761  0.406958  0.404063  691.346006  694.720407
2       XGBoostRegressor     0.7745         0.7761  0.368080  0.434784  736.668210  658.906857
3      CatBoostRegressor     0.7776         0.7775  0.412339  0.446774  685.072842  644.930163

Classifier Selected Features
['RSI9X', 'ATR21', 'STOK7211', 'ATR2X', 'STOK7212']
Perf Results
                    Model  Base Perf  Sel Feat Perf   Base R2    Sel R2  Base MSE   Sel MSE
0  RandomForestClassifier   0.787966       0.780265  0.151841  0.121034  0.212034  0.219735
1                LightGBM   0.788327       0.789049  0.153286  0.156174  0.211673  0.210951
2                 XGBoost   0.780866       0.782792  0.123441  0.131143  0.219134  0.217208
3                CatBoost   0.789771       0.786402  0.159062  0.145584  0.210229  0.213598


"""