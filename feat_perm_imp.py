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
        mse = mean_squared_error(y_test, y_pred)
        
        if "Regressor" in name:
            perf, total, mse, rmse, mae = gen_reg_stats(y_test, y_pred)
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
        sel_mse = mean_squared_error(y_test, y_pred_selected)
        sel_model_r2 = r2_score(y_test,y_pred_selected)
        
        if "Regressor" in name:
            perf, total, mse, rmse, mae = gen_reg_stats(y_test, y_pred_selected)
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
            'data/ndata_lucky13_lag_3070_oos.csv', 
            'data/ndata_lucky13_lag_3070.csv', #5
            'data/new_model_Z_lucky13_3070_oos.csv',   
            'data/new_model_Z_lucky13_3070.csv',  #7
            'data/new_model_HLC_lucky13.csv', #8
    ]

    file_loaded = pd.read_csv(datafile[8])
    X = file_loaded
    X = X.drop(columns=['output', 'outputC'])
    y = file_loaded['outputC'].values
    y2 = file_loaded['output'].values
    
    threshold = 75

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


    df1_html = importance_df_sorted_r.to_html()
    df2_html = importance_df_sorted_c.to_html()
    df3_html = important_features_r
    df4_html = performance_df_r.to_html()
    df5_html = important_features_c
    df6_html = performance_df_c.to_html()
        
    # Convert each DataFrame to HTML with titles and spacing
    html_string = """
    <html>
    <head><title>Permutations DataFrames</title></head>
    <body>
    <h1>Report of Multiple DataFrames</h1>

    <h2>importance_df_sorted_r</h2>
    {df1_html}
    <br><br> 
    <h2>importance_df_sorted_c</h2>
    {df2_html}
    <br><br>
    <h2>important_features_r</h2>
    {df3_html}
    <br><br> 
    <h2>performance_df_r</h2>
    {df4_html}
    <br><br> 
    <h2>important_features_c/h2>
    {df5_html}
    <br><br> 
    <h2>performance_df_c</h2>
    {df6_html}
    <br><br>

    </body>
    </html>
    """.format(
            df1_html=df1_html
           , df2_html=df2_html
           , df3_html=df3_html
           , df4_html=df4_html
           , df5_html=df5_html
            , df6_html=df6_html
           )

    # Save the final HTML string to a file
    with open('multiple_dataframes.html', 'w') as f:
        f.write(html_string)

if __name__ == "__main__":
    run()


"""
'data/Lucky13_3070.csv',  #1
Features Selected based on threshold of 75 percent

Regressor Selected Features
['RSI', 'ATR2']
Perf Results
                   Model  Base Perf  Sel Feat Perf   Base R2    Sel R2   Base MSE    Sel MSE
0  RandomForestRegressor     0.7571         0.7467  0.331625  0.257686  11.536986  12.813274
1      LightGBMRegressor     0.7575         0.7571  0.403233  0.433011  10.300953   9.786943
2       XGBoostRegressor     0.7532         0.7563  0.356036  0.394698  11.115619  10.448273
3      CatBoostRegressor     0.7569         0.7571  0.374997  0.428228  10.788345   9.869507

Classifier Selected Features
['RSI', 'ATR2']
Perf Results
                    Model  Base Perf  Sel Feat Perf   Base R2    Sel R2  Base MSE   Sel MSE
0  RandomForestClassifier   0.783251       0.754127  0.132758  0.016231  0.216749  0.245873
1                LightGBM   0.780835       0.779627  0.123092  0.118259  0.219165  0.220373
2                 XGBoost   0.770635       0.775198  0.082281  0.100539  0.229365  0.224802
3                CatBoost   0.780969       0.779090  0.123629  0.116111  0.219031  0.220910 
 
 
 'data/ndata_diff_lucky13_3070.csv', #3
Features Selected based on threshold of 75 percent

Regressor Selected Features
['RSI9X', 'ROC14X', 'RSI14X', 'STOK714X', 'RSI7X', 'ATR2X', 'ATR3X']
Perf Results
                   Model  Base Perf  Sel Feat Perf   Base R2    Sel R2   Base MSE    Sel MSE
0  RandomForestRegressor     0.7567         0.7541  0.338584  0.304073  11.416871  12.012573
1      LightGBMRegressor     0.7564         0.7567  0.411518  0.409335  10.157930  10.195626
2       XGBoostRegressor     0.7517         0.7516  0.366225  0.333527  10.939751  11.504164
3      CatBoostRegressor     0.7540         0.7552  0.394704  0.414240  10.448164  10.110949

Classifier Selected Features
['RSI9X', 'ROC14X', 'RSI14X', 'ATR2X', 'RSI3X', 'RSI7X', 'ATR21']
Perf Results
                    Model  Base Perf  Sel Feat Perf   Base R2    Sel R2  Base MSE   Sel MSE
0  RandomForestClassifier   0.780969       0.773587  0.123629  0.094095  0.219031  0.226413
1                LightGBM   0.778419       0.780701  0.113426  0.122555  0.221581  0.219299
2                 XGBoost   0.771440       0.772648  0.085503  0.090336  0.228560  0.227352
3                CatBoost   0.782848       0.778419  0.131147  0.113426  0.217152  0.221581    
    
data/new_model_Z_lucky13_3070.csv',  #7
Features Selected based on threshold of 75 percent

Regressor Selected Features
['RSI', 'ATR5', 'RSI14Z', 'ATR3Z', 'STOK7143Z', 'ATR2', 'L01Z', 'L02Z']
Perf Results
                   Model  Base Perf  Sel Feat Perf   Base R2    Sel R2   Base MSE    Sel MSE
0  RandomForestRegressor     0.7572         0.7565  0.322231  0.312034  11.699143  11.875157
1      LightGBMRegressor     0.7541         0.7533  0.399271  0.402950  10.369336  10.305835
2       XGBoostRegressor     0.7512         0.7553  0.330910  0.375647  11.549342  10.777115
3      CatBoostRegressor     0.7553         0.7557  0.352401  0.422722  11.178367   9.964535

Classifier Selected Features
['RSI', 'RSI14Z', 'ATR2', 'ATR21', 'ATR54', 'ATR53']
Perf Results
                    Model  Base Perf  Sel Feat Perf   Base R2    Sel R2  Base MSE   Sel MSE
0  RandomForestClassifier   0.777882       0.778285  0.111279  0.112890  0.222118  0.221715
1                LightGBM   0.781372       0.783116  0.125240  0.132221  0.218628  0.216884
2                 XGBoost   0.767280       0.775064  0.068856  0.100002  0.232720  0.224936
3                CatBoost   0.775064       0.782177  0.100002  0.128462  0.224936  0.217823    



'data/ndata_lucky13_lag_3070.csv', #5
Features Selected based on threshold of 75 percent

Regressor Selected Features
['RSI9Y', 'STOK714Y', 'ATR3Y', 'RSI14Y', 'ZL9Y', 'ATR5Y', 'RSI7Y', 'ROC91Y', 'ZL21Y', 'ATR2Y', 'RSI5Y', 'ZH21Y', 'SDKC7Y']
Perf Results
                   Model  Base Perf  Sel Feat Perf   Base R2    Sel R2  Base MSE    Sel MSE
0  RandomForestRegressor     0.7591         0.7567  0.367963  0.326789  10.90975  11.620462
1      LightGBMRegressor     0.7547         0.7551  0.398790  0.414898  10.37764  10.099598
2       XGBoostRegressor     0.7540         0.7555  0.331799  0.359155  11.53399  11.061791
3      CatBoostRegressor     0.7563         0.7582  0.368733  0.395915  10.89646  10.427259

Classifier Selected Features
['RSI9Y', 'RSI14Y', 'RSI7Y', 'STOK7Y', 'ZL9Y', 'VOLMA7Y', 'ZH91Y', 'STOK5Y']
Perf Results
                    Model  Base Perf  Sel Feat Perf   Base R2    Sel R2  Base MSE   Sel MSE
0  RandomForestClassifier   0.778151       0.774661  0.112353  0.098391  0.221849  0.225339
1                LightGBM   0.781237       0.776137  0.124703  0.104298  0.218763  0.223863
2                 XGBoost   0.767145       0.769964  0.068319  0.079596  0.232855  0.230036
3                CatBoost   0.780566       0.778016  0.122018  0.111816  0.219434  0.221984
    
    
    """