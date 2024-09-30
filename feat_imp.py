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

    importance_df_sorted = importance_df.sort_values(by='Composite Standard Importance', ascending=False)
    
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
        'data/Lucky13_EX_3070_oos.csv',  
        'data/Lucky13_EX_3070.csv',  #3
        'data/new_model_Z_lucky13_3070.csv', #4
        'data/ReFried_5M_ALL.csv' #5
        
    ]

    df = pd.read_csv(datafile[5])
    df = df[((df['RSI'] > 20) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 80)]  
    
    #f_list = ['RSI','ADX1','STOK1','ATR5','ATR51','SDKC9','EMAL21213',
    #            'EMAL10102','ADX2','SDLR93','EMAL10103','EMAL21211','EMAL10101','FOSC','FOSC1','ADX','ATR54','SDLR92','ROC', 'output','outputC'] 
    
    
    #f_list = ['RSI','STOK1','ROC','ATR2','SDLR310','FOSC1','ADX1','SDKC9','EMAL10101','EMAL21211','SDLR93','ATR21','EMAL10103','ADX2','FOSC2','ATR54','ATR5', 'output','outputC']
    #df = df[f_list]
    
    #f_list = ['RSI', 'ATR2', 'ATR5', 'STOK1', 'SDLR310','FOSC1','ADX1','SDKC9','EMAL10101','EMAL21211', 'output', 'outputC']
    #df = df[f_list]
    
    X = df
    X = X.drop(columns=['output', 'outputC'])
    
    y = df['outputC'].values
    y2 = df['output'].values
    
    threshold = 75


    X_train_c, X_test_c, y_train_c, y_test_c = simple_split_and_scale(X, y, 0.2, 42)
    X_train_r, X_test_r, y_train_r, y_test_r = simple_split_and_scale(X, y2, 0.2, 42)
    

    #X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y, test_size=0.2, random_state=42)
    #X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y2, test_size=0.2, random_state=42)
    
    models_c = {
        #'RandomForestClassifier' :RandomForestClassifier(random_state=42, verbose=2, n_jobs=-1),
        'LightGBM': LGBMClassifier(random_state=42, verbose=2, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=2),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=2)
    }

    models_r = {
        #'RandomForestRegressor' :RandomForestRegressor(random_state=42, verbose=2, n_jobs=-1),
        'LightGBMRegressor': LGBMRegressor(random_state=42, verbose=2, n_jobs=-1),
        'XGBoostRegressor': XGBRegressor(random_state=42, use_label_encoder=False, verbosity=2),
        'CatBoostRegressor': CatBoostRegressor(random_state=42, verbose=2)
    }

    use_class = False
    use_reg = True

    if use_class:
        importance_df_sorted_c, important_features_c, performance_df_c  = gen_results(models_c, X_train_c, y_train_c, X_test_c, y_test_c, X.columns, threshold)    
        
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
        importance_df_sorted_r, important_features_r, performance_df_r  = gen_results(models_r, X_train_r, y_train_r, X_test_r, y_test_r, X.columns, threshold)            
        
        
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

