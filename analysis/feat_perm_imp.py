import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier


def calc_importances_and_baseline(models, X_train, y_train, X_test, y_test, features):

    importance_df = pd.DataFrame(features, columns=['Feature'])
    baseline_data =[]
    
    # Train models, calculate importances, and store results
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate baseline performance
        y_pred = model.predict(X_test)
        baseline_performance = accuracy_score(y_test, y_pred)
        baseline_data.append({'Model': name, 'Baseline Performance': baseline_performance})
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
        selected_performance = accuracy_score(y_test, y_pred_selected)
        baseline_perf = baseline_df[baseline_df['Model'] == name]['Baseline Performance'].values[0]
        composite_results.append({'Model': name, 'Baseline Performance': baseline_perf, 'Selected Features Performance': selected_performance})
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

    
# --------------
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
    
    threshold = 75

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'RandomForestClassifier' :RandomForestClassifier(random_state=42, verbose=2),
        'LightGBM': LGBMClassifier(random_state=42, verbose=2),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=2),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=2)
    }

    models_r = {
        'RandomForest' :RandomForestRegressor(random_state=42, verbose=2),
        'LightGBM': LGBMRegressor(random_state=42, verbose=2),
        'XGBoost': XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=2),
        'CatBoost': CatBoostRegressor(random_state=42, verbose=2)
    }


    importance_df_sorted, baseline_df = calc_importances_and_baseline(models, X_train, y_train, X_test, y_test, X.columns)
    #plot_importances(models, importance_df_sorted)
    important_features = select_features(importance_df_sorted, threshold)    
    performance_df = retrain_models(models, important_features, X_train, X_test, y_train, y_test, baseline_df)   
    #plot_new_results(performance_df)
    
    print(" ")
    print(f"Features Selected based on threshold of {threshold}")
    print(important_features)
    print("New Results")
    print(performance_df)

if __name__ == "__main__":
    run()




