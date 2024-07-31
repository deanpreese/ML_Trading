import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import plot_partial_dependence

def main():
    # Sample data
    X = pd.DataFrame(np.random.randn(1000, 10), columns=[f'feature_{i}' for i in range(10)])
    y_cont = np.random.randn(1000)
    y_bin = np.random.randint(0, 2, size=1000)

    # Train-test split
    X_train, X_test, y_cont_train, y_cont_test, y_bin_train, y_bin_test = train_test_split(X, y_cont, y_bin, test_size=0.2, random_state=42)

    # Base models
    regressor_xgb = XGBRegressor()
    classifier_xgb = XGBClassifier()
    regressor_rf = RandomForestRegressor()
    classifier_rf = RandomForestClassifier()
    regressor_svr = SVR()

    # Train models
    regressor_xgb.fit(X_train, y_cont_train)
    classifier_xgb.fit(X_train, y_bin_train)
    regressor_rf.fit(X_train, y_cont_train)
    classifier_rf.fit(X_train, y_bin_train)
    regressor_svr.fit(X_train, y_cont_train)

    # Predictions
    y_cont_pred_xgb = regressor_xgb.predict(X_test)
    y_bin_pred_xgb = classifier_xgb.predict(X_test)
    y_cont_pred_rf = regressor_rf.predict(X_test)
    y_bin_pred_rf = classifier_rf.predict(X_test)
    y_cont_pred_svr = regressor_svr.predict(X_test)

    # Ensemble predictions (example: weighted average)
    ensemble_pred_cont = (y_cont_pred_xgb + y_cont_pred_rf + y_cont_pred_svr) / 3
    ensemble_pred_bin = (y_bin_pred_xgb + y_bin_pred_rf) / 2

    # Evaluation
    mse_xgb = mean_squared_error(y_cont_test, y_cont_pred_xgb)
    accuracy_xgb = accuracy_score(y_bin_test, y_bin_pred_xgb)
    mse_rf = mean_squared_error(y_cont_test, y_cont_pred_rf)
    accuracy_rf = accuracy_score(y_bin_test, y_bin_pred_rf)
    mse_svr = mean_squared_error(y_cont_test, y_cont_pred_svr)

    print(f'Mean Squared Error (XGBoost Regressor): {mse_xgb}')
    print(f'Accuracy (XGBoost Classifier): {accuracy_xgb}')
    print(f'Mean Squared Error (Random Forest Regressor): {mse_rf}')
    print(f'Accuracy (Random Forest Classifier): {accuracy_rf}')
    print(f'Mean Squared Error (SVR): {mse_svr}')

    # Plotting distribution of predictions
    plt.figure(figsize=(12, 6))
    sns.histplot(y_cont_pred_xgb, kde=True, color='blue', label='XGB Regressor')
    sns.histplot(y_bin_pred_xgb, kde=True, color='green', label='XGB Classifier')
    sns.histplot(y_cont_pred_rf, kde=True, color='purple', label='RF Regressor')
    sns.histplot(y_bin_pred_rf, kde=True, color='orange', label='RF Classifier')
    sns.histplot(y_cont_pred_svr, kde=True, color='cyan', label='SVR')
    sns.histplot(ensemble_pred_cont, kde=True, color='red', label='Ensemble Regressor')
    plt.legend()
    plt.title('Distribution of Predictions')
    plt.show()

    # Plot overlapping regions
    plt.figure(figsize=(8, 6))
    plt.scatter(y_cont_pred_xgb, y_bin_pred_xgb, alpha=0.5, label='XGB')
    plt.scatter(y_cont_pred_rf, y_bin_pred_rf, alpha=0.5, label='RF')
    plt.scatter(y_cont_pred_svr, y_bin_pred_xgb, alpha=0.5, label='SVR')
    plt.title('Overlap between Model Predictions')
    plt.xlabel('Regressor Predictions')
    plt.ylabel('Classifier Predictions')
    plt.legend()
    plt.show()

    # Outlier analysis
    outliers = np.abs(y_cont_pred_xgb - y_cont_test) > 2 * np.std(y_cont_pred_xgb - y_cont_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[outliers].iloc[:, 0], y_cont_test[outliers], color='red', label='Outliers')
    plt.scatter(X_test[~outliers].iloc[:, 0], y_cont_test[~outliers], color='blue', alpha=0.5, label='Non-Outliers')
    plt.legend()
    plt.title('Outlier Analysis')
    plt.show()

    # Partial Dependence
    features = [0, 1, 2]  # Example feature indices
    plot_partial_dependence(regressor_xgb, X_test, features, grid_resolution=50)
    plt.suptitle('Partial Dependence - XGBoost Regressor')
    plt.show()

    plot_partial_dependence(classifier_xgb, X_test, features, grid_resolution=50)
    plt.suptitle('Partial Dependence - XGBoost Classifier')
    plt.show()

    plot_partial_dependence(regressor_rf, X_test, features, grid_resolution=50)
    plt.suptitle('Partial Dependence - Random Forest Regressor')
    plt.show()

    plot_partial_dependence(classifier_rf, X_test, features, grid_resolution=50)
    plt.suptitle('Partial Dependence - Random Forest Classifier')
    plt.show()

    plot_partial_dependence(regressor_svr, X_test, features, grid_resolution=50)
    plt.suptitle('Partial Dependence - SVR')
    plt.show()

if __name__ == "__main__":
    main()
