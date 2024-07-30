import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('your_time_series_data.csv')

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Feature and target separation
features = data.drop(['target'], axis=1)
target = data['target']

# Standardize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Train-test split (ensure temporal order is maintained)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

# Part 1: Boosted Tree Model
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
bst = xgb.train(params, dtrain, num_boost_round=100)

# Part 2: Individual Feature Training and Aggregation
individual_models = {}
for i in range(X_train.shape[1]):
    dtrain_feat = xgb.DMatrix(X_train[:, i].reshape(-1, 1), label=y_train)
    model = xgb.train(params, dtrain_feat, num_boost_round=100)
    individual_models[i] = model

aggregated_predictions = np.zeros(X_test.shape[0])
for i, model in individual_models.items():
    dtest_feat = xgb.DMatrix(X_test[:, i].reshape(-1, 1))
    pred = model.predict(dtest_feat)
    aggregated_predictions += pred

aggregated_predictions /= len(individual_models)

# Part 3: Predict Next Feature Values
next_feature_predictions = {}
for i in range(X_train.shape[1]):
    X_feat_train = X_train[:, i].reshape(-1, 1)
    y_next = np.roll(y_train, -1)
    dtrain_feat_next = xgb.DMatrix(X_feat_train[:-1], label=y_next[:-1])
    model = xgb.train(params, dtrain_feat_next, num_boost_round=100)
    dtest_feat_next = xgb.DMatrix(X_feat_train[-1].reshape(1, -1))
    next_feature_predictions[i] = model.predict(dtest_feat_next)[0]

next_features = np.array(list(next_feature_predictions.values())).reshape(1, -1)
dnext = xgb.DMatrix(next_features)
next_output_prediction = bst.predict(dnext)

# Generate Outputs
model_one_output = bst.predict(dtest)
model_three_output = next_output_prediction
combined_output = model_one_output + model_three_output

# Evaluation Metrics
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

rmse_one, mae_one, r2_one = evaluate_model(y_test, model_one_output)
rmse_three, mae_three, r2_three = evaluate_model(y_test, model_three_output)
rmse_combined, mae_combined, r2_combined = evaluate_model(y_test, combined_output)

print(f"Model One - RMSE: {rmse_one}, MAE: {mae_one}, R^2: {r2_one}")
print(f"Model Three - RMSE: {rmse_three}, MAE: {mae_three}, R^2: {r2_three}")
print(f"Combined Model - RMSE: {rmse_combined}, MAE: {mae_combined}, R^2: {r2_combined}")

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(features):
    X_train_cv, X_test_cv = features[train_index], features[test_index]
    y_train_cv, y_test_cv = target[train_index], target[test_index]
    dtrain_cv = xgb.DMatrix(X_train_cv, label=y_train_cv)
    dtest_cv = xgb.DMatrix(X_test_cv, label=y_test_cv)
    model_cv = xgb.train(params, dtrain_cv, num_boost_round=100)
    y_pred_cv = model_cv.predict(dtest_cv)
    rmse_cv, mae_cv, r2_cv = evaluate_model(y_test_cv, y_pred_cv)
    print(f"CV Fold - RMSE: {rmse_cv}, MAE: {mae_cv}, R^2: {r2_cv}")

# Direct Multi-Step Forecasting
horizons = [1, 3, 5]
multi_step_predictions = {}
for horizon in horizons:
    model = xgb.train(params, dtrain, num_boost_round=100)
    multi_step_predictions[horizon] = model.predict(dtest)

# Recursive Forecasting
recursive_predictions = []
current_input = X_test[0].reshape(1, -1)
for _ in range(len(X_test)):
    pred = bst.predict(xgb.DMatrix(current_input))
    recursive_predictions.append(pred[0])
    current_input = np.roll(current_input, -1)
    current_input[0, -1] = pred

# Residual Analysis
residuals = y_test - model_one_output
plt.plot(residuals)
plt.title('Residuals Analysis')
plt.show()

# Sensitivity Analysis
sensitivity = {}
for i in range(X_test.shape[1]):
    perturbed_input = X_test.copy()
    perturbed_input[:, i] += np.std(X_test[:, i])
    perturbed_output = bst.predict(xgb.DMatrix(perturbed_input))
    sensitivity[i] = np.mean(np.abs(perturbed_output - model_one_output))

print("Sensitivity Analysis:", sensitivity)
