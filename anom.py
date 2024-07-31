import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load your data
data = pd.read_csv('your_data.csv')

# Define your feature columns and target column
features = ['SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR34', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1']
target = 'output'

X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for LSTM (samples, time steps, features)
X_train_lstm = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

# Build LSTM Autoencoder
input_dim = X_train_lstm.shape[2]
timesteps = X_train_lstm.shape[1]

inputs = layers.Input(shape=(timesteps, input_dim))
encoded = layers.LSTM(64, activation='relu', return_sequences=True)(inputs)
encoded = layers.LSTM(32, activation='relu')(encoded)
decoded = layers.RepeatVector(timesteps)(encoded)
decoded = layers.LSTM(32, activation='relu', return_sequences=True)(decoded)
decoded = layers.LSTM(64, activation='relu', return_sequences=True)(decoded)
decoded = layers.TimeDistributed(layers.Dense(input_dim))(decoded)

autoencoder = models.Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the LSTM Autoencoder
autoencoder.fit(X_train_lstm, X_train_lstm, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

# Get reconstruction error
X_test_pred = autoencoder.predict(X_test_lstm)
mse = np.mean(np.power(X_test_lstm - X_test_pred, 2), axis=1).flatten()

# Filter out anomalies
autoencoder_threshold = np.percentile(mse, 90)
autoencoder_filtered_indices = mse < autoencoder_threshold

X_test_filtered = X_test[autoencoder_filtered_indices]
y_test_filtered = y_test[autoencoder_filtered_indices]

# Train XGBoost model
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# Train CatBoost model
cat_model = CatBoostRegressor(silent=True)
cat_model.fit(X_train, y_train)
cat_pred = cat_model.predict(X_test)

# Train LightGBM model
lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)

# Weighted prediction (simple averaging)
weights = [0.3, 0.3, 0.4]  # Custom weights for XGBoost, CatBoost, LightGBM
weighted_pred = weights[0] * xgb_pred + weights[1] * cat_pred + weights[2] * lgb_pred

# Calculate performance
unfiltered_mse = mean_squared_error(y_test, weighted_pred)
filtered_mse = mean_squared_error(y_test_filtered, weighted_pred[autoencoder_filtered_indices])

unfiltered_r2 = r2_score(y_test, weighted_pred)
filtered_r2 = r2_score(y_test_filtered, weighted_pred[autoencoder_filtered_indices])

print("Unfiltered MSE:", unfiltered_mse)
print("Filtered MSE:", filtered_mse)
print("Unfiltered R2:", unfiltered_r2)
print("Filtered R2:", filtered_r2)
