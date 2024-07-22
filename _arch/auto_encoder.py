
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

from ml_model.data_func import simple_split_and_scale

# Load data
dtx = pd.read_csv('data/buildSeqInd_Lucky13_F_3070.csv')
X = dtx.drop(columns=['output', 'outputC'])
y = dtx['output'].values

# Split and scale data
X_train, X_test, y_train, y_test = simple_split_and_scale(X, y, 0.7, 42)

# Define the autoencoder model
input_dim = X_train.shape[1]
encoding_dim = 8  # Dimension of the encoded representation (reduces from 13 to 8)

input_layer = Input(shape=(input_dim,))
x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_layer)
#x = BatchNormalization()(x)
x = Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = LeakyReLU(alpha=0.1)(x)
x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = Dropout(0.2)(x)

encoded = Dense(encoding_dim, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = BatchNormalization()(encoded)
x = LeakyReLU(alpha=0.1)(x)
x = Dropout(0.2)(x)

decoded = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = BatchNormalization()(decoded)
x = LeakyReLU(alpha=0.1)(x)
x = Dropout(0.2)(x)

decoded = Dense(input_dim, activation='sigmoid')(x)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
autoencoder.summary()

# Add EarlyStopping and ReduceLROnPlateau callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Train the autoencoder
autoencoder.fit(X_train, X_train,
                epochs=100,
                batch_size=32,
                shuffle=True,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr])

# Generate encoded inputs
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# Train an XGBoost regressor using the encoded inputs
xgb = XGBRegressor(random_state=42)
xgb.fit(X_train_encoded, y_train)

# Test the XGBoost regressor
y_pred = xgb.predict(X_test_encoded)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Data: {mse}')
