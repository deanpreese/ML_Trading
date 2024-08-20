
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout, MaxPooling1D, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ml_model.model_stats import gen_reg_stats_x 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_cnn_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=2, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=128, kernel_size=2, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)  # Output layer with 1 neuron for regression
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.summary()
    return model


def build_cnn_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=2, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=64, kernel_size=2, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)    
    x = LSTM(50, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(50, return_sequences=True)(x)
    x = Dropout(0.2)(x)    
    x = LSTM(50)(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)  # Output layer with 1 neuron for regression
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.summary()
    return model

datafile = [ 
    'data/Lucky13_3070_oos.csv',   
    'data/Lucky13_3070.csv',  #1
    'data/ndata_diff_lucky13_3070_oos.csv', 
    'data/ndata_diff_lucky13_3070.csv', #3
    'data/ndata_lucky_13_lag_3070_oos.csv', 
    'data/ndata_lucky13_lag_3070.csv', #5
    'new_model_Z_lucky13_3070_oos.csv',
    'new_model_Z_lucky13_3070.csv', #7,
    'data/Lucky13_3070_oos_3.csv',   
    'data/Lucky13_3070_3.csv',  #8
    'data/Lucky13_3070_oos_5.csv',   
    'data/Lucky13_3070_5.csv',  #10
]

file_path = datafile[1]
df = pd.read_csv(file_path)
df = df.drop(columns=['outputC'])
X = df.drop(columns=['output']).values
y = df['output'].values

"""
timesteps = 3
Xd, yd = [], []
for i in range(timesteps, len(X)):
    Xd.append(X[i-timesteps:i])
    yd.append(y[i])
X, y = np.array(Xd), np.array(yd)
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#input_shape=(X_train.shape[1], X_train.shape[2])
input_shape=(X_train.shape[1], 1)

#model = build_cnn_model(input_shape)
model = build_cnn_lstm_model(input_shape)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history_out = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32, callbacks=[early_stopping])
y_pred = model.predict(X_test)
 
correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(y_test, y_pred)
print(f"Test MSE: {mse}, Test MAE: {mae}, R2: {r2}")
print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.2f}%")
print(f"Number of Samples: {total}")
    

