import numpy as np
import pandas as pd
import tensorflow as tf
import random
from keras.layers import Input, LSTM, Concatenate, Reshape, Flatten, Dense, LeakyReLU, Dropout, MultiHeadAttention
from tensorflow.keras.layers import   BatchNormalization, Layer,  Attention, Bidirectional, TimeDistributed, Conv1D, Conv2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import MeanSquaredError, BinaryCrossentropy, BinaryAccuracy, AUC  
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#from xgboost import XGBRegressor
#from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ml_model.data_func import sequence_and_normalize

tf.config.set_visible_devices([], 'GPU')


def eval_results(history_in, model_in, X_test_in, y_test_in, scalers_in, timesteps_in, num_features_in):
    
    # Generate predictions
    predictions = model_in.predict(X_test_in)
    ups = 0
    dwns = 0
    zeros = 0
    total = 0

    colors = []
    for i in range(len(predictions)):
        #print(f"Predicted: {predictions[i][0]} Actual: {y_test_in[i]}")
        
        total += 1
        
        if ((predictions[i][0] < 0 and y_test_in[i] > 0) or (predictions[i][0] > 0 and y_test_in[i] < 0)):
            
            if abs(y_test_in[i]) > 1.0:
                colors.append('red')
                dwns += 1
            else:        
                colors.append('red')
                dwns += 1
        
        elif ((predictions[i][0] > 0 and y_test_in[i] > 0) or (predictions[i][0] < 0 and y_test_in[i] < 0)):
            
            if abs(y_test_in[i]) > 1.0:
                colors.append('green')
                ups += 1
            else:        
                colors.append('green')   
                ups += 1
                           
        else:
            colors.append('white') 
            zeros += 1                
            
    print(f" ups: {ups}  dwns: {dwns}  Zeros: {zeros}  Total: {total}  Perf {round(((ups+zeros)/total),4)}  PerfX {round(ups/(ups+dwns),4)}")        


def build_model(input_dim,  lay1, lay2, lay3, sequence_length):
    
    init = RandomNormal(stddev=0.02)
    dropout_rate=0.5
    
    input_layer = Input(shape=(sequence_length, input_dim))
    print("Input Shape:", input_layer.shape)

    reshaped_input = Reshape((sequence_length, input_dim, 1))(input_layer)
    conv1 = TimeDistributed(Conv1D(filters=32, kernel_size=7, activation='relu', padding='same'))(reshaped_input)
    conv1 = Flatten()(conv1)  
    conv1 = Reshape((sequence_length, -1))(conv1)

    x = LSTM(lay1, return_sequences=True, kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.l2(0.01))(conv1)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
       
    x = MultiHeadAttention(num_heads=2, key_dim=15,kernel_regularizer=tf.keras.regularizers.l2(0.01))(x, x)
    x = Dropout(dropout_rate)(x)
   
    #x = LSTM(lay2, return_sequences=True, kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Bidirectional(LSTM(lay2,return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)))(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Attention()([x, x])    
    
    x = LSTM(lay3, return_sequences=False, kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
        
    #x = Dense(1, activation='tanh')(x)
    x = Dense(1)(x)
    return Model(input_layer, x)

def build_modelX(input_dim,  lay1, lay2, lay3, sequence_length):
   
    init = RandomNormal(stddev=0.02)
   
    dropout_rate=0.5
    
    input_layer = Input(shape=(sequence_length, input_dim))
    x = LSTM(lay1, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_layer)
    x = Dropout(dropout_rate)(x)
    x = LSTM(lay2, return_sequences=True)(x)
    x = Dropout(0.2)(x) 
    x = LSTM(lay3, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.2)(x) 
    x = Dense(lay3//2, kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    #x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

    return Model(input_layer, x)


# -----------------------------------------------------------------------

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

time_steps = 7
learning_rate=0.0001
beta_1 = 0.5
lay1 = 32
lay2 = 128
lay3 = 32
epocs = 100
batch = 64

feature_dims, X_train, X_test, y_train, y_test, scalers = sequence_and_normalize(file_path, time_steps)

print(X_train.shape)
print(y_train.shape)


t_model = build_model(feature_dims,  lay1, lay2, lay3, time_steps)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)

#c_metrics = ['BinaryAccuracy', 'AUC', 'MeanSquaredError',]
#t_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=c_metrics )

r_metrics = ['MeanSquaredError','BinaryAccuracy', 'AUC']
t_model.compile(optimizer=optimizer, loss='mse',metrics=r_metrics)

t_model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history_out = t_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epocs, batch_size=batch, callbacks=[early_stopping])


eval_results(history_out, t_model, X_test, y_test, scalers, time_steps, feature_dims)

#oos_file = 'data/lucky13_oos.csv'
#load_and_predict_oos(oos_file, model_result, time_steps, feature_dim_out)

