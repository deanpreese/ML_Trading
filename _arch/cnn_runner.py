
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout, MaxPooling1D, LSTM, Attention, Bidirectional, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ml_model.model_stats import gen_reg_stats_x 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.config.set_visible_devices([], 'GPU')
from models.cnn_lstm_model import CNN_LSTM


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
model = CNN_LSTM()

train = True
test = False
single_item = False

if train:
    file_path = datafile[1]
    history_out, y_pred = model.train_model(file_path)
    model.evaluate_model(y_pred)

if test:
    file_path = datafile[0]
    model.load_saved_model("train")
    model.run_batch_test(file_path)

if single_item:
    file_path = datafile[0]
    model.load_saved_model("run")

    df = pd.read_csv(file_path)
    df = df.drop(columns=['outputC'])
    X = df.drop(columns=['output']).values
    y = df['output'].values 

    model.X_test = X
    model.y_test = y

    yn = False
    count = 0
    ycount = 0

    y_pred = []

    for i in range(len(y)):
        x_val = X[i]
        x_val = x_val.reshape((1, 14, 1)) 
        y_val = model.model.predict(x_val)
        
        y_pred.append(y_val[0][0])
        print(y_val[0][0])

    model.evaluate_model(y_pred)


