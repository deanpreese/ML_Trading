import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from models.kan_mixer_model import KANMixerModel

tf.config.set_visible_devices([], 'GPU')

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



model = KANMixerModel(epochs=100, batch_size=32)

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
    