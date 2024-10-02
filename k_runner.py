import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib 
import matplotlib.pyplot as plt
import pywt

from ml_model.model_stats import gen_reg_stats_x, gen_class_stats 
from ml_model.data_func import split_three_ways

tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)


def evaluate_model( y_test, y_pred):
    
    correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(y_test, y_pred)
    print(" ")
    print(f"Pred MSE: {mse},  MAE: {mae}, R2: {r2}")
    print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.4f}")
    print(f"Number of Samples: {total}")        
    print(" ")        
    
    return rmse, mse, mae, r2
    
        
        
def plot_training_history(history):
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()
        
        
        
def run():

    datafile = [ 
        'data/Lucky13_3070_oos.csv',   
        'data/Lucky13_3070.csv',  #1
        'data/Lucky13_EX_3070_oos.csv',  
        'data/Lucky13_EX_3070.csv',  #3
        'data/Lucky13_ALL_oos.csv',  #4
        'data/Lucky13_ALL.csv',  #5
    ]

    run_oos = False
    run_perf = False

    model_to_load = ""    

    if run_oos:
        file_path = datafile[0]
        df = pd.read_csv(file_path)
        df = df.drop(columns=['outputC'])
        X = df.drop(columns=['output']).values
        y = df['output'].values 

        oos_model = tf.keras.models.load_model(model_to_load)
        y_pred = oos_model.predict(X)
        evaluate_model( y, y_pred)

    if run_perf:
        file_path = datafile[0]
        
        perf_model = tf.keras.models.load_model(model_to_load)
        
        df = pd.read_csv(file_path)
        df = df.drop(columns=['outputC'])
        X = df.drop(columns=['output']).values
        y = df['output'].values 

        yn = False
        count = 0
        ycount = 0

        y_pred = []

        for i in range(len(y)):
            x_val = X[i]
            x_val = x_val.reshape((1, 14, 1)) 
            y_val = perf_model.model.predict(x_val)
            
            y_pred.append(y_val[0][0])
            print(y_val[0][0])

        evaluate_model(y, y_pred)


if __name__ == "__main__":
    run()            
        