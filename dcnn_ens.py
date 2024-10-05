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
    run_perf = True

    if run_oos:
        
        file_path = datafile[0]
        df = pd.read_csv(file_path)
        df = df.drop(columns=['outputC'])
        X = df.drop(columns=['output']).values
        y = df['output'].values 
        
        
        model_dir = "saved_models/dcnn/"
        files = os.listdir(model_dir)
        model_to_load = files[0]
        model_xxx= os.path.join(model_dir, model_to_load)
        
        oos_model = tf.keras.models.load_model(model_xxx)
        y_pred = oos_model.predict(X, verbose=0)
        evaluate_model( y, y_pred)

    if run_perf:
        
        print("")
        
        file_path = datafile[0]
        model_dir_upper = "saved_models/dcnn_ens/upper"
        model_dir_lower = "saved_models/dcnn_ens/lower"
        lower_models = []
        upper_models = []
        
        
        files_upper = [ f for f in os.listdir(model_dir_upper) if f.endswith('.keras') ]
        for f in files_upper:
            
            fm = os.path.join(model_dir_upper, f)
            print(f"Loading ... {fm}")
            xm = tf.keras.models.load_model(fm)
            upper_models.append(xm)
            

        files_lower = [ f for f in os.listdir(model_dir_lower) if f.endswith('.keras') ]
        for f in files_lower:
            fm = os.path.join(model_dir_lower, f)
            print(f"Loading ... {fm}")
            ym = tf.keras.models.load_model(fm)
            lower_models.append(ym)
            
        
        print("")
        df = pd.read_csv(file_path)
        df = df.drop(columns=['outputC'])
        
        y_pred = []
        y_target = []
        
        for idx, row in df.iterrows():
            
            rsi = row['RSI']
            target = row['output']
            
            upper_offset = 1
            lower_offset = 1
            
            if rsi > (30-lower_offset) and rsi < 40:
                
                x_val = row[:-1].values
                x_val = x_val.reshape((1, 14, 1)) 
                
                lower_predicts = 0
                for m in range(len(lower_models)):
                
                    lower_predicts += lower_models[m].predict(x_val)[0]
                
                y_val = lower_predicts/len(lower_models)
                
                y_pred.append(y_val)
                y_target.append(target)
              
            if rsi > 60 and rsi < (75+upper_offset):
            
                x_val = row[:-1].values
                x_val = x_val.reshape((1, 14, 1)) 
                
                 
                upper_predicts = 0
                for m in range(len(upper_models)):
                
                    upper_predicts += upper_models[m].predict(x_val)[0]
                
                y_val = upper_predicts/len(upper_models)
                
                y_pred.append(y_val)
                y_target.append(target)
                
        evaluate_model(y_target, y_pred)


if __name__ == "__main__":
    run()            
        