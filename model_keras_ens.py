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
        

def load_models(model_dir):
    
    model_list = []
    files_ = [ f for f in os.listdir(model_dir) if f.endswith('.keras') ]
    for f in files_:
        fm = os.path.join(model_dir, f)
        print(f"Loading ... {fm}")
        xm = tf.keras.models.load_model(fm)
        model_list.append(xm)
        
    return model_list

def run_predicts(x_val, model_list):
    
    x_val = x_val.reshape((1, 14, 1)) 
    predicts = 0
    for m in range(len(model_list)):
        predicts += model_list[m].predict(x_val)[0]
        
        print(m)

    count_up = len([x for x in predicts if x > 0])
    cnt_pct = count_up/len(predicts)

    if cnt_pct > .5:
        y_val = 1
    else:
        y_val = -1

    #y_val = predicts/len(model_list)
    return y_val
        



def run():
    datafile = [ 
        'data/Lucky13_3070_oos.csv',   
        'data/Lucky13_3070.csv',  #1
        'data/Lucky13_EX_3070_oos.csv',  
        'data/Lucky13_EX_3070.csv',  #3
        'data/Lucky13_ALL_oos.csv',  #4
        'data/Lucky13_ALL.csv',  #5
    ]

    file_path = datafile[0]
    print("")
    
    c_kan_model_dir_upper = "saved_models/c_kan/upper"
    c_kan_model_dir_lower = "saved_models/c_kan/lower"
    upper_models = load_models(c_kan_model_dir_upper)
    lower_models = load_models(c_kan_model_dir_lower)    
    
    model_dir_upper = "saved_models/dcnn_ens/upper"
    model_dir_lower = "saved_models/dcnn_ens/lower"
    #dcnn_upper_models = load_models(model_dir_upper)
    #dcnn_lower_models = load_models(model_dir_lower)    
    
    comp_model_dir_upper = "saved_models/comp/upper"
    comp_model_dir_lower = "saved_models/comp/lower"
    #upper_models = load_models(comp_model_dir_upper)
    #lower_models = load_models(comp_model_dir_lower)    
    
    
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
            #y_val = run_dcnn_models(x_val, dcnn_lower_models)     
            #y_val = run_predicts(x_val, c_kan_lower_models)     
            
            y_val = run_predicts(x_val, lower_models)     
            
            y_pred.append(y_val)
            y_target.append(target)
            
        if rsi > 60 and rsi < (75+upper_offset):
        
            x_val = row[:-1].values
            #y_val = run_dcnn_models(x_val, dcnn_upper_models)        
            #y_val = run_predicts(x_val, c_kan_upper_models)    
            
            y_val = run_predicts(x_val, upper_models)    
            
            
            y_pred.append(y_val)
            y_target.append(target)
            
    evaluate_model(y_target, y_pred)


if __name__ == "__main__":
    run()            
        