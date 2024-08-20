import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from models.kan_mixer import KANMixerModel

tf.config.set_visible_devices([], 'GPU')

def main():
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_dim = X_train.shape[1]
    hidden_units = 32  # Adjust this based on your data
    output_dim = 1  # Single continuous output
    epochs = 100
    model_save_path = 'models/best_model_kan_ts.keras'
    
    model = KANMixerModel(input_dim, hidden_units=hidden_units, output_dim=output_dim, epochs=100, batch_size=32)
    model.train(X_train, y_train, epochs, model_save_path, batch_size=32, validation_split=0.2 )
    metrics = model.evaluate(X_test, y_test)
    print(f"Test MSE: {metrics['MSE']}, Test MAE: {metrics['MAE']}, R2: {metrics['R2']}")
    print(f"Total Wins: {metrics['Total Wins']}, Total Losses: {metrics['Total Losses']}, Win Percentage: {metrics['Win Percentage']:.2f}%")
    print(f"Number of Samples: {metrics['Samples']}")
    
if __name__ == "__main__":
    main()
