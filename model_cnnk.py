import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import pywt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, Average, Conv2D, LeakyReLU, Reshape, Concatenate, Multiply,LayerNormalization 
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Add, Dense, Dropout, MaxPooling1D, LSTM, MultiHeadAttention, Attention

from tensorflow.keras.layers import AdditiveAttention

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2
from ml_model.model_stats import gen_reg_stats_x, gen_class_stats
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from ml_model.data_func import split_three_ways
from ml_model.k_model_base import K_MODEL_BASE


tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)


class MODEL_CNNK (K_MODEL_BASE):
    def __init__(self):
        
        self.checkpoint_dir = 'checkpoints/'
        self.trained_dir = 'trained_models/'
       
        self.checkpoint_model = os.path.join(self.checkpoint_dir, 'model_cnnk.keras')
        self.trained_model = os.path.join(self.trained_dir, 'model_cnnk.keras')
        self.model_plot = os.path.join(self.checkpoint_dir, 'model_cnnk.png')

        self.drop_out = 0.2
        self.l2_reg = l2(0.01)
        self.initializer = GlorotUniform(seed=42)


    def create_model(self, input_shape):
        
        l2_reg = l2(0.01)
        inputs = Input(shape=input_shape)
                
        input_dim = inputs.shape[1]               
        hidden_units = 16 
        output_units = 8             
        reshaped_inputs = Reshape((input_dim, 1))(inputs)
        
        univariate_outputs = []
        for i in range(input_dim):
            
            x = Reshape((1, -1))(reshaped_inputs[:, i, :])
            inx = LSTM(32, return_sequences=True, activation='relu')(x)
            iny = LSTM(32, return_sequences=True, activation='relu')(x)
            
            #set x
            x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
            x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
            x = LSTM(32, return_sequences=True, activation='relu')(x)
            
            #set y
            y = Conv1D(filters=32, kernel_size=1, activation='sigmoid', kernel_initializer=self.initializer)(iny)
            y = Conv1D(filters=32, kernel_size=1, activation='tanh', kernel_initializer=self.initializer)(y)
            y = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(y)
            
            x_out = LSTM(hidden_units, return_sequences=False, activation='relu')(x)
            y_out = LSTM(hidden_units, return_sequences=False, activation='relu')(y)
            
            xs = Dense(hidden_units, activation='sigmoid', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(x_out)
            xt = Dense(hidden_units, activation='tanh', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(x_out)
            
            ys = Dense(hidden_units, activation='sigmoid', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(y_out)
            yt = Dense(hidden_units, activation='tanh', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(y_out)
            
            x_output = Average()([xs, xt])
            y_output = Average()([ys, yt])
            
            xy_output = Average()([x_output,  y_output, xs, xt,  ys, yt])
            univariate_outputs.append(xy_output)

        concatenated_outputs = Concatenate(axis=1)(univariate_outputs)
        
        reshaped_attention_input = Reshape((input_dim, hidden_units))(concatenated_outputs)
        ao = MultiHeadAttention(num_heads=input_dim//2, key_dim=input_dim//2, kernel_regularizer=l2_reg)(reshaped_attention_input, reshaped_attention_input)
        flattened_output = Reshape((-1,))(ao)
        dense_output = Dense(output_units, activation='relu')(flattened_output)
    
        sum_output = Add()(univariate_outputs)
        sum_output = Dense(output_units, activation='relu')(sum_output)
        ave_output = sum_output 

        #ave_output = 0.5*sum_output + 0.5*dense_output
        #ave_output = 0.4*sum_output + 0.6*dense_output
        ave_output = 0.3*sum_output + 0.7*dense_output                
        
        outputs = Dense(1)(ave_output)
        
        self.model = Model(inputs, outputs)
        return self.model  
        
    

def process_data_file(file_to_load):
    
    print(f"Loading {file_to_load}" )
    df = pd.read_csv(file_to_load)
    
    
    model_x_3070_imp_full =['SDKC7CU', 'ZL79X', 'SeqClose', 'TV3', 'ROC14', 'ATR2', 'STO5135D', 'STO7143D', 
                            'TV4', 'SDKC91', 'ZC79X', 'ATR9', 'TV5', 'RSIRAW', 'COMP3', 'SDBB9CL', 'SDKC9', 'TV2', 'TV6', 
                            'SDKC7CL', 'COMP0', 'ZH79X', 'SDBB91', 'TV1', 'COMP2', 'output', 'outputC']
    
    model_x_3070_imp_slim = ['SDKC9', 'COMP3', 'STO7143D', 'ATR9', 'SDBB91', 'RSIRAW', 'COMP2', 'TV6', 'SDKC7CU','output', 'outputC']


    col_filter = model_x_3070_imp_full
    #df = df.drop(columns=['TimeTicks','SeqClose'])
    df=df[col_filter]
        
    #df = df[((df['RSI'] > 20) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 80)]  
    #df = df[((df['RSI'] > 25) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 75)] 
    #df = df[(df['RSI9'] > 60)]  
    #df = df[(df['RSI9'] < 40)]  
    
    
    X = df.drop(columns=['output', 'outputC'])
    y = df['output'].values

    return X, y

def main():
    
    datafile = [ 
            'data/Lucky13_3070_oos.csv',   
            'data/Lucky13_3070.csv',  #1
            'data/Model_X_3070_oos.csv',  
            'data/Model_X_3070.csv',  #3
    ]   

    file_train = 3
    file_oos = 2

    X_data, y_data = process_data_file(datafile[file_train])
    X_oos, y_oos = process_data_file(datafile[file_oos])
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_three_ways(X_data, y_data)
    input_shape = (X_train.shape[1], 1)
    #(14, 1)


    model_cnnk = MODEL_CNNK()
    history_out, y_pred = model_cnnk.train_model(input_shape, X_train, X_test, y_train, y_test, X_val, y_val, 1000 )

    best_model = tf.keras.models.load_model(model_cnnk.checkpoint_model)
    y_pred = best_model.predict(X_test)
    oos_model = tf.keras.models.load_model(model_cnnk.checkpoint_model)
    y_pred_oos = oos_model.predict(X_oos)
    
    print("-----")
    print(f"Train Shape {X_train.shape}")
    print(f"Val Shape   {X_val.shape}")
    print(f"Test Shape  {X_test.shape}")
    print("-----")
    print("Test Pred")
    rmse, mse, mae, r2 = model_cnnk.evaluate_model( y_test, y_pred)
    
    print("-----")
    print(f"OOS Test Shape {X_oos.shape}")
    print("-----")
    print("OOS Pred")
    rmse_oos, mse_oos, mae_oos, r2_oos = model_cnnk.evaluate_model( y_oos, y_pred_oos)

    model_file = f"model_cnnk_{mse_oos}_{mae_oos}_{r2_oos}_model.keras"
    oos_file_path = os.path.join(model_cnnk.checkpoint_dir, model_file)
    oos_model.save(oos_file_path)
    
    oos_model_plot_file = f"model_cnnk_{mse_oos}_{mae_oos}_{r2_oos}_model.png"
    oos_model_plot_path = os.path.join(model_cnnk.checkpoint_dir, oos_model_plot_file)
    
    tf.keras.utils.plot_model(best_model, to_file=oos_model_plot_path, 
        show_shapes=True, 
        show_dtype=True,
        show_layer_names=True,
        expand_nested=True,
        show_layer_activations=True,
        show_trainable=True
    )   
    
            
if __name__ == "__main__":
    main()