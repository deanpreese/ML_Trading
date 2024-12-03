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
            
            #set x
            x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
            x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
            x = LSTM(32, return_sequences=True, activation='relu')(x)
            x = MaxPooling1D(pool_size=1, strides=1)(x)
            
            #set y
            y = Conv1D(filters=16, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
            y = Conv1D(filters=16, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(y)
            y = Conv1D(filters=16, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(y)
            x = MaxPooling1D(pool_size=1, strides=1)(y)
            
            x_out = LSTM(hidden_units, return_sequences=False, activation='relu')(x)
            y_out = LSTM(hidden_units, return_sequences=False, activation='relu')(y)
            
            xy_output = Average()([x_out, y_out])
            #xy_output = 0.2*x_out + 0.8*y_out
            #xy_output = x_out
            
            univariate_outputs.append(xy_output)

        concatenated_outputs = Concatenate(axis=1)(univariate_outputs)
        reshaped_attention_input = Reshape((input_dim, hidden_units))(concatenated_outputs)
        attention_output = MultiHeadAttention(num_heads=input_dim//2, key_dim=input_dim//2, kernel_regularizer=l2_reg)(reshaped_attention_input, reshaped_attention_input)
                        
        #attention_output = MultiHeadAttention(num_heads=4, key_dim=8, kernel_regularizer=l2_reg)(reshaped_attention_input, reshaped_attention_input)

        flattened_output = Reshape((-1,))(attention_output)
        dense_output = Dense(output_units, activation='relu')(flattened_output)
    
        sum_output = Add()(univariate_outputs)
        sum_output = Dense(output_units, activation='relu')(sum_output)

        #ave_output = Average()([sum_output, dense_output, sum_output])
        #ave_output = Average()([sum_output, dense_output])
        ave_output = 0.5*sum_output + 0.5*dense_output
        outputs = Dense(1)(ave_output)
        
        self.model = Model(inputs, outputs)
        return self.model  
        
    

def process_data_file(file_to_load):
    
    print(f"Loading {file_to_load}" )
    df = pd.read_csv(file_to_load)
    
    f_list_f = ['SDBB91', 'COMP2', 'COMP3', 'ATR5', 'TV3', 'HourOfDay', 'TV1', 'ZH79X', 'SDKC29C', 
                        'ZL57X', 'COMP0', 'ATR2', 'TV6', 'RSI14', 'RSI9', 'ATR51' ,'output','outputC']   

    f_list_r = ['RSI9', 'ATR2', 'ATR5', 'ATR51', 'RSI14', 'TV3', 'TV6', 'COMP2', 'SDKC29C', 'COMP3', 'output','outputC']
    
    f_list_rx = ['RSI9', 'ATR2', 'ATR5', 'RSI14', 'TV3', 'TV6', 'COMP2', 'output','outputC']
    
    f_list_c = ['RSI9', 'ATR2', 'RSI14', 'TV6', 'TV1', 'ZL57X', 'COMP0', 'HourOfDay', 'SDBB91', 'ZH79X', 'output','outputC']

    f_list_cx = ['RSI9', 'ATR2', 'RSI14', 'TV6', 'TV1', 'ZL57X', 'COMP0', 'ZH79X', 'output','outputC']

    #col_filter = f_list_rx
    #df = df.drop(columns=['TimeTicks','SeqClose'])
    #df=df[col_filter]
        
    #df = df[((df['RSI'] > 20) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 80)]  
    #df = df[((df['RSI'] > 25) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 75)] 
    #df = df[(df['RSI9'] > 60)]  
    #df = df[(df['RSI9'] < 40)]  
    
    X = df.drop(columns=['output', 'outputC']).values
    y = df['output'].values

    return X, y

def main():
    datafile = [ 
            'data/Lucky13_3070_oos.csv',
            'data/Lucky13_3070.csv',  #1
                          
            'data/NewModel_3070_oos.csv',   
            'data/NewModel_3070.csv',  #3
            'data/NewModel_ALL_oos.csv',   
            'data/NewModel_ALL.csv',  #5

            'data/NewModel_ALL_SPAN2.csv',   #6  
            'data/NewModel_ALL_SPAN3.csv',   #7
            'data/NewModel_ALL_SPAN6.csv',   #8
    ]   

    file_train = 1
    file_oos = 0

    X_data, y_data = process_data_file(datafile[file_train])
    X_oos, y_oos = process_data_file(datafile[file_oos])
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_three_ways(X_data, y_data)
    input_shape = (X_train.shape[1], 1)
    #(14, 1)


    model_cnnk = MODEL_CNNK()
    history_out, y_pred = model_cnnk.train_model(input_shape, X_train, X_test, y_train, y_test, X_val, y_val, 10 )

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