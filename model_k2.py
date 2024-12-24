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

from ml_model.k_model_base import FFTLayer, FFTOrRFTLayer , K_MODEL_BASE

import ml_model.feature_filter as feature_filter


tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)

class K2 (K_MODEL_BASE):
    def __init__(self, model_name='k2'):
        
        model_name = model_name
        self.setup_model(model_name)


        
    def create_feature_model_h(self, inputs, output_dim):

        """ 
        H
        Val MSE: 9.1925, Val MAE: 1.7172, R2: 0.46744909954386704
        Total Wins: 5652, Total Losses: 1799, Win Percentage: 0.759
        Number of Samples: 7451
        """        

        h = Conv1D(filters=64, kernel_size=4, activation='relu', kernel_initializer=self.initializer)(inputs) 
        #h = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(h)
        h = Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer=self.initializer)(h)
        #h = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(h)
        h = Conv1D(filters=16, kernel_size=2, activation='relu', kernel_initializer=self.initializer)(h)

        attention_output = MultiHeadAttention(num_heads=4, key_dim=16)(h, h)
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
        h = Dropout(self.drop_out)(attention_output)

        h = Bidirectional(LSTM(32,name="BIC", kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(h)
        h = Bidirectional(LSTM(32,name="BIC2", kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer))(h)
        h = Dense(32, activation='relu', kernel_regularizer=self.l2_reg,  kernel_initializer=self.initializer)(h)         

        attention_h = Dense(32, activation='softmax', kernel_initializer=self.initializer, name='attention_h')(h)
        h = Multiply()([h, attention_h])                
        
        h = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg, name="h_out", kernel_initializer=self.initializer)(h)           
       
        return h


    def create_feature_model_rx(self, input_shape):
        
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)
        x = LSTM(32, return_sequences=True, activation='relu')(reshaped_inputs)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        smx_out = Dense(8, activation='linear')(x) 
        subx_model = Model(input, smx_out)
        return subx_model


    def create_feature_model_rx2(self, input_shape):
        
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)
        x = LSTM(32, return_sequences=True, activation='relu')(reshaped_inputs)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        smx_out = Dense(8, activation='linear')(x) 
        subx_model = Model(input, smx_out)
        return subx_model


   
    def create_model(self, input_shape ):
        
        inputs = Input(shape=input_shape)
        output_dim = 16

        #inputs = FFTLayer()(inputs)
        #inputs = FFTOrRFTLayer(use_rft=True, return_magnitude=True, name="fft_or_rft_layer")(inputs)
        #model_h = self.create_feature_model_h(inputs, output_dim)        
        
        feature_outputs = []
      
        for i in range(input_shape[0]):
            feature_input = inputs[:, i:i+1]
            
            rx = self.create_feature_model_rx((1,))
            rx_out = rx(feature_input)
            
            rx2 = self.create_feature_model_rx2((1,))
            rx2_out = rx2(feature_input)
            
            #rx_ave
            #x = Average()([rx2_out, rx_out])
            
            #rx_ave 40rx2 + 60rx
            #x = 0.4 * rx2_out + 0.6 * rx_out
            
            #rx_ave 20rx2 + 80rx
            x = 0.2 * rx2_out + 0.8 * rx_out
            
            
            # xxx out
            #x = rx2_out
            
            feature_outputs.append(x)   
            #feature_outputs.append(f2_out)   
             
                   
        x = Concatenate(axis=1)(feature_outputs)
        
        x = Dense(32, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(x)
        
        x = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(x)
        #x = Average()([model_h,x])
        
        output = Dense(1, activation='linear')(x)
        self.model = Model(inputs=inputs, outputs=output)
        return self.model
    

def main():
    
    datafile = [ 
                'data/Lucky13_3070_oos.csv',   
                'data/Lucky13_3070.csv',  #1
                'data/Model_X_3070_oos.csv',  
                'data/Model_X_3070.csv',  #3
                'data/The_13_R_3070_oos.csv',
                'data/The_13_R_3070.csv' #5
                
        ]

    model_13_r = [
                'SDBB9L','SDBB9U',
                'SDKC9U','SDKC9L',
                'ROC',
                'ATR54','ATR53','ATR52','ATR51',
                'ATR5',
                'ATR21',
                'ATR2',
                'RSI','STOK1']

    col_filter = model_13_r
    #col_filter = feature_filter.lucky13_all         
    #col_filter = feature_filter.lucky13_3070_comp
    #col_filter = feature_filter.model_x_3070_imp_full        
    #col_filter = feature_filter.model_x_3070_imp_slim
         
    model = K2('k2_13_r')
    X_train, X_val, X_test, y_train, y_val, y_test,  X_oos, y_oos, input_shape = model.process_data_split(datafile[5], datafile[4], col_filter)
    
    history_out, y_pred = model.train_model(input_shape, X_train, X_test, y_train, y_test, X_val, y_val, 5000 )
    #best_model = tf.keras.models.load_model(model.checkpoint_model)
    best_model = tf.keras.models.load_model(model.checkpoint_model, custom_objects={"FFTLayer": FFTLayer, "FFTOrRFTLayer": FFTOrRFTLayer})
    
    model.evaluate_finished_model(best_model, X_val, X_test, y_train, y_val, y_test,  X_oos, y_oos)

            
if __name__ == "__main__":
    main()