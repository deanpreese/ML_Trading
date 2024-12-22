import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib 
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Lambda, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Average, Multiply, GlobalAveragePooling1D, Reshape, Concatenate, ConvLSTM1D, Flatten, SeparableConv1D, LayerNormalization, Bidirectional, Add, Dense,  Dropout, MaxPooling1D, LSTM, MultiHeadAttention, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2

from ml_model.model_stats import gen_reg_stats_x, gen_class_stats 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from ml_model.data_func import split_three_ways
from ml_model.k_model_base import K_MODEL_BASE
from ml_model.k_model_base import FFTLayer

import ml_model.feature_filter as feature_filter

tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)
   

class CNN_LG (K_MODEL_BASE):
    
    def __init__(self, model_name='cnn_lg'):
        
        model_name = model_name
        self.setup_model(model_name)


    def build_model_hfft(self, inputs):
        # Apply FFT using the custom layer
        h = FFTLayer()(inputs)
        #h = FFTLayer()(h)
        
        h2 = FFTLayer()(inputs)
        #h2 = FFTLayer()(h2)
        
        
        h = Conv1D(filters=64, kernel_size=8, activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)(h)
        h = Conv1D(filters=64, kernel_size=4, activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)(h)
        h = Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)(h)
        h = LayerNormalization()(h)
        h = Dropout(0.2)(h)
        
        h2 = Conv1D(filters=64, kernel_size=8, activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)(h2)
        h2 = Conv1D(filters=64, kernel_size=4, activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)(h2)
        h2 = Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)(h2)
        h2 = LayerNormalization()(h2)
        h2 = Dropout(0.2)(h2)
        
        
        # Multi-Head Attention to capture global relations
        # key_dim should divide the model dimension evenly for MHA
        # Let’s assume h.shape[-1] is now ~64; num_heads=4 and key_dim=16 gives 64 total dimensions
        
        ao = MultiHeadAttention(num_heads=3, key_dim=5, kernel_regularizer=self.l2_reg)(h, h, h)
        ao = LayerNormalization()(ao)
        ao = Dropout(0.2)(ao)

        ao2 = MultiHeadAttention(num_heads=3, key_dim=5, kernel_regularizer=self.l2_reg)(h2, h2, h2)
        ao2 = LayerNormalization()(ao2)
        ao2 = Dropout(0.2)(ao2)
        
        h = (0.5 *ao) + (0.5 *ao2)
        #h = ao
        
        # LSTM for temporal sequence modeling
        # Using LSTM after attention to summarize attended sequence
        tm1 = LSTM(32, kernel_regularizer=self.l2_reg, activation='tanh', return_sequences=False, kernel_initializer=self.initializer)(h)
        tm2 = LSTM(32, kernel_regularizer=self.l2_reg, activation='sigmoid', return_sequences=False, kernel_initializer=self.initializer)(h)
        tm_r = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=False, kernel_initializer=self.initializer)(h)
        
        h = Average()([tm1,tm2, tm_r])  
        h = Dropout(0.2)(h)
        
        # Dense layers for final prediction
        h = Dense(64, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(h)
        h = Dropout(0.2)(h)
        h = Dense(16, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(h)

        return h



    def build_model_hx(self, inputs):
    
        # designed for
        #col_filter = feature_filter.model_x_3070_imp_full   
    
        h = Conv1D(filters=64, kernel_size=8, activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)(inputs)
        h = Conv1D(filters=64, kernel_size=4, activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)(h)
        h = Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)(h)
        h = LayerNormalization()(h)
        h = Dropout(0.2)(h)
        
        # Multi-Head Attention to capture global relations
        # key_dim should divide the model dimension evenly for MHA
        # Let’s assume h.shape[-1] is now ~64; num_heads=4 and key_dim=16 gives 64 total dimensions
        
        ao = MultiHeadAttention(num_heads=3, key_dim=5, kernel_regularizer=self.l2_reg)(h, h, h)
        ao = LayerNormalization()(ao)
        ao = Dropout(0.2)(ao)

        ao2 = MultiHeadAttention(num_heads=3, key_dim=5, kernel_regularizer=self.l2_reg)(h, h, h)
        ao2 = LayerNormalization()(ao2)
        ao2 = Dropout(0.2)(ao2)

        
        h = (0.5 *ao) + (0.5 *ao2)

        # LSTM for temporal sequence modeling
        # Using LSTM after attention to summarize attended sequence
        tm1 = LSTM(32, kernel_regularizer=self.l2_reg, activation='tanh', return_sequences=False, kernel_initializer=self.initializer)(h)
        
        tm2 = LSTM(32, kernel_regularizer=self.l2_reg, activation='sigmoid', return_sequences=False, kernel_initializer=self.initializer)(h)
        
        h = Average()([tm1,tm2])  
                
        h = Dropout(0.2)(h)
        
        # Dense layers for final prediction
        h = Dense(64, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(h)
        h = Dropout(0.2)(h)
        h = Dense(16, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(h)

        return h




    def create_model(self, input_shape):
        
        inputs = Input(shape=input_shape)
        
        hx = self.build_model_hx(inputs)
        hft = self.build_model_hfft(inputs)
        ave_output = 0.5*hx + 0.5*hft
        
        
        """
        
        tm1 = Dense(64, kernel_regularizer=self.l2_reg, activation='tanh',kernel_initializer=self.initializer)(ave_out)
        tm2 = Dense(64, kernel_regularizer=self.l2_reg, activation='sigmoid', kernel_initializer=self.initializer)(ave_out)
        tm_r = Dense(64, kernel_regularizer=self.l2_reg, activation='relu', kernel_initializer=self.initializer)(ave_out)
        final_ave_out = Average()([tm1,tm2, tm_r])  
        """
        
        final_ave_out = ave_output
        
        final_out = Dense(32, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(final_ave_out)
                
        outputs = Dense(1)(final_out)  
        model = Model(inputs=inputs, outputs=outputs)
        self.model = model
        return model        
                
        
def main():


    datafile = [ 
            'data/Lucky13_3070_oos.csv',   
            'data/Lucky13_3070.csv',  #1
            'data/Model_X_3070_oos.csv',  
            'data/Model_X_3070.csv',  #3
    ]   

    col_filter = feature_filter.lucky13_all         
    #col_filter = feature_filter.model_x_3070_imp_full        
    #col_filter = feature_filter.model_x_3070_imp_slim
    #col_filter = feature_filter.model_x_3070_comp
    
    model = CNN_LG('cnn_lg_lucky13_all')
    X_train, X_val, X_test, y_train, y_val, y_test,  X_oos, y_oos, input_shape = model.process_data_split(datafile[1], datafile[0], col_filter)
    
    history_out, y_pred = model.train_model(input_shape, X_train, X_test, y_train, y_test, X_val, y_val, 1000 )
    best_model = tf.keras.models.load_model(model.checkpoint_model, custom_objects={"FFTLayer": FFTLayer})
    model.evaluate_finished_model(best_model, X_val, X_test, y_train, y_val, y_test,  X_oos, y_oos)
    
            
if __name__ == "__main__":
    main()
    
