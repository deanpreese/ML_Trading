import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib 
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Average, Multiply, GlobalAveragePooling1D, Reshape, Concatenate, ConvLSTM1D, Flatten, SeparableConv1D, LayerNormalization, Bidirectional, Add, Dense,  Dropout, MaxPooling1D, LSTM, MultiHeadAttention, Attention
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import l2

from ml_model.data_func import split_three_ways
from ml_model.k_model_base import K_MODEL_BASE
from ml_model.k_model_base import FFTLayer

import ml_model.feature_filter as feature_filter

tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)
   

class CNN_SM (K_MODEL_BASE):
    
    def __init__(self, model_name='cnn_sm'):
        
        model_name = model_name
        self.setup_model(model_name)
    
    def build_model_hy2(self, inputs):
    
        h = FFTLayer()(inputs)

        h = Conv1D(filters=512, kernel_size=3, activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)(h)
        h = Conv1D(filters=256, kernel_size=2, activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)(h)
        h = Conv1D(filters=128, kernel_size=2, activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)(h)
        h = Conv1D(filters=64, kernel_size=1, activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)(h)
        h = LayerNormalization()(h)
        h = Dropout(0.2)(h)
        
        #attention_output = MultiHeadAttention(num_heads=3, key_dim=3, kernel_initializer=self.initializer)(h, h, h)
        #attention_output = LayerNormalization()(attention_output)
        #h = Add()([h, attention_output])
        
        m1 = LSTM(32, kernel_regularizer=self.l2_reg, activation='tanh', return_sequences=True, kernel_initializer=self.initializer)(h)
        m2 = LSTM(32, kernel_regularizer=self.l2_reg, activation='sigmoid', return_sequences=True, kernel_initializer=self.initializer)(h)
        m_r = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(h)
        h = Average()([m1, m2, m_r])  
        
        h = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=False, kernel_initializer=self.initializer)(h)
        
        #h = Dropout(0.2)(h)
        #h = Dense(64, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(h)
        h = Dropout(0.2)(h)
        h = Dense(64, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(h)
        return h
    
    
    def build_model_hy(self, inputs):
    
        h = FFTLayer()(inputs)

        h = Conv1D(filters=128, kernel_size=3, activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)(h)
        h = Conv1D(filters=128, kernel_size=2, activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)(h)
        h = Conv1D(filters=128, kernel_size=2, activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)(h)
        h = Conv1D(filters=64, kernel_size=1, activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.l2_reg)(h)
        h = LayerNormalization()(h)
        h = Dropout(0.2)(h)
        
        attention_output = MultiHeadAttention(num_heads=4, key_dim=32, kernel_initializer=self.initializer)(h, h, h)
        attention_output = LayerNormalization()(attention_output)
        h = Add()([h, attention_output])
        
        m1 = LSTM(32, kernel_regularizer=self.l2_reg, activation='tanh', return_sequences=False, kernel_initializer=self.initializer)(h)
        m2 = LSTM(32, kernel_regularizer=self.l2_reg, activation='sigmoid', return_sequences=False, kernel_initializer=self.initializer)(h)
        m_r = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=False, kernel_initializer=self.initializer)(h)
        h = Average()([m1, m2, m_r])  
        
        h = Dropout(0.2)(h)
        h = Dense(64, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(h)
        h = Dropout(0.2)(h)
        h = Dense(64, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(h)
        return h


    def create_model(self, input_shape):
        
        inputs = Input(shape=input_shape)
        h = self.build_model_hy(inputs)

        #hy2 = self.build_model_hy2(inputs)
        #h = Average()([hy2, hy])
        
        outputs = Dense(1)(h)  
        model = Model(inputs=inputs, outputs=outputs)
        self.model = model
        return model        
                

def main():

    datafile = [ 
                'data/Lucky13_3070_oos.csv',   
                'data/Lucky13_3070.csv',  #1
                'data/Model_X_3070_oos.csv',  
                'data/Model_X_3070.csv',  #3
                'data/Model_YD_3070_oos.csv',  
                'data/Model_YD_3070.csv',  #5
        ]

    col_filter = feature_filter.lucky13_all         
    #col_filter = feature_filter.model_x_3070_imp_full        
    #col_filter = feature_filter.model_x_3070_imp_slim
    
    col_filter = feature_filter.model_yd_columns
    
    model_cnn_sm = CNN_SM()
    X_train, X_val, X_test, y_train, y_val, y_test,  X_oos, y_oos, input_shape = model_cnn_sm.process_data_split(datafile[5], datafile[4], col_filter)
    
    history_out, y_pred = model_cnn_sm.train_model(input_shape, X_train, X_test, y_train, y_test, X_val, y_val, 1000 )
    best_model = tf.keras.models.load_model(model_cnn_sm.checkpoint_model, custom_objects={"FFTLayer": FFTLayer})
    model_cnn_sm.evaluate_finished_model(best_model, X_val, X_test, y_train, y_val, y_test,  X_oos, y_oos)

            
if __name__ == "__main__":
    main()
    
