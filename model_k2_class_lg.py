from datetime import datetime
import os
import numpy as np
import pandas as pd
import tensorflow as tf
#import joblib
#import pywt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, Flatten, Average, Conv2D, LeakyReLU, Reshape, Concatenate, Multiply,LayerNormalization 
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling1D, Bidirectional, Add, Dense, Dropout, MaxPooling1D, LSTM, MultiHeadAttention, Attention

from tensorflow.keras.layers import AdditiveAttention

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2
from ml_model.model_stats import gen_reg_stats_x, gen_class_stats
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from ml_model.k_model_base import FFTLayer, FFTOrRFTLayer , K_MODEL_BASE
from ml_model.k_model_base_classifier import K_MODEL_BASE_CLASSIFIER
import ml_model.feature_filter as feature_filter


tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)

class K2 (K_MODEL_BASE_CLASSIFIER):
    def __init__(self, model_name='k2'):
        
        model_name = model_name
        self.setup_model(model_name)


    def build_pre(self, inputs ):
        
        #inputs = FFTLayer()(inputs)
        #inputs = FFTOrRFTLayer(use_rft=True, return_magnitude=True, name="fft_or_rft_layer")(inputs)
        
        #x = Dense(128, activation='relu', kernel_initializer=self.initializer)(inputs)
        #x = Dense(64, activation='relu', kernel_initializer=self.initializer)(inputs)
        x = Dense(32, activation='relu', kernel_initializer=self.initializer)(inputs)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)
        
        return x

    def prune_dims(self,x, output_dim ):

               
        c = x
        
        c = Conv1D(filters=256, kernel_size=8, activation='relu', kernel_initializer=self.initializer)(c)
        c = LayerNormalization()(c)
        c = Conv1D(filters=128, kernel_size=8, activation='relu', kernel_initializer=self.initializer)(c)
        c = LayerNormalization()(c)
        c = Dropout(0.2)(c)
        c = Conv1D(filters=64, kernel_size=4, activation='relu', kernel_initializer=self.initializer)(c)
        c = LayerNormalization()(c)
        c = Dropout(0.2)(c)
        c = Conv1D(filters=32, kernel_size=2, activation='relu', kernel_initializer=self.initializer)(c)
        c = LayerNormalization()(c)
        c = Dropout(0.2)(c)
        c = Conv1D(filters=16, kernel_size=1, activation='sigmoid', kernel_initializer=self.initializer)(c)
        
        c = LSTM(output_dim, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(c)
        return c

    def create_feature_model_2(self, input_shape, output_dim):

        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)

        x_factor=3
        
        x = Dense(64 * x_factor, activation='relu', kernel_initializer=self.initializer)(reshaped_inputs)
        x = Dense(32 * x_factor, activation='relu', kernel_initializer=self.initializer)(x)
        x = Dense(8 * x_factor, activation='relu', kernel_initializer=self.initializer)(x)
        x = Reshape((8, x_factor))(x)
        
        x = LSTM(64, return_sequences=True, activation='relu')(x)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        
        x = Dense(32, activation='relu', kernel_initializer=self.initializer)(x)
        
        smx_out = Dense(output_dim, activation='linear')(x) 
        subx_model = Model(input, smx_out)
        return subx_model

    def create_feature_model(self, input_shape, output_dim):
                
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)
        x = LSTM(64, return_sequences=True, activation='relu')(reshaped_inputs)
        x = Conv1D(filters=64, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        smx_out = Dense(output_dim, activation='linear')(x) 
        subx_model = Model(input, smx_out)
        return subx_model



    def add_feature_dims(self, input_tensor, output_dim):
        
        if isinstance(input_tensor, (tuple, list)):
            raise TypeError(f"add_feature_dims expects Tensor/KerasTensor, got {type(input_tensor)} -> {input_tensor}")
        if not hasattr(input_tensor, "shape"):
            raise TypeError(f"add_feature_dims expects Tensor/KerasTensor, got {type(input_tensor)} -> {input_tensor}")

        rank = len(input_tensor.shape)


        # Normalize to (batch, features)
        if rank == 2:
            x_feat = input_tensor
        elif rank == 3:
            # (batch, time, features) -> (batch, features)
            x_feat = GlobalAveragePooling1D(name="time_pool")(input_tensor)
        else:
            raise ValueError(f"Expected rank-2 or rank-3 input, got shape {input_tensor.shape}")

        num_features = x_feat.shape[-1]
        if num_features is None:
            raise ValueError("num_features must be statically known (last dim cannot be None).")

        feature_embeddings = []
        for i in range(int(num_features)):
            # (batch, 1)
            fi = x_feat[:, i:i+1]

            # (batch, 1, 1) to match RX input shape (1,1)
            fi = Reshape((1, 1), name=f"feat_{i}_in")(fi)

            rx = self.create_feature_model((1, 1), output_dim)
            rx_out = rx(fi)

            # ---- THE CRITICAL LINE ----
            # Force RX output to (batch, 1, output_dim) regardless of whether rx_out is rank-2 or rank-3
            rx_out = Reshape((1, output_dim), name=f"feat_{i}_out")(rx_out)

            feature_embeddings.append(rx_out)

        # Stack features along axis=1 -> (batch, num_features, output_dim)
        x = Concatenate(axis=1, name="feature_stack")(feature_embeddings)
        #x = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(x)
        x = GlobalAveragePooling1D(name="feature_pool")(x)

        return x       
        



    # -------------------------------------
    def create_model(self, input_shape ):
        
        inputs = Input(shape=input_shape)
        output_dim = 16

        c = self.build_pre(inputs )
        c = self.prune_dims(c, output_dim)
        c = self.add_feature_dims(c, output_dim )
                                
        #c = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', kernel_initializer=self.initializer)(c)
        
        h = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg, name="m_out", kernel_initializer=self.initializer)(c)           
        output = Dense(1, activation='sigmoid')(h)
        self.model = Model(inputs=inputs, outputs=output)
        return self.model
    
    
    

def main():
    
    datafile = [ 
                'data/Model_3LB_ALL_oos.csv', #6
                'data/Model_3LB_ALL.csv' #7
                
        ] 
    
    file_train = datafile[1]
    file_oos = datafile[0]
    
    col_filter_all =["Year","Month","Day","DayOfWeek","HourOfDay","MinOfHour","SeqClose","RSIRAW",
     "SDLR310","SDBB91","SDKC91","SDKC9","ROC",
     "ATR54","ATR53","ATR52","ATR51","ATR5","ATR21","ATR2","RSI","STOK1",
     "dv0","dv1","dv2","dv3","dv4","dv5","dv6","dv7","dv8","dv9","dv10","dv11","dv12","dv13","dv14","dv15","dv16","dv17","dv18","dv19","dv20","dv21","dv22","dv23",
     "cv0","cv1","cv2","cv3","cv4","cv5","cv6","cv7","cv8","cv9","cv10","cv11","cv12","cv13","cv14","cv15","cv16","cv17","cv18","cv19","cv20","cv21","cv22","cv23",
     "chv0","chv1","chv2","chv3","chv4","chv5","chv6","chv7","chv8","chv9","chv10","chv11","chv12","chv13","chv14","chv15","chv16","chv17","chv18","chv19","chv20",
     "chv21","chv22","chv23"]

    col_filter_x =["SDLR310","SDBB91","SDKC91","SDKC9","ROC",
     "ATR54","ATR53","ATR52","ATR51","ATR5","ATR21","ATR2","RSI","STOK1",
     "dv0","dv1","dv2","dv3","dv4","dv5","dv6","dv7","dv8","dv9","dv10","dv11","dv12","dv13","dv14","dv15","dv16","dv17","dv18","dv19","dv20","dv21","dv22","dv23",
     "cv0","cv1","cv2","cv3","cv4","cv5","cv6","cv7","cv8","cv9","cv10","cv11","cv12","cv13","cv14","cv15","cv16","cv17","cv18","cv19","cv20","cv21","cv22","cv23",
     "chv0","chv1","chv2","chv3","chv4","chv5","chv6","chv7","chv8","chv9","chv10","chv11","chv12","chv13","chv14","chv15","chv16","chv17","chv18","chv19","chv20",
     "chv21","chv22","chv23"]
        
    col_filter_y = ["SDLR310","SDBB91","SDKC91","SDKC9","ROC","ATR54","ATR53","ATR52",
                       "ATR51","ATR5","ATR21","ATR2","STOK1",       
                        'cv2', 'cv1', 'cv0', 'cv3','cv4',
                        'chv2', 'chv1', 'chv0', 'chv12', 'chv18', 'chv11',
                        "dv0","dv1","dv2","dv3","dv4"]

    col_filter = ["SDLR310","SDBB91","SDKC91","SDKC9","ROC","ATR54","ATR53","ATR52",
                       "ATR51","ATR5","ATR21","ATR2","STOK1",       
                        'cv2', 'cv1', 'cv0', 'cv3','cv4','cv5',
                        'chv2', 'chv1', 'chv0', 'chv3', 'chv4', 'chv5',
                        "dv0","dv1","dv2","dv3","dv4", "dv5"]


        
    model_run = 'k2_3LBL13' + datetime.now().strftime("_%Y%m%d_%H%M%S")
    
    model = K2(model_run)
    
    X_train, X_val, X_test, y_train, y_val, y_test,  X_oos, y_oos, input_shape = model.process_data_split(file_train, file_oos, col_filter)
    
    history_out, y_pred = model.train_model(input_shape, X_train, X_test, y_train, y_test, X_val, y_val, 10 )
    best_model = tf.keras.models.load_model(model.checkpoint_model)
    #best_model = tf.keras.models.load_model(model.checkpoint_model, custom_objects={"FFTLayer": FFTLayer, "FFTOrRFTLayer": FFTOrRFTLayer})
    
    model.evaluate_finished_model_c(best_model, X_val, X_test, y_train, y_val, y_test,  X_oos, y_oos)

            
if __name__ == "__main__":
    main()
    
