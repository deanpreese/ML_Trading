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


    def create_feature_model_a(self, inputs, output_dim):
        
        """ 
        Key classification metrics on OOS:
        Accuracy:  0.7996
        AUC:       0.8267
        Precision: 0.8157
        Recall:    0.8165
        F1-score:  0.8161

        Confusion Matrix (OOS, threshold=0.5):
                    Pred 0     Pred 1
        Actual 0       3228       914
        Actual 1        909      4045

        Detailed classification report (OOS):
                    precision    recall  f1-score   support

                0     0.7803    0.7793    0.7798      4142
                1     0.8157    0.8165    0.8161      4954

            accuracy                         0.7996      9096
        macro avg     0.7980    0.7979    0.7980      9096
        weighted avg     0.7996    0.7996    0.7996      9096
        """ 

        x = Dense(256, activation='relu', kernel_initializer=self.initializer)(inputs)
        x = Dense(128, activation='relu', kernel_initializer=self.initializer)(x)
        x = Dense(64, activation='relu', kernel_initializer=self.initializer)(x)
        x = Dense(32, activation='relu', kernel_initializer=self.initializer)(x)
        x = Flatten()(x)
        x = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg, name="a_out", kernel_initializer=self.initializer)(x)           
       
        return x

    def create_feature_model_b(self, inputs, output_dim):
        
        """ 
        Key classification metrics on OOS:
        Accuracy:  0.7989
        AUC:       0.8288
        Precision: 0.8153
        Recall:    0.8155
        F1-score:  0.8154

        Confusion Matrix (OOS, threshold=0.5):
                    Pred 0     Pred 1
        Actual 0       3227       915
        Actual 1        914      4040

        Detailed classification report (OOS):
                    precision    recall  f1-score   support

                0     0.7793    0.7791    0.7792      4142
                1     0.8153    0.8155    0.8154      4954

            accuracy                         0.7989      9096
        macro avg     0.7973    0.7973    0.7973      9096
        weighted avg     0.7989    0.7989    0.7989      9096
        """ 
        c = Conv1D(filters=64, kernel_size=3, activation='relu', kernel_initializer=self.initializer)(inputs)
        c = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(c)
        
        c = Conv1D(filters=32, kernel_size=2, activation='relu', kernel_initializer=self.initializer)(c)
        c = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(c)
        
        c = Conv1D(filters=16, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(c)
        c = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(c)
        
        c = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', kernel_initializer=self.initializer)(c)
        h = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg, name="b_out", kernel_initializer=self.initializer)(c)           
        
        return h


    def create_feature_model_h(self, inputs, output_dim):
        
        """ 
        Key classification metrics on OOS:
        Accuracy:  0.7996
        AUC:       0.8219
        Precision: 0.8157
        Recall:    0.8165
        F1-score:  0.8161

        Confusion Matrix (OOS, threshold=0.5):
                    Pred 0     Pred 1
        Actual 0       3228       914
        Actual 1        909      4045

        Detailed classification report (OOS):
                    precision    recall  f1-score   support

                0     0.7803    0.7793    0.7798      4142
                1     0.8157    0.8165    0.8161      4954

            accuracy                         0.7996      9096
        macro avg     0.7980    0.7979    0.7980      9096
        weighted avg     0.7996    0.7996    0.7996      9096
        
        """        
        x = Dense(256, activation='relu', kernel_initializer=self.initializer)(inputs)
        x = Dense(128, activation='relu', kernel_initializer=self.initializer)(x)
        x = Dense(64, activation='relu', kernel_initializer=self.initializer)(x)
        #x = Reshape((9, 64))(x)
        
        h = Conv1D(filters=64, kernel_size=3, activation='relu', kernel_initializer=self.initializer)(x)
        h = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(h)
        h = Conv1D(filters=32, kernel_size=2, activation='relu', kernel_initializer=self.initializer)(h)
        h = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(h)
        h = Conv1D(filters=16, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(h)
        h = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', kernel_initializer=self.initializer)(h)

        h = Dense(16, activation='relu', kernel_regularizer=self.l2_reg,  kernel_initializer=self.initializer)(h)         
        h = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg, name="h_out", kernel_initializer=self.initializer)(h)           
       
        return h

    def create_feature_model_h2(self, inputs, output_dim):

        """         
        Key classification metrics on OOS:
        Accuracy:  0.7996
        AUC:       0.8127
        Precision: 0.8163
        Recall:    0.8155
        F1-score:  0.8159

        Confusion Matrix (OOS, threshold=0.5):
                    Pred 0     Pred 1
        Actual 0       3233       909
        Actual 1        914      4040

        Detailed classification report (OOS):
                    precision    recall  f1-score   support

                0     0.7796    0.7805    0.7801      4142
                1     0.8163    0.8155    0.8159      4954

            accuracy                         0.7996      9096
        macro avg     0.7980    0.7980    0.7980      9096
        weighted avg     0.7996    0.7996    0.7996      9096
        
        """        
        
        x = Dense(256, activation='relu', kernel_initializer=self.initializer)(inputs)
        x = Dense(128, activation='relu', kernel_initializer=self.initializer)(x)
        x = Dense(64, activation='relu', kernel_initializer=self.initializer)(x)
        
        x = LayerNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Conv1D(filters=64, kernel_size=3, activation='relu', kernel_initializer=self.initializer)(x)
        #x = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(x)
        x = Bidirectional(LSTM(32, kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(x)
        
        x = LayerNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Conv1D(filters=32, kernel_size=2, activation='relu', kernel_initializer=self.initializer)(x)
        #x = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True,kernel_initializer=self.initializer)(x)
        x = Bidirectional(LSTM(32, kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(x)
        #x = LSTM(64, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True,kernel_initializer=self.initializer)(x)
                
        x = LayerNormalization()(x)
        x = Dropout(0.5)(x)        
                
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        #x = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', kernel_initializer=self.initializer)(x)
        x = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(x)
        #x = Bidirectional(LSTM(32, kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(x)
        
        x = LayerNormalization()(x)
        x = Dropout(0.5)(x)
        x = GlobalAveragePooling1D()(x)
        
        #x = Dense(128, activation='relu', kernel_initializer=self.initializer)(x)
        #x = Dense(64, activation='relu', kernel_initializer=self.initializer)(x)
        #x = Dense(32, activation='relu', kernel_initializer=self.initializer)(x)
        x = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg, name="h2_out", kernel_initializer=self.initializer)(x)           
       
        return x



    def create_feature_model_rx(self, input_shape, output_dim):
        
        """
        Key classification metrics on OOS:
        Accuracy:  0.7996
        AUC:       0.8288
        Precision: 0.8156
        Recall:    0.8167
        F1-score:  0.8161

        Confusion Matrix (OOS, threshold=0.5):
                    Pred 0     Pred 1
        Actual 0       3227       915
        Actual 1        908      4046

        Detailed classification report (OOS):
                    precision    recall  f1-score   support

                0     0.7804    0.7791    0.7798      4142
                1     0.8156    0.8167    0.8161      4954

            accuracy                         0.7996      9096
        macro avg     0.7980    0.7979    0.7979      9096
        weighted avg     0.7996    0.7996    0.7996      9096
        """
        
        
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)
        x = LSTM(64, return_sequences=True, activation='relu')(reshaped_inputs)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        smx_out = Dense(output_dim, activation='linear')(x) 
        subx_model = Model(input, smx_out)
        return subx_model


    def create_feature_model_rx2(self, input_shape, output_dim):
        
        """
        Key classification metrics on OOS:
        Accuracy:  0.7996
        AUC:       0.8268
        Precision: 0.8156
        Recall:    0.8167
        F1-score:  0.8161

        Confusion Matrix (OOS, threshold=0.5):
                    Pred 0     Pred 1
        Actual 0       3227       915
        Actual 1        908      4046

        Detailed classification report (OOS):
                    precision    recall  f1-score   support

                0     0.7804    0.7791    0.7798      4142
                1     0.8156    0.8167    0.8161      4954

            accuracy                         0.7996      9096
        macro avg     0.7980    0.7979    0.7979      9096
        weighted avg     0.7996    0.7996    0.7996      9096         
                
        """
        
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)        
        
        x_factor=2
        
        x = Dense(128 * x_factor, activation='relu', kernel_initializer=self.initializer)(reshaped_inputs)
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

    def create_feature_model_rx3(self, input_shape, output_dim):
               
        """
        Key classification metrics on OOS:
        Accuracy:  0.7996
        AUC:       0.8283
        Precision: 0.8156
        Recall:    0.8167
        F1-score:  0.8161

        Confusion Matrix (OOS, threshold=0.5):
                    Pred 0     Pred 1
        Actual 0       3227       915
        Actual 1        908      4046

        Detailed classification report (OOS):
                    precision    recall  f1-score   support

                0     0.7804    0.7791    0.7798      4142
                1     0.8156    0.8167    0.8161      4954

            accuracy                         0.7996      9096
        macro avg     0.7980    0.7979    0.7979      9096
        weighted avg     0.7996    0.7996    0.7996      9096
        """
        
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)        
        
        x = Dense(128, activation='gelu', kernel_initializer=self.initializer)(reshaped_inputs)
        x = Dense(64, activation='gelu', kernel_initializer=self.initializer)(x)
        x = Dense(32, activation='gelu', kernel_initializer=self.initializer)(x)
        x = Reshape((8, 4))(x)
        
        x = LSTM(64, return_sequences=True, activation='relu')(x)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
                
        smx_out = Dense(output_dim, activation='linear')(x) 
        subx_model = Model(input, smx_out)
        return subx_model
   
   

    def create_feature_model_z(self, inputs, output_dim, ):
        
        """
        Key classification metrics on OOS:
        Accuracy:  0.7996
        AUC:       0.8246
        Precision: 0.8156
        Recall:    0.8167
        F1-score:  0.8161

        Confusion Matrix (OOS, threshold=0.5):
                    Pred 0     Pred 1
        Actual 0       3227       915
        Actual 1        908      4046

        Detailed classification report (OOS):
                    precision    recall  f1-score   support

                0     0.7804    0.7791    0.7798      4142
                1     0.8156    0.8167    0.8161      4954

            accuracy                         0.7996      9096
        macro avg     0.7980    0.7979    0.7979      9096
        weighted avg     0.7996    0.7996    0.7996      9096
        
        """
        
        # ----- Stem (feature projection, keeps sequence shape) -----
        initializer = self.initializer
        dropout=0.2
        l2_reg =self.l2_reg 
        
        x = Dense(128, activation='relu', kernel_initializer=initializer)(inputs)
        x = LayerNormalization()(x)
        x = Dropout(dropout)(x)

        # ----- Conv block (local temporal patterns) -----
        c = Conv1D(64, kernel_size=3, padding="same", activation='relu',kernel_initializer=initializer)(x)
        c = LayerNormalization()(c)
        c = Dropout(dropout)(c)

        # Residual connection (project if needed)
        if x.shape[-1] != c.shape[-1]:
            x_proj = Dense(int(c.shape[-1]), kernel_initializer=initializer)(x)
        else:
            x_proj = x
        x = Add()([x_proj, c])

        # ----- Recurrent block (long-range dependencies) -----
        x = Bidirectional(LSTM(64,return_sequences=True,kernel_regularizer=l2_reg,recurrent_dropout=0.0,  kernel_initializer=initializer))(x)
        x = LayerNormalization()(x)
        x = Dropout(dropout)(x)

        x = LSTM(64,return_sequences=True, kernel_regularizer=l2_reg,kernel_initializer=initializer)(x)
        x = LayerNormalization()(x)
        x = Dropout(dropout)(x)

        # ----- Head -----
        x = GlobalAveragePooling1D()(x)  # stable pooling instead of last timestep dependence
        x = Dense(128, activation='relu', kernel_initializer=initializer)(x)
        x = Dropout(dropout)(x)
        x = Dense(64, activation='relu', kernel_initializer=initializer)(x)

        out = Dense(output_dim, activation='linear', kernel_regularizer=l2_reg,kernel_initializer=initializer)(x)
        return out

   
   
   
   
    def create_model(self, input_shape ):
        
        inputs = Input(shape=input_shape)
        output_dim = 16

        #inputs = FFTLayer()(inputs)
        #inputs = FFTOrRFTLayer(use_rft=True, return_magnitude=True, name="fft_or_rft_layer")(inputs)
        
        model_a = self.create_feature_model_a(inputs, output_dim)
        model_b = self.create_feature_model_b(inputs, output_dim)
        model_h = self.create_feature_model_h(inputs, output_dim)
        model_h2 = self.create_feature_model_h2(inputs, output_dim)        
        model_z = self.create_feature_model_z(inputs, output_dim)        
        
        feature_outputs = []
      
        for i in range(input_shape[0]):
            feature_input = inputs[:, i:i+1]
            
            rx = self.create_feature_model_rx((1,),output_dim)
            rx_out = rx(feature_input)
            
            rx2 = self.create_feature_model_rx2((1,),output_dim)
            rx2_out = rx2(feature_input)

            rx3 = self.create_feature_model_rx3((1,),output_dim)
            rx3_out = rx3(feature_input)

            # xxx out
            x = rx3_out
            
            feature_outputs.append(x)   
             
                   
        x = Concatenate(axis=1)(feature_outputs)
        #x = Dense(16, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(x)
        x = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(x)
        
        # b like h2.  a like h
        
        #x = Average()([model_a, model_b, x])
        x = model_b       
        
        
        output = Dense(1, activation='sigmoid')(x)
        self.model = Model(inputs=inputs, outputs=output)
        return self.model
    

def main():
    
    datafile = [ 
                'data/Lucky13_3070_oos.csv',   
                'data/Lucky13_3070.csv',  #1
                'data/Model_X_3070_oos.csv',  
                'data/Model_X_3070.csv',  #3
                'data/The_13_X_3070_oos.csv',
                'data/The_13_X_3070.csv', #5,
                'data/Model_3LB_ALL_oos.csv', #6
                'data/Model_3LB_ALL.csv' #7
                
        ] 
    
    file_train = datafile[7]
    file_oos = datafile[6]
    
    col_filter =["Year","Month","Day","DayOfWeek","HourOfDay","MinOfHour","SeqClose","RSIRAW",
     "SDLR310","SDBB91","SDKC91","SDKC9","ROC",
     "ATR54","ATR53","ATR52","ATR51","ATR5","ATR21","ATR2","RSI","STOK1",
     "dv0","dv1","dv2","dv3","dv4","dv5","dv6","dv7","dv8","dv9","dv10","dv11","dv12","dv13","dv14","dv15","dv16","dv17","dv18","dv19","dv20","dv21","dv22","dv23",
     "cv0","cv1","cv2","cv3","cv4","cv5","cv6","cv7","cv8","cv9","cv10","cv11","cv12","cv13","cv14","cv15","cv16","cv17","cv18","cv19","cv20","cv21","cv22","cv23",
     "chv0","chv1","chv2","chv3","chv4","chv5","chv6","chv7","chv8","chv9","chv10","chv11","chv12","chv13","chv14","chv15","chv16","chv17","chv18","chv19","chv20",
     "chv21","chv22","chv23"]

    
    
    #col_filter = ['SDKC9', 'ATR5', 'ROC', 'ATR2', 'SDBB91', 'ATR21', 'RSI',       
    #                  'cv2', 'cv1', 'cv0', 'chv2', 'chv0', 'cv4', 'chv12', 'chv18', 'chv11']
  
    #col_filter = ['cv2', 'cv1', 'cv0', 'chv2', 'chv0', 'cv4', 'chv12', 'chv18', 'chv11']
    
    model = K2('k2_3lb_x')
    
    X_train, X_val, X_test, y_train, y_val, y_test,  X_oos, y_oos, input_shape = model.process_data_split(file_train, file_oos, col_filter)
    
    history_out, y_pred = model.train_model(input_shape, X_train, X_test, y_train, y_test, X_val, y_val, 10 )
    best_model = tf.keras.models.load_model(model.checkpoint_model)
    #best_model = tf.keras.models.load_model(model.checkpoint_model, custom_objects={"FFTLayer": FFTLayer, "FFTOrRFTLayer": FFTOrRFTLayer})
    
    model.evaluate_finished_model_x(best_model, X_val, X_test, y_train, y_val, y_test,  X_oos, y_oos)

            
if __name__ == "__main__":
    main()
    
