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
from tensorflow.keras.layers import Input, Conv1D, Average, Conv2D, LeakyReLU, Reshape, Concatenate, Multiply 
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
import ml_model.feature_filter as feature_filter


tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)


class KAN_13 (K_MODEL_BASE):
    def __init__(self, epochs=50, batch_size=32):
        
        model_name = 'kan_13'
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
        h = Bidirectional(LSTM(32,name="BIC", kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(h)
        h = Bidirectional(LSTM(32,name="BIC2", kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer))(h)

        h = Dense(32, activation='relu', kernel_regularizer=self.l2_reg,  kernel_initializer=self.initializer)(h)         
        attention_h = Dense(32, activation='softmax', kernel_initializer=self.initializer, name='attention_h')(h)
        h = Multiply()([h, attention_h])                
        h = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg, name="h_out", kernel_initializer=self.initializer)(h)           
        
        return h   


    def create_feature_model_x(self, inputs, output_dim):
        
        """
        X        
        Val MSE: 9.2086, Val MAE: 1.7184, R2: 0.4665146637449804
        Total Wins: 5651, Total Losses: 1800, Win Percentage: 0.758
        Number of Samples: 7451    
        """        
        
        x = Conv1D(filters=64, kernel_size=4, activation='relu', kernel_initializer=self.initializer)(inputs)       
        #x = LSTM(64, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(x)
        x = Conv1D(filters=32, kernel_size=3,  activation='relu', kernel_initializer=self.initializer)(x)
        #x = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(x)
        x = Conv1D(filters=16, kernel_size=2, activation='relu', kernel_initializer=self.initializer)(x)
        x = Bidirectional(LSTM(32, kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(x)
        x = Bidirectional(LSTM(64, kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(x)
        
        x = MaxPooling1D(pool_size=1, strides=1)(x)
        
        x = Bidirectional(LSTM(32, kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer))(x)
        x = Dense(32, activation='relu', kernel_regularizer=self.l2_reg,  kernel_initializer=self.initializer)(x) 
        attention_x = Dense(32, activation='softmax', kernel_initializer=self.initializer, name='attention_x')(x)
        x = Multiply()([x, attention_x])                
        x = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg, name="x_out", kernel_initializer=self.initializer)(x) 
                     
        return x

    def create_feature_model_c1(self, inputs, output_dim):
        
        inx = LSTM(32, return_sequences=True, activation='relu')(inputs)
        #inx = Dropout(self.drop_out)(inx)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
       # x = Dropout(self.drop_out)(x)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        #x = Dropout(self.drop_out)(x)
        x = MaxPooling1D(pool_size=1, strides=1)(x)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        #x = Dropout(self.drop_out)(x)
        c1_out = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg, name="c1_out", kernel_initializer=self.initializer)(x) 
        
        return c1_out


    def create_feature_model_c2(self, inputs, output_dim):
        
        x = LSTM(32, return_sequences=True, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        c2_out = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg, name="c2_out", kernel_initializer=self.initializer)(x) 

        return c2_out    

    def create_feature_model_a(self, input_shape):
        
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)
        inx = LSTM(32, return_sequences=True, activation='relu')(reshaped_inputs)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        smx_out = Dense(1, activation='linear')(x) 
        subx_model = Model(input, smx_out)
        return subx_model

    def create_feature_model_s(self, input_shape):
        
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)
        inx = LSTM(32, return_sequences=True, activation='relu')(reshaped_inputs)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        smx_out = Dense(1, activation='sigmoid')(x) 
        subx_model = Model(input, smx_out)
        return subx_model

    def create_feature_model_t(self, input_shape):
        
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)
        inx = LSTM(32, return_sequences=True, activation='relu')(reshaped_inputs)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        smx_out = Dense(1, activation='tanh')(x) 
        subx_model = Model(input, smx_out)
        return subx_model

   
    def create_model_a(self, input_shape ):
        
        inputs = Input(shape=input_shape)
        feature_outputs = []
        feature_outputs1 = []
        feature_outputs2 = []
        
        #concat_dims = 32
        concat_dims = input_shape[0]
        output_dim = 16
        
        model_a = self.create_feature_model_h(inputs, output_dim)
        model_b = self.create_feature_model_x(inputs, output_dim)
        model_c1 = self.create_feature_model_c1(inputs, output_dim)
        model_c2 = self.create_feature_model_c2(inputs, output_dim)
        
        for i in range(input_shape[0]):
            feature_input = inputs[:, i:i+1]

            #if ((i > 10) | (i==4)) : 
            if (i > -1) : 
                fa = self.create_feature_model_a((1,))
                fax = fa(feature_input)
                feature_outputs.append(fax)
                
                fm1 = self.create_feature_model_s((1,))
                fm1x = fm1(feature_input)
                feature_outputs1.append(fm1x)
                
                fm2 = self.create_feature_model_t((1,))
                fm2x = fm2(feature_input)
                feature_outputs2.append(fm2x)
       
        concatenated_outputs = Concatenate(axis=1)(feature_outputs)
        concatenated_outputs1 = Concatenate(axis=1)(feature_outputs1)
        concatenated_outputs2 = Concatenate(axis=1)(feature_outputs2)
        
        
        xa_out = Dense(concat_dims, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(concatenated_outputs)
        xs_out = Dense(concat_dims, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(concatenated_outputs1)
        xt_out = Dense(concat_dims, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(concatenated_outputs2)
        
        xa = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(xa_out)
        xs = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(xs_out)
        xt = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(xt_out)

        xa1 = Dense(output_dim, activation='softmax', kernel_initializer=self.initializer, name='attention_xa')(xa)
        x1 = Multiply()([xa1, xa])
        xs1 = Dense(output_dim, activation='softmax', kernel_initializer=self.initializer, name='attention_xs')(xs)
        x2 = Multiply()([xs1, xs])
        xt1 = Dense(output_dim, activation='softmax', kernel_initializer=self.initializer, name='attention_xt')(xt)
        x3 = Multiply()([xt1, xt])

        xc_ta = Multiply()([xt1, xa])
        xc_sa = Multiply()([xs1, xa])
        xc_ts = Multiply()([xt1, xs1])

        #s_ave_output = Average()([xa, xs, xt, x1,x2,x3])
        #s_ave_output = Average()([xa, xs, xt, x1,x2,x3])
        s_ave_output = Average(name='final_average')([xa, xs, xt, x1,x2,x3, xc_ta, xc_sa, xc_ts])
        x = s_ave_output
        
        full_concat = concatenated_outputs + concatenated_outputs1 + concatenated_outputs2
        fc = Dense(64, activation='relu')(full_concat)
        #fc = Dense(32, activation='relu')(fc)
        fc = Dense(output_dim, activation='relu')(fc)
        
        model_ave = Average()([model_a, model_b, model_c1, model_c2])
        
        x = 0.75 + s_ave_output + fc * 0.125 + 0.125 * model_ave
        
        #ave_output = Average()([model_a, model_b, s_ave_output])
        #ave_output = Average()([model_a, model_b, model_c1, model_c2, s_ave_output])
        #xa = Average()([model_a, model_b, model_c1, model_c2, ave_output])
        
        output = Dense(1, activation='linear')(x)
        
        self.model = Model(inputs=inputs, outputs=output)
        return self.model
    

    def create_model_b(self, input_shape):
        
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



    def create_model(self, input_shape ):
        
        inputs = Input(shape=input_shape)
        feature_outputs = []
        feature_outputs1 = []
        feature_outputs2 = []
        
        #concat_dims = 32
        concat_dims = input_shape[0]
        output_dim = 16
        
        model_a = self.create_feature_model_h(inputs, output_dim)
        model_b = self.create_feature_model_x(inputs, output_dim)
        model_c1 = self.create_feature_model_c1(inputs, output_dim)
        model_c2 = self.create_feature_model_c2(inputs, output_dim)
        
        for i in range(input_shape[0]):
            feature_input = inputs[:, i:i+1]

            #if ((i > 10) | (i==4)) : 
            if (i > -1) : 
                fa = self.create_feature_model_a((1,))
                fax = fa(feature_input)
                feature_outputs.append(fax)
                
                fm1 = self.create_feature_model_s((1,))
                fm1x = fm1(feature_input)
                feature_outputs1.append(fm1x)
                
                fm2 = self.create_feature_model_t((1,))
                fm2x = fm2(feature_input)
                feature_outputs2.append(fm2x)
       
        concatenated_outputs = Concatenate(axis=1)(feature_outputs)
        concatenated_outputs1 = Concatenate(axis=1)(feature_outputs1)
        concatenated_outputs2 = Concatenate(axis=1)(feature_outputs2)
        
        xa_out = Dense(concat_dims, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(concatenated_outputs)
        xs_out = Dense(concat_dims, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(concatenated_outputs1)
        xt_out = Dense(concat_dims, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(concatenated_outputs2)
        
        xa = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(xa_out)
        xs = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(xs_out)
        xt = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(xt_out)

        xa1 = Dense(output_dim, activation='softmax', kernel_initializer=self.initializer, name='attention_xa')(xa)
        x1 = Multiply()([xa1, xa])
        xs1 = Dense(output_dim, activation='softmax', kernel_initializer=self.initializer, name='attention_xs')(xs)
        x2 = Multiply()([xs1, xs])
        xt1 = Dense(output_dim, activation='softmax', kernel_initializer=self.initializer, name='attention_xt')(xt)
        x3 = Multiply()([xt1, xt])

        xc_ta = Multiply()([xt1, xa])
        xc_sa = Multiply()([xs1, xa])
        xc_ts = Multiply()([xt1, xs1])

        #s_ave_output = Average()([xa, xs, xt, x1,x2,x3])
        #s_ave_output = Average()([xa, xs, xt, x1,x2,x3])
        s_ave_output = Average(name='final_average')([xa, xs, xt, x1,x2,x3, xc_ta, xc_sa, xc_ts])
        x = s_ave_output
        
        #ave_output = Average()([model_a, model_b, s_ave_output])
        #ave_output = Average()([model_a, model_b, model_c1, model_c2, s_ave_output])
        #xa = Average()([model_a, model_b, model_c1, model_c2, ave_output])
        
        output = Dense(1, activation='linear')(x)
        
        self.model = Model(inputs=inputs, outputs=output)
        return self.model
    


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
    
    model = KAN_13()
    X_train, X_val, X_test, y_train, y_val, y_test,  X_oos, y_oos, input_shape = model.process_data_split(datafile[1], datafile[0], col_filter)
    
    history_out, y_pred = model.train_model(input_shape, X_train, X_test, y_train, y_test, X_val, y_val, 10 )
    best_model = tf.keras.models.load_model(model.checkpoint_model)
    model.evaluate_finished_model(best_model, X_val, X_test, y_train, y_val, y_test,  X_oos, y_oos)
    
    
            
if __name__ == "__main__":
    main()