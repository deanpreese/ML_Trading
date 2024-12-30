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

from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error,r2_score, root_mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score



tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)


class FEAT_KAN:
    def __init__(self, epochs=50, batch_size=32):
        
        self.checkpoint_dir = 'checkpoints/'
        self.trained_dir = 'trained_models/'
       
        self.checkpoint_model = os.path.join(self.checkpoint_dir, 'f_kan_z_model.keras')
        self.trained_model = os.path.join(self.trained_dir, 'f_kan_z_model.keras')
        self.model_plot = os.path.join(self.checkpoint_dir, 'f_kan_z_model.png')

        self.drop_out = 0.2
        self.l2_reg = l2(0.01)
        self.initializer = GlorotUniform(seed=42)



    def create_feature_model_a(self, input_shape):
        
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)
        inx = LSTM(32, return_sequences=True, activation='relu')(reshaped_inputs)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
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

   
    def create_model(self, input_shape ):
        
        inputs = Input(shape=input_shape)
        feature_outputs = []
        feature_outputs1 = []
        feature_outputs2 = []
        
        concat_dims = 64
        output_dim = 16
        
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
        
        xa = Dense(concat_dims, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(concatenated_outputs)
        xa = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(xa)
        xs = Dense(concat_dims, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(concatenated_outputs1)
        xs = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(xs)
        xt = Dense(concat_dims, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(concatenated_outputs2)
        xt = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(xt)

        s_ave_output = Average()([xa, xs, xt])
        ave_output = s_ave_output
        output = Dense(1, activation='linear')(ave_output)
        self.model = Model(inputs=inputs, outputs=output)
        return self.model
    

    def train_model(self, input_shape, X_train, X_test, y_train, y_test,  X_val, y_val, epocs ):
        
        self.create_model(input_shape )

        self.model.compile(optimizer=Adam(learning_rate=0.001), 
                loss='mse', metrics=['mae', tf.keras.metrics.R2Score()])
        self.model.summary()
        
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, verbose=1,
            mode="auto", min_delta=0.000001, cooldown=0, min_lr=0,
            )

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_model, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min'
            )
        
        history_out = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), 
            initial_epoch=0, epochs=epocs, verbose=1, batch_size=64, 
            callbacks=[early_stopping, reduce_lr, model_checkpoint]
            )      
        
        y_pred = self.model.predict(X_test)
        return history_out, y_pred



def evaluate_model( y_test, y_pred):
    
    correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(y_test, y_pred)
    print(" ")
    print(f"Pred MSE: {mse},  MAE: {mae}, R2: {r2}")
    print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.4f}")
    print(f"Number of Samples: {total}")        
    print(" ")        
    
    return rmse, mse, mae, r2
    


def main():
    
    datafile = [ 
                'data/Lucky13_3070_oos.csv',   
                'data/Lucky13_3070.csv',  #1
                'data/Model_X_3070_oos.csv',  
                'data/Model_X_3070.csv',  #3
                'data/The_13_X_3070_oos.csv',
                'data/The_13_X_3070.csv' #5
                ]   

    file_train = 3


    np.random.seed(42)
    tf.random.set_seed(42)

    file_path = datafile[file_train]

    lucky_13_columns = [
        "SDLR310", "SDBB91", "SDKC91", "SDKC9", "ROC", "ATR54", "ATR53", "ATR52", 
        "ATR51", "ATR5", "ATR21", "ATR2", "RSI", "STOK1", "output", "outputC"
    ]
    
    model_x_columns = [
        #"Year", "Month", "Day", "DayOfWeek", "HourOfDay", "MinOfHour", "SeqClose",
        "SDBB9", "SDBB91", "SDKC9", "SDKC91", "SDBB29", "SDBB291", "SDKC29", "SDKC291",
        "SDBB14CU", "SDBB14CL", "SDBB9CU", "SDBB9CL", "SDKC10CU", "SDKC10CL", "SDKC7CU",
        "SDKC7CL", "ROC14", "ROC9", "ROC7", "ATR14", "ATR9", "ATR5", "ATR2", "RSI14", 
        "RSI9", "ADX14", "ADX9", "STO5135K", "STO5135D", "STO7143K", "STO7143D", "TV1", 
        "TV2", "TV3", "TV4", "TV5", "TV6", "ZH79X", "ZL79X", "ZC79X", "COMP0", "COMP1", 
        "COMP2", "COMP3"
    ]
    


    column_results = []
    #column_list = data_13_x
    column_list = model_x_columns
    
    for i in range(len(column_list)):
        
        print(f"Loading {file_path}" )
        data = pd.read_csv(file_path)
        df = data.drop(columns=['SeqClose', 'outputC'])
        X = df.drop(columns=['output'])
        y = df['output']
        
        print(column_list[i])
        X = X[[column_list[i]]]
    
        X_train, X_val, X_test, y_train, y_val, y_test = split_three_ways(X, y)

        input_shape = (X_train.shape[1], 1)
        #(14, 1)

        print(X_train.shape)
        print(X_test.shape)
        print(X_val.shape)
        
        epocs = 10
    
        c_kan = FEAT_KAN()
        history_out, y_pred = c_kan.train_model(input_shape, X_train, X_test, y_train, y_test, X_val, y_val, epocs )

        # Load best model and evaluate
        best_model = tf.keras.models.load_model(c_kan.checkpoint_model)
        y_pred = best_model.predict(X_test)
        
        rmse = root_mean_squared_error(y_test, y_pred)
        mse = rmse **2.0
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results_out = f"{column_list[i]}   {rmse}   {mse}  {mae}  {r2}"
        print(results_out)
        
        column_results.append(results_out)


    with open('column_values.txt', 'a') as f:
        for param in column_results:
            f.write(f"{param}\n")
    
if __name__ == "__main__":
    main()
    
    