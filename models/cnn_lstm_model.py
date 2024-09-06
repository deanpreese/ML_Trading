import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib 

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, SeparableConv1D,  Bidirectional, Dense,  Dropout, MaxPooling1D, LSTM, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ml_model.model_stats import gen_reg_stats_x 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.initializers import GlorotUniform

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from ml_model.model_stats import gen_reg_stats_x 
tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)
        

class CNN_LSTM:
    def __init__(self, epochs=50, batch_size=32):
        
        np.random.seed(42)
        tf.random.set_seed(42)
        
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.drop_out = 0.2
        self.l2_reg = l2(0.01)
        self.initializer = GlorotUniform(seed=42)  
        
        self.checkpoint_dir = 'checkpoints/'
        self.checkpoint_model = os.path.join(self.checkpoint_dir, 'cnn_lstm_model.keras')
        
        self.trained_dir = 'trained_models/'
        self.trained_model = os.path.join(self.trained_dir, 'cnn_lstm_model.keras')
        
    def build_model_t(self, input_shape):

        """
        Val MSE: 9.4813, Val MAE: 1.7181, R2: 0.45071796821989885
        Total Wins: 5653, Total Losses: 1798, Win Percentage: 0.759
        Number of Samples: 7451
        
        Val MSE: 9.5755, Val MAE: 1.7195, R2: 0.44525831241871294
        Total Wins: 5643, Total Losses: 1808, Win Percentage: 0.757
        Number of Samples: 7451
        
        seq_dims = 16 
        Val MSE: 9.5292, Val MAE: 1.7217, R2: 0.4479422346778489
        Total Wins: 5644, Total Losses: 1807, Win Percentage: 0.757
        Number of Samples: 7451
        """ 
        inputs = Input(shape=input_shape)
        
        output_units = 16
        bid_lstm_units = 64
        tft_lstm_1 = 32
        
        filters = 64
        kernel = 4
        
        hidden_dims = 64 
        seq_dims = 32 
        
        
        c = Conv1D(name="C", filters=filters, kernel_size=kernel, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(inputs)
        c = Conv1D(filters=filters//2, kernel_size=kernel-1, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(c)
        c = Conv1D(filters=filters, kernel_size=kernel-2, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(c)
        c = MaxPooling1D(pool_size=kernel, strides=2)(c)
        c = Dropout(self.drop_out)(c)

        lstm_out = Bidirectional(LSTM(bid_lstm_units, name="BICA", return_sequences=True, kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer))(c)
        lstm_out = Dropout(self.drop_out)(lstm_out)
        lstm_out = LSTM(tft_lstm_1, kernel_regularizer=self.l2_reg, name="LS1", return_sequences=True, recurrent_regularizer=self.l2_reg)(lstm_out)
        lstm_out = Dropout(self.drop_out)(lstm_out)

        x = tf.keras.layers.Permute((2, 1))(lstm_out)
        x = tf.keras.layers.Dense(hidden_dims, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(x)
        x = tf.keras.layers.Dense(seq_dims, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(x)
        x = tf.keras.layers.Permute((2, 1))(x)
        dense = tf.keras.layers.Flatten()(x)
        
        dense = Dense(output_units, activation='relu', kernel_regularizer=self.l2_reg)(dense)
        dense = Dropout(0.3)(dense)
        outputs = Dense(1)(dense)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', tf.keras.metrics.R2Score()])
        model.summary()
                
        dot_img_file = os.path.join(self.checkpoint_dir, 'dcnn_t_plot.png')
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=True)
        self.checkpoint_model = os.path.join(self.checkpoint_dir, 'dcnn_model_t.keras')
        self.trained_model = os.path.join(self.trained_dir, 'dcnn_model_t.keras')
        print(" ")
        self.model = model
        return model
            

    def build_model_x(self, input_shape):
        l2_reg = l2(0.02)
        inputs = Input(shape=input_shape)

        x = SeparableConv1D(filters=64, kernel_size=2, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        z = Dense(128, activation='relu')(inputs)
        z = Conv1D(filters=64, kernel_size=4, activation='relu')(z)
        z = MaxPooling1D(pool_size=2)(z)
        z = Dropout(0.2)(z)
        
        x = MultiHeadAttention(num_heads=input_shape[0]//2, key_dim=input_shape[0]//2, kernel_regularizer=l2_reg)(x, z)
        x = Dropout(0.2)(x)    
        
        x = LSTM(32)(x)
        x = Dropout(0.2)(x)
        
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)  # Output layer with 1 neuron for regression
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        model.summary()
        
            
        dot_img_file = os.path.join(self.checkpoint_dir, 'cnn_lstm_plot.png')
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    
        
        print(" ")
        print(" ----- ")
        print(" ")
        self.model = model
        return model
    
    

    def build_model(self, input_shape):
        """
        Val MSE: 9.3559, Val MAE: 1.7374, R2: 0.4579851446826114
        Total Wins: 5652, Total Losses: 1799, Win Percentage: 0.759
        Number of Samples: 7451
       
        """
        l2_reg = l2(0.02)
        drop_out = 0.2
        output_units = 16
        lay1_lstm_units = 32
        inputs = Input(shape=input_shape)
        
        initializer = GlorotUniform(seed=42)  

        c = Conv1D(name="C", filters=64, kernel_size=2, activation='relu',kernel_regularizer=l2_reg, kernel_initializer=initializer)(inputs)
        c = Conv1D(filters=32, kernel_size=2, activation='relu',kernel_regularizer=l2_reg, kernel_initializer=initializer)(c)
        c = Conv1D(filters=64, kernel_size=2, activation='relu',kernel_regularizer=l2_reg, kernel_initializer=initializer)(c)
        c = MaxPooling1D(pool_size=2, strides=2)(c)
        c = Dropout(drop_out)(c)
        
        c = Bidirectional(LSTM(lay1_lstm_units,name="BIC", kernel_regularizer=l2_reg, kernel_initializer=initializer))(c)
        c = Dropout(drop_out)(c)
        c = Dense(name="C_out", units=output_units, activation='relu',kernel_regularizer=l2_reg, kernel_initializer=initializer)(c)
        
        hidden_dim = 16
        seq_length = 8 
        x = tf.keras.layers.Dense(hidden_dim, activation='relu', kernel_regularizer=l2_reg, kernel_initializer=initializer)(c)
        x = tf.keras.layers.Dense(seq_length, activation='relu', kernel_regularizer=l2_reg, kernel_initializer=initializer)(x)
        x = tf.keras.layers.Flatten()(x)

        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', tf.keras.metrics.R2Score()])
        model.summary()
            
            
        dot_img_file = os.path.join(self.checkpoint_dir, 'dcnn_plot.png')
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=True)
        self.checkpoint_model = os.path.join(self.checkpoint_dir, 'dcnn_model.keras')
        self.trained_model = os.path.join(self.trained_dir, 'dcnn_model.keras')
        print(" ")

        self.model = model

        return model
    
    def train_model(self, file_path):
    
        df = pd.read_csv(file_path)
        df = df.drop(columns=['outputC'])
        X = df.drop(columns=['output']).values
        y = df['output'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
        input_shape=(X_train.shape[1], 1)
        self.build_model_t(input_shape)
    
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(self.checkpoint_model, save_best_only=True)
        history_out = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stopping,model_checkpoint])
        y_pred = self.model.predict(X_test)

        return history_out, y_pred


    def evaluate_model(self, y_pred):
        correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(self.y_test, y_pred)
        print(f"Test MSE: {mse}, Test MAE: {mae}, R2: {r2}")
        print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.3f}")
        print(f"Number of Samples: {total}")
    
    
    def load_saved_model(self, mode):
        
        if mode == "run":
           self.model = tf.keras.models.load_model(self.trained_model)
        
        if mode == "train":
           self.model = tf.keras.models.load_model(self.checkpoint_model)
           

    def run_batch_test(self, file_path):
    
        df = pd.read_csv(file_path)
        df = df.drop(columns=['outputC'])
        X = df.drop(columns=['output']).values
        y = df['output'].values 
               
        self.X_test = X
        self.y_test = y
        y_pred = self.model.predict(self.X_test)
        
        
        self.evaluate_model(y_pred)
        