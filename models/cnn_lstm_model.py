import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib 

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout, MaxPooling1D, LSTM, Attention, Bidirectional, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ml_model.model_stats import gen_reg_stats_x 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from ml_model.model_stats import gen_reg_stats_x 
tf.config.set_visible_devices([], 'GPU')


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
        
        self.checkpoint_dir = 'checkpoints/'
        self.saved_cnn_lstm = os.path.join(self.checkpoint_dir, 'cnn_lstm_model.keras')
        
    def build_model_o(self, input_shape):
        l2_reg = l2(0.01)
        inputs = Input(shape=input_shape)
        x = Conv1D(filters=64, kernel_size=2, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        x = Conv1D(filters=64, kernel_size=2, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = MultiHeadAttention(num_heads=input_shape[0]//2, key_dim=input_shape[0]//2, kernel_regularizer=l2_reg)(x, x)
        x = Dropout(0.2)(x)    
        x = LSTM(32, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(50)(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)  # Output layer with 1 neuron for regression
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        model.summary()
        self.model = model
        return model
            

    def build_model(self, input_shape):
        l2_reg = l2(0.01)
        inputs = Input(shape=input_shape)
        x = Conv1D(filters=64, kernel_size=4, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)

        x = Conv1D(filters=64, kernel_size=4, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=4)(x)
        x = Dropout(0.2)(x)
        
        x = Conv1D(filters=64, kernel_size=4, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = MultiHeadAttention(num_heads=input_shape[0]//2, key_dim=input_shape[0]//2, kernel_regularizer=l2_reg)(x, x)
        x = Dropout(0.2)(x)    
        x = LSTM(32, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(50)(x)
        x = Dropout(0.2)(x)
        
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
               
        
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)  # Output layer with 1 neuron for regression
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        model.summary()
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
        self.build_model(input_shape)
    
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(self.saved_cnn_lstm, save_best_only=True)
        history_out = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stopping,model_checkpoint])
        y_pred = self.model.predict(X_test)

        return history_out, y_pred


    def evaluate_model(self, y_pred):
        correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(self.y_test, y_pred)
        print(f"Test MSE: {mse}, Test MAE: {mae}, R2: {r2}")
        print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.2f}%")
        print(f"Number of Samples: {total}")
    
    
    def load_saved_model(self):
        self.model = tf.keras.models.load_model(self.saved_cnn_lstm)

    def run_batch_test(self, file_path):
    
        df = pd.read_csv(file_path)
        df = df.drop(columns=['outputC'])
        X = df.drop(columns=['output']).values
        y = df['output'].values 
               
        self.X_test = X.values
        self.y_test = y
        y_pred = self.model.predict(self.X_test)
        
        
        self.evaluate_model(y_pred)
        