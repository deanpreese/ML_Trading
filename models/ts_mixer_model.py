import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib 
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Add, LSTM, Attention, Average, Reshape, Concatenate
from tensorflow.keras.optimizers import Adam
from keras.layers import LeakyReLU, Dropout, MultiHeadAttention
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import MeanSquaredError, BinaryCrossentropy, BinaryAccuracy, AUC  
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ml_model.model_stats import gen_reg_stats_x 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

tf.config.set_visible_devices([], 'GPU')

class TSMixerModel:
    def __init__(self, epochs=50, batch_size=32, validation_split=0.2):
        
        np.random.seed(42)
        tf.random.set_seed(42)
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        
        self.hidden_dim = 64
        
        self.checkpoint_dir = 'checkpoints/'
        self.saved_model_path = os.path.join(self.checkpoint_dir, 'ts_mixer_model.keras')
        self.saved_scaler_path = os.path.join(self.checkpoint_dir, 'ts_mixer_scaler.pkl')
        self.saved_scaler = None


    def build_model(self, input_shape):

        num_features = self.X_train.shape[2]
        seq_length = self.X_train.shape[1]

        input_layer = tf.keras.layers.Input(shape=(seq_length, num_features))

        # Feature mixing
        x = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(input_layer)
        x = tf.keras.layers.Dense(num_features, activation='relu')(x)

        # Time mixing
        x = tf.keras.layers.Permute((2, 1))(x)
        x = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(x)
        x = tf.keras.layers.Dense(seq_length, activation='relu')(x)
        x = tf.keras.layers.Permute((2, 1))(x)

        # Flatten and output
        x = tf.keras.layers.Flatten()(x)
        output_layer = tf.keras.layers.Dense(1)(x)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
        model.summary()

        self.model = model
        return self.model
        

    def train_model(self, file_path):
    
        data = pd.read_csv(file_path)
        
        # Assuming 'output' is the target and other columns are features
        X = data.drop(columns=['output', 'outputC'])  # Dropping outputC as per your context
        y = data['output'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.saved_scaler = scaler
        joblib.dump(scaler, self.saved_scaler_path)
                
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # [batch_size, seq_length, num_features]
        
        # Split into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        
        self.build_model("XYZ")
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(self.saved_model_path, save_best_only=True, monitor='val_loss', mode='min')
        
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=0,
            mode="auto",
            min_delta=0.000001,
            cooldown=0,
            min_lr=0,
        )
        
        history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_val, y_val),   
            callbacks=[early_stopping, 
                       #reduce_lr, 
                        model_checkpoint
                        ]
            )
        
        y_pred = self.model.predict(X_test)
        
        return history, y_pred


    def evaluate_model(self, y_pred):
        correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(self.y_test, y_pred)
        print(f"Test MSE: {mse}, Test MAE: {mae}, R2: {r2}")
        print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.2f}%")
        print(f"Number of Samples: {total}")
    

    def plot_training_history(self, history):
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('MAE over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()
        plt.show()


    def load_saved_model(self):
        self.model = tf.keras.models.load_model(self.saved_model_path)
        self.saved_scaler =  joblib.load( self.saved_scaler_path )


    def run_batch_test(self, file_path):
    
        df = pd.read_csv(file_path)
        df = df.drop(columns=['outputC'])
        X = df.drop(columns=['output'])
        y = df['output'].values 
               
        self.X_test = X
        X_scaled = self.saved_scaler.transform(X)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # [batch_size, seq_length, num_features]
        
        self.y_test = y
        y_pred = self.model.predict(X_scaled)
        
        self.evaluate_model(y_pred)