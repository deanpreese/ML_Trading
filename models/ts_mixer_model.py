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
from tensorflow.keras.initializers import GlorotUniform, Ones
from tensorflow.keras.metrics import MeanSquaredError, BinaryCrossentropy, BinaryAccuracy, AUC  
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from ml_model.model_stats import gen_reg_stats_x 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)


class TSMixerModel:
    def __init__(self, epochs=50, batch_size=32, validation_split=0.2):
        
        tf.keras.backend.clear_session()
        
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
        
        self.hidden_dim = 32
        
        
        self.checkpoint_dir = 'checkpoints/'
        self.checkpoint_model = os.path.join(self.checkpoint_dir, 'ts_mixer_model.keras')
        self.checkpoint_scaler = os.path.join(self.checkpoint_dir, 'ts_mixer_scaler.pkl')
        self.saved_scaler = None

        self.trained_dir = 'trained_models/'
        self.trained_model = os.path.join(self.trained_dir, 'ts_mixer_model.keras')
        self.trained_scaler = os.path.join(self.trained_dir, 'ts_mixer_scaler.pkl')

    def build_model(self, input_shape):
        l2_reg = l2(0.01)
        initializer = GlorotUniform(seed=42)  # Consistent weight initialization

        num_features = self.X_train.shape[2]
        seq_length = self.X_train.shape[1]

        input_layer = tf.keras.layers.Input(shape=(seq_length, num_features))

        # Feature mixing
        x = tf.keras.layers.Dense(self.hidden_dim, activation='relu', kernel_regularizer=l2_reg, kernel_initializer=initializer)(input_layer)
        x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2_reg, kernel_initializer=initializer)(x)
        x = tf.keras.layers.Dense(num_features, activation='relu', kernel_regularizer=l2_reg, kernel_initializer=initializer)(x)

        # Time mixing
        x = tf.keras.layers.Permute((2, 1))(x)
        x = tf.keras.layers.Dense(self.hidden_dim, activation='relu', kernel_regularizer=l2_reg, kernel_initializer=initializer)(x)
        x = tf.keras.layers.Dense(seq_length, activation='relu', kernel_regularizer=l2_reg, kernel_initializer=initializer)(x)
        x = tf.keras.layers.Permute((2, 1))(x)

        # Flatten and output
        x = tf.keras.layers.Flatten()(x)
        output_layer = tf.keras.layers.Dense(1, kernel_initializer=initializer)(x)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
        model.summary()

    
        dot_img_file = os.path.join(self.checkpoint_dir, 'ts_mixer_plot.png')
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    
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
        joblib.dump(scaler, self.checkpoint_scaler )
                
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
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(self.checkpoint_model, save_best_only=True, monitor='val_loss', mode='min')
        
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=0,
            mode="auto",
            min_delta=0.0001,
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



    def train_model_k(self, file_path):
        data = pd.read_csv(file_path)
        
        # Assuming 'output' is the target and other columns are features
        X = data.drop(columns=['output', 'outputC'])  # Dropping outputC as per your context
        y = data['output'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.saved_scaler = scaler
        joblib.dump(scaler, self.checkpoint_scaler)
                
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # [batch_size, seq_length, num_features]
        
        # KFold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_index, val_index in kf.split(X_scaled):
            X_train, X_val = X_scaled[train_index], X_scaled[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            self.X_train = X_train
            self.X_val = X_val
            self.y_train = y_train
            self.y_val = y_val
            
            self.build_model(input_shape=X_train.shape[1:])
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model_checkpoint = ModelCheckpoint(self.checkpoint_model, save_best_only=True, monitor='val_loss', mode='min')
            
            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                verbose=0,
                mode="auto",
                min_delta=0.0001,
                cooldown=0,
                min_lr=0,
            )
            
            history = self.model.fit(
                X_train, y_train, 
                epochs=self.epochs, 
                batch_size=self.batch_size, 
                validation_data=(X_val, y_val),   
                callbacks=[early_stopping, reduce_lr, model_checkpoint]
            )
        
        # Using the last fold for evaluation, but ideally, you'd average metrics across folds
        self.X_test = X_val
        self.y_test = y_val
        y_pred = self.model.predict(X_val)
        
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


    def load_saved_model(self, mode):

        if mode == "run":
            self.model = tf.keras.models.load_model(self.trained_model)
            self.saved_scaler =  joblib.load( self.trained_scaler )
        
        if mode == "train":
           self.model = tf.keras.models.load_model(self.checkpoint_model)
           self.saved_scaler = joblib.load( self.checkpoint_scaler )
        


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