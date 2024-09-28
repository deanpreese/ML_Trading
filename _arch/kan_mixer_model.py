import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Add, LSTM, Attention, Average, Reshape, Concatenate, Conv1D, MaxPooling1D 
from tensorflow.keras.optimizers import Adam
from keras.layers import LeakyReLU, Dropout, MultiHeadAttention
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.metrics import MeanSquaredError, BinaryCrossentropy, BinaryAccuracy, AUC  
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from ml_model.model_stats import gen_reg_stats_x 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
tf.random.set_seed(42)
tf.config.set_visible_devices([], 'GPU')

class KANMixerModel:
    def __init__(self, epochs=50, batch_size=32, validation_split=0.2):
        
        np.random.seed(42)
        tf.random.set_seed(42)
        tf.keras.backend.clear_session()
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        
        self.checkpoint_dir = 'checkpoints/'
        self.checkpoint_model = os.path.join(self.checkpoint_dir, 'kan_ts_model.keras')
        
        self.trained_dir = 'trained_models/'
        self.trained_model = os.path.join(self.trained_dir, 'kan_ts_model.keras')
        
        self.input_dim = 0
        self.hidden_units = 32
        self.output_dim = 1
        self.epochs = epochs
        self.batch_size = batch_size


        self.drop_out = 0.2
        self.l2_reg = l2(0.01)
        self.initializer = GlorotUniform(seed=42)  


    def build_model(self, input_shape):

        l2_reg = l2(0.01)
        inputs = Input(shape=input_shape)
                
        reshaped_inputs = Reshape((self.input_dim, 1))(inputs)
        
        univariate_outputs = []
        for i in range(self.input_dim):
            x = LSTM(self.hidden_units, return_sequences=True, activation='relu')(reshaped_inputs[:, i:i+1, :])
            
            # Feature mixing
            x = Dense(self.hidden_units, activation='relu')(x)
            x = Dense(self.input_dim, activation='relu')(x)
            
            # Time mixing and second LSTM layer
            x = Reshape((1, self.input_dim))(x)
            x = LSTM(self.hidden_units, return_sequences=False, activation='relu')(x)
            
            univariate_outputs.append(x)

        # Combine univariate outputs using Concatenate
        concatenated_outputs = Concatenate(axis=1)(univariate_outputs)
        
        # Reshape the concatenated outputs to fit the expected input shape of the Attention layer
        reshaped_attention_input = Reshape((self.input_dim, self.hidden_units))(concatenated_outputs)
        attention_output = MultiHeadAttention(num_heads=self.input_dim//2, key_dim=self.input_dim//2, kernel_regularizer=l2_reg)(reshaped_attention_input, reshaped_attention_input)
        #attention_output = Attention()([reshaped_attention_input, reshaped_attention_input])
        
        # Flatten and final Dense layers
        flattened_output = Reshape((-1,))(attention_output)
        dense_output = Dense(self.hidden_units, activation='relu')(flattened_output)
    
        # Averaging and interaction layers
        sum_output = Add()(univariate_outputs)
        sum_output = Dense(self.hidden_units, activation='relu')(sum_output)
        #ave_output = Average()([sum_output, dense_output, sum_output,sum_output,])
        ave_output = Average()([sum_output, dense_output, sum_output])
        
        outputs = Dense(self.output_dim)(ave_output)
        
        self.model = Model(inputs, outputs)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', tf.keras.metrics.R2Score()])
        self.model.summary()
    
        dot_img_file = os.path.join(self.checkpoint_dir, 'kan_ts_plot.png')
        tf.keras.utils.plot_model(self.model, to_file=dot_img_file, show_shapes=True)
    
        return self.model
        

    def train_model(self, file_path):
    
        df = pd.read_csv(file_path)
        df = df.drop(columns=['outputC'])
        X = df.drop(columns=['output']).values
        y = df['output'].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        #self.saved_scaler = scaler
        #joblib.dump(scaler, self.checkpoint_scaler )


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.validation_split, random_state=42)
    
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
        self.input_dim = X_train.shape[1]
        input_shape=(X_train.shape[1], 1)
        
        
        self.build_model(input_shape)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(self.checkpoint_model, save_best_only=True, monitor='val_loss')
        
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
        
        history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, 
            validation_split=self.validation_split,  
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
        print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.4f}%")
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