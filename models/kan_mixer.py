import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Add, LSTM, Attention, Average, Reshape, Concatenate
from tensorflow.keras.optimizers import Adam

from keras.layers import LeakyReLU, Dropout, MultiHeadAttention
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import MeanSquaredError, BinaryCrossentropy, BinaryAccuracy, AUC  
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class KANMixerModel:
    def __init__(self, input_dim, hidden_units=32, output_dim=1, epochs=100, batch_size=32):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.build_model()

    def build_model(self):

        l2_reg = l2(0.01)
        inputs = Input(shape=(self.input_dim,))
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
        ave_output = Average()([sum_output, dense_output, sum_output,sum_output,])
        
        outputs = Dense(self.output_dim)(ave_output)
        
        # Build and compile the model
        self.model = Model(inputs, outputs)
        self.model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
        self.model.summary()
    
        return self.model
        


    def train(self, X_train, y_train, epochs, model_save_path, batch_size=32, validation_split=0.2 ):
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss')
        
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
        
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
            validation_split=validation_split, 
            callbacks=[
                early_stopping, 
                #reduce_lr, 
                model_checkpoint
                ]
            )
        
        
        return history

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Win/Loss calculation
        wins = 0
        losses = 0
        
        # Win/Loss calculation
        #wins = np.sum((y_pred > 0) & (y_test > 0)) + np.sum((y_pred < 0) & (y_test < 0))
        #losses = np.sum((y_pred > 0) & (y_test < 0)) + np.sum((y_pred < 0) & (y_test > 0))
        #total_samples = wins + losses
        #win_percentage = (wins / total_samples) * 100 if total_samples > 0 else 0
        
        
        for i in range(len(y_test)):
            if (y_pred[i] > 0 and y_test[i] > 0) or (y_pred[i] < 0 and y_test[i] < 0):
                wins += 1
            elif (y_pred[i] > 0 and y_test[i] < 0) or (y_pred[i] < 0 and y_test[i] > 0):
                losses += 1
            elif (y_pred[i] == 0 and y_test[i] == 0):
                wins += 1
            elif (y_pred[i] == 0 and y_test[i] != 0):
                losses += 1
        
        total_samples = wins + losses
        win_percentage = (wins / total_samples) * 100
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'Total Wins': wins,
            'Total Losses': losses,
            'Win Percentage': win_percentage,
            'Samples': total_samples
        }
        
        return metrics

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
