import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Input, Conv1D, Average, Conv2D, LeakyReLU, Reshape, Concatenate, Multiply,LayerNormalization 
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Add, Dense, Dropout, MaxPooling1D, LSTM, MultiHeadAttention, Attention
from tensorflow.keras.layers import AdditiveAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2
from ml_model.model_stats import gen_reg_stats_x, gen_class_stats
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.metrics import confusion_matrix, accuracy_score

from ml_model.data_func import split_three_ways

#tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)

class FFTLayer(Layer):
    def call(self, inputs):
        # Ensure the inputs are cast to complex64 for FFT
        complex_inputs = tf.cast(inputs, tf.complex64)
        # Perform FFT
        fft_output = tf.signal.fft(complex_inputs)
        # Return the magnitude (abs value) of the FFT result
        return tf.math.abs(fft_output)
    
    def get_config(self):
        # Return the layer configuration for serialization
        config = super().get_config()
        return config



class K_MODEL_BASE:


    def setup_model(self, model_name):

        self.checkpoint_dir = 'checkpoints/'
        self.trained_dir = 'trained_models/'

        self.model_name = model_name

        self.checkpoint_model = os.path.join(self.checkpoint_dir, f"model_{model_name}.keras")
        self.trained_model = os.path.join(self.trained_dir, f"model_{model_name}.keras")
        self.model_plot = os.path.join(self.checkpoint_dir, f"model_{model_name}.png")

        self.drop_out = 0.3
        self.l2_reg = l2(0.01)
        self.initializer = GlorotUniform(seed=42)        
    

    def train_model(self, input_shape, X_train, X_test, y_train, y_test,  X_val, y_val, epochs ):
        
        self.create_model(input_shape )

        self.model.compile(optimizer=Adam(learning_rate=0.001), 
                loss='mse', metrics=['mae', tf.keras.metrics.R2Score()])
        self.model.summary()
        
        tf.keras.utils.plot_model(self.model, to_file=self.model_plot, 
            show_shapes=True, show_dtype=True, show_layer_names=True,
            expand_nested=True, show_layer_activations=True, show_trainable=True
            )   
        
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, verbose=1,
            mode="min", min_delta=1e-6, cooldown=0, min_lr=0
            )
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_model, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min'
            )
        
        history_out = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), 
            initial_epoch=0, epochs=epochs, verbose=1, batch_size=32, 
            callbacks=[early_stopping, reduce_lr, model_checkpoint]
            )      
        
        y_pred = self.model.predict(X_test)
        return history_out, y_pred


    def evaluate_finished_model(self, best_model, X_val, X_test, y_train, y_val, y_test,  X_oos, y_oos): 
    
        y_pred_test = best_model.predict(X_test)
        y_pred_val = best_model.predict(X_val)
        y_pred_oos = best_model.predict(X_oos)

        print("-----")
        print(f"Val Shape   {X_val.shape}")
        print(f"Test Shape  {X_test.shape}")
        print(f"OOS Shape {X_oos.shape}")
        print("-----")

        self.evaluate_model( y_val, y_pred_val, "Val")
        self.evaluate_model( y_test, y_pred_test, "Test")
        rmse_oos, mse_oos, mae_oos, r2_oos = self.evaluate_model( y_oos, y_pred_oos, "OOS")

        model_file = f"model_{self.model_name}_{mse_oos}_{mae_oos}_{r2_oos}_model.keras"
        oos_file_path = os.path.join(self.checkpoint_dir, model_file)
        best_model.save(oos_file_path)
        
        oos_model_plot_file = f"model_{self.model_name}_{mse_oos}_{mae_oos}_{r2_oos}_model.png"
        oos_model_plot_path = os.path.join(self.checkpoint_dir, oos_model_plot_file)
        
        tf.keras.utils.plot_model(best_model, to_file=oos_model_plot_path, 
            show_shapes=True, 
            show_dtype=True,
            show_layer_names=True,
            expand_nested=True,
            show_layer_activations=True,
            show_trainable=True
        )           
            
    

    def evaluate_model(self, y_test, y_pred, addtional_text=""):
        
        directional_accuracy = accuracy_score(np.sign(y_test), np.sign(y_pred))
        correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(y_test, y_pred)

        print(addtional_text)
        print(" ")
        print(f"Pred MSE: {mse},  MAE: {mae}, R2: {r2}")
        print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.6f}")
        print(f"Accuracy Score: {directional_accuracy}")        
        print(f"Number of Samples: {total}")        
        print(" ")        
        return rmse, mse, mae, r2
        
    
    def combined_plots(self, history, y_true, y_pred):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Subplot 1: Training History (Loss)
        axes[0, 0].plot(history.history['loss'], label='Train Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()

        # Subplot 2: Training History (MAE)
        axes[0, 1].plot(history.history['mae'], label='Train MAE')
        axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Model MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()

        # Subplot 4: Actual vs Predicted
        axes[1, 1].scatter(y_true, y_pred, alpha=0.5)
        axes[1, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Actual vs Predicted Values')

        plt.tight_layout()
        plt.show()



    def process_data_file(self, file_to_load, col_filter):
        
        print(f"Loading {file_to_load}" )
        df = pd.read_csv(file_to_load)
        X=df[col_filter]
        y = df['output'].values
        return X, y        


    def process_data_split(self, train_file, oos_file, col_filter):
        
        X_data, y_data = self.process_data_file(train_file, col_filter)
        X_oos, y_oos = self.process_data_file(oos_file, col_filter)
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_three_ways(X_data, y_data)
        input_shape = (X_train.shape[1], 1)
        #(14, 1)
        
        return X_train, X_val, X_test, y_train, y_val, y_test,  X_oos, y_oos, input_shape