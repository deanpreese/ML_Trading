import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib 
import matplotlib.pyplot as plt


from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Average, Reshape, Concatenate, ConvLSTM1D, Flatten, SeparableConv1D, LayerNormalization, Bidirectional, Add, Dense,  Dropout, MaxPooling1D, LSTM, MultiHeadAttention, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2

from ml_model.model_stats import gen_reg_stats_x 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from ml_model.model_stats import gen_reg_stats_x 
tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)


class KA_CNN:
    def __init__(self, epochs=50, batch_size=32):
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.checkpoint_dir = 'checkpoints/'
        self.trained_dir = 'trained_models/'
       
        self.checkpoint_model = os.path.join(self.checkpoint_dir, 'ka_cnn_model.keras')
        self.trained_model = os.path.join(self.trained_dir, 'ka_cnn_model.keras')


        self.drop_out = 0.2
        self.l2_reg = l2(0.02)
        self.initializer = GlorotUniform(seed=42)
        

    def build_model(self, input_shape):

        """
        set x 
        Val MSE: 9.3376, Val MAE: 1.7260, R2: 0.45904197704603666
        Total Wins: 5653, Total Losses: 1798, Win Percentage: 0.7587
        Number of Samples: 7451
        
        
        Val MSE: 9.5018, Val MAE: 1.7274, R2: 0.449529568922427
        Total Wins: 5653, Total Losses: 1798, Win Percentage: 0.7587
        Number of Samples: 7451
        
        """


        l2_reg = l2(0.01)
        inputs = Input(shape=input_shape)
                
        input_dim = inputs.shape[1]               
        hidden_units = 32                
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
            y = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
            y = Conv1D(filters=64, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(y)
            y = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(y)
            x = LSTM(64, return_sequences=True, activation='relu')(x)
            y = MaxPooling1D(pool_size=1, strides=1)(y)
            
            
            x = LSTM(32, return_sequences=False, activation='relu')(y)
            univariate_outputs.append(x)

        # Combine univariate outputs using Concatenate
        concatenated_outputs = Concatenate(axis=1)(univariate_outputs)
        
        # Reshape the concatenated outputs to fit the expected input shape of the Attention layer
        reshaped_attention_input = Reshape((input_dim, hidden_units))(concatenated_outputs)
        attention_output = MultiHeadAttention(num_heads=input_dim//2, key_dim=input_dim//2, kernel_regularizer=l2_reg)(reshaped_attention_input, reshaped_attention_input)
        
        # Flatten and final Dense layers
        flattened_output = Reshape((-1,))(attention_output)
        dense_output = Dense(hidden_units, activation='relu')(flattened_output)
    
        # Averaging and interaction layers
        sum_output = Add()(univariate_outputs)
        sum_output = Dense(hidden_units, activation='relu')(sum_output)
        ave_output = Average()([sum_output, dense_output, sum_output])
        outputs = Dense(1)(ave_output)
        
        self.model = Model(inputs, outputs)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', tf.keras.metrics.R2Score()])
        self.model.summary()
    
        dot_img_file = os.path.join(self.checkpoint_dir, 'kan_cnn_plot.png')
        tf.keras.utils.plot_model(self.model, to_file=dot_img_file, show_shapes=True)
    
        return self.model  
        
    
    def train_model(self, file_path):
    
        df = pd.read_csv(file_path)
        df = df.drop(columns=['outputC'])
        X = df.drop(columns=['output']).values
        y = df['output'].values

        # Reshape X to ensure it has the correct shape for LSTM
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
        input_shape = (X_train.shape[1], X_train.shape[2])
        #(14, 1)
        
        
        model = self.build_model(input_shape)

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.2,
            patience=5, verbose=1,
            mode="auto", min_delta=0.000001,
            cooldown=0, min_lr=0,
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_model, 
                monitor='val_loss', 
                    save_best_only=True, 
                        save_weights_only=False, mode='min')
        
        history_out = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                                initial_epoch=0, epochs=150, 
                                batch_size=32, callbacks=[
                                    early_stopping,
                                    reduce_lr,
                                    model_checkpoint])

        y_pred = model.predict(X_test)
        return history_out, y_pred


    def evaluate_model(self, y_pred):
        correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(self.y_test, y_pred)
        print(f"Val MSE: {mse}, Val MAE: {mae}, R2: {r2}")
        print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.4f}")
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
        
        
        
def run():

    datafile = [ 
        'data/Lucky13_3070_oos.csv',   
        'data/Lucky13_3070.csv',  #1
        'data/ndata_diff_lucky13_3070_oos.csv', 
        'data/ndata_diff_lucky13_3070.csv', #3
        'data/ndata_lucky_13_lag_3070_oos.csv', 
        'data/ndata_lucky13_lag_3070.csv', #5
        'new_model_Z_lucky13_3070_oos.csv',
        'new_model_Z_lucky13_3070.csv', #7,
        'data/Lucky13_3070_oos_3.csv',   
        'data/Lucky13_3070_3.csv',  #8
        'data/Lucky13_3070_oos_5.csv',   
        'data/Lucky13_3070_5.csv',  #10
        'data/new_model_HLC_lucky13.csv', #11
    ]

    file_path = datafile[1]
    model = KA_CNN()

    train = True
    test = False
    single_item = False

    if train:
        file_path = datafile[1]
        history_out, y_pred = model.train_model(file_path)
        model.evaluate_model(y_pred)
        model.plot_training_history(history_out)

    if test:
        file_path = datafile[0]
        model.load_saved_model("train")
        model.run_batch_test(file_path)

    if single_item:
        file_path = datafile[0]
        model.load_saved_model("run")

        df = pd.read_csv(file_path)
        df = df.drop(columns=['outputC'])
        X = df.drop(columns=['output']).values
        y = df['output'].values 

        model.X_test = X
        model.y_test = y

        yn = False
        count = 0
        ycount = 0

        y_pred = []

        for i in range(len(y)):
            x_val = X[i]
            x_val = x_val.reshape((1, 14, 1)) 
            y_val = model.model.predict(x_val)
            
            y_pred.append(y_val[0][0])
            print(y_val[0][0])

        model.evaluate_model(y_pred)
        

if __name__ == "__main__":
    run()            
        