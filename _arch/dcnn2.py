import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib 
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Average, LayerNormalization, Bidirectional, Add, Dense,  Dropout, MaxPooling1D, LSTM, MultiHeadAttention, Attention
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

from ml_model.data_func import sequence_and_normalize
from ml_model.model_stats import gen_reg_stats_x 
tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)


class DCNN:
    def __init__(self, epochs=50, batch_size=32):
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.checkpoint_dir = 'checkpoints/'
        self.checkpoint_model = os.path.join(self.checkpoint_dir, 'dcnn_model.keras')
        self.trained_dir = 'trained_models/'
        self.trained_model = os.path.join(self.trained_dir, 'dcnn_model.keras')



    def build_model(self, input_shape):

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

        outputs = Dense(1)(c)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', tf.keras.metrics.R2Score()])
        model.summary()
            
        checkpoint_dir = 'checkpoints/'        
        dot_img_file = os.path.join(checkpoint_dir, 'dcnn_s_plot.png')
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=True)
        print(" ")

        return model



    def build_model_o(self, input_shape):
        
        l2_reg = l2(0.01)
        drop_out = 0.2
        output_units = 16
        lay1_lstm_units = 64
        
        inputs = Input(shape=input_shape)
        initializer = GlorotUniform(seed=42)  
        
        c = Conv1D(name="C", filters=128, kernel_size=4, activation='relu',kernel_regularizer=l2_reg, kernel_initializer=initializer)(inputs)
        c = Conv1D(filters=64, kernel_size=3, activation='relu',kernel_regularizer=l2_reg, kernel_initializer=initializer)(c)
        c = Conv1D(filters=128, kernel_size=2, activation='relu',kernel_regularizer=l2_reg, kernel_initializer=initializer)(c)
        c = MaxPooling1D(pool_size=6, strides=2)(c)
        c = Dropout(drop_out)(c)
        c = Bidirectional(LSTM(lay1_lstm_units,name="BIC", kernel_regularizer=l2_reg, kernel_initializer=initializer))(c)
        c = Dropout(drop_out)(c)
        c = Dense(name="C_out", units=output_units, activation='relu',kernel_regularizer=l2_reg, kernel_initializer=initializer)(c)
        
        d = Conv1D(name="D", filters=128, kernel_size=4, activation='relu',kernel_regularizer=l2_reg, kernel_initializer=initializer)(inputs)
        d = Conv1D(filters=64, kernel_size=3, activation='relu',kernel_regularizer=l2_reg, kernel_initializer=initializer)(d)
        d = Conv1D(filters=128, kernel_size=2, activation='relu',kernel_regularizer=l2_reg, kernel_initializer=initializer)(d)
        #d = MaxPooling1D(pool_size=2, strides=2)(d)
        d = Dropout(drop_out)(d)
        d = Bidirectional(LSTM(lay1_lstm_units,return_sequences=True, name="BID",kernel_regularizer=l2_reg, kernel_initializer=initializer))(d)
        d = Conv1D(filters=64, kernel_size=3, activation='relu',kernel_regularizer=l2_reg, kernel_initializer=initializer)(d)
        d = Dropout(drop_out)(d)
        d = Dense(name="D_out",units=output_units, activation='relu',kernel_regularizer=l2_reg, kernel_initializer=initializer)(d)        
        
        #ave1 = Average()([c,d]) 
        ave1 = c
        
        hidden_dim = 64
        seq_length = 32
        x = tf.keras.layers.Dense(hidden_dim, activation='relu', kernel_regularizer=l2_reg, kernel_initializer=initializer)(ave1)
        x = tf.keras.layers.Dense(seq_length, activation='relu', kernel_regularizer=l2_reg, kernel_initializer=initializer)(x)
        # Flatten and output
        x = tf.keras.layers.Flatten()(x)
        outputs = Dense(1)(c)  
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', tf.keras.metrics.R2Score()])
        model.summary()
            
        dot_img_file = os.path.join(self.checkpoint_dir, 'dcnn_plot.png')
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=True)
        print(" ")
        self.model = model
        return model
    
    
    def train_model(self, file_path):
    
        time_steps = 9
        feature_dims, X_train, X_test, y_train, y_test, scalers = sequence_and_normalize(file_path, time_steps)

        print(X_train.shape)
        print(y_train.shape)
    
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
        input_shape=(X_train.shape[1], 1)
        
        input_layer = Input(shape=(time_steps, feature_dims))
        print("Input Shape:", input_layer.shape)
        
        self.build_model(input_shape)

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

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_model, monitor='val_loss', save_best_only=True, save_weights_only=False, mode='min')
        
        history_out = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                                     initial_epoch=0, epochs=100, 
                                     batch_size=32, callbacks=[
                                         early_stopping,
                                         reduce_lr,
                                         model_checkpoint])

        y_pred = self.model.predict(X_test)
        return history_out, y_pred


    def evaluate_model(self, y_pred):
        correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(self.y_test, y_pred)
        print(f"Val MSE: {mse}, Val MAE: {mae}, R2: {r2}")
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
    ]

    file_path = datafile[1]
    model = DCNN()

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
        