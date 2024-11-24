import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib 
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Average, Multiply, GlobalAveragePooling1D, Reshape, Concatenate, ConvLSTM1D, Flatten, SeparableConv1D, LayerNormalization, Bidirectional, Add, Dense,  Dropout, MaxPooling1D, LSTM, MultiHeadAttention, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2

from ml_model.model_stats import gen_reg_stats_x, gen_class_stats 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from ml_model.data_func import split_three_ways

tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)


class DCNN:
    def __init__(self):

        self.checkpoint_dir = 'checkpoints/'
        self.trained_dir = 'trained_models/'
       
        self.checkpoint_model = os.path.join(self.checkpoint_dir, 'dcnn_model.keras')
        self.trained_model = os.path.join(self.trained_dir, 'dcnn_model.keras')
        self.dot_img_file = os.path.join(self.checkpoint_dir, 'dcnn.png')

        self.drop_out = 0.3
        self.l2_reg = l2(0.01)
        self.initializer = GlorotUniform(seed=42)        
    
    
    def build_model_h(self, inputs):

        """ 
        H
        Val MSE: 9.1925, Val MAE: 1.7172, R2: 0.46744909954386704
        Total Wins: 5652, Total Losses: 1799, Win Percentage: 0.759
        Number of Samples: 7451
        """        

        h = Conv1D(filters=64, kernel_size=4, activation='relu', kernel_initializer=self.initializer)(inputs) 
        #h = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(h)
        h = Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer=self.initializer)(h)
        #h = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(h)
        h = Conv1D(filters=16, kernel_size=2, activation='relu', kernel_initializer=self.initializer)(h)
        h = Bidirectional(LSTM(32,name="BIC", kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(h)
        h = Bidirectional(LSTM(32,name="BIC2", kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer))(h)

        h = Dense(32, activation='relu', kernel_regularizer=self.l2_reg,  kernel_initializer=self.initializer)(h)         
        attention_h = Dense(32, activation='softmax', kernel_initializer=self.initializer, name='attention_h')(h)
        h = Multiply()([h, attention_h])                
        h = Dense(8, activation='relu', kernel_regularizer=self.l2_reg, name="h_out", kernel_initializer=self.initializer)(h)           
        
        return h        
        
        
    def build_model_x(self, inputs):
        
        """
        X        
        Val MSE: 9.2086, Val MAE: 1.7184, R2: 0.4665146637449804
        Total Wins: 5651, Total Losses: 1800, Win Percentage: 0.758
        Number of Samples: 7451    
        """        
        
        x = Conv1D(filters=64, kernel_size=4, activation='relu', kernel_initializer=self.initializer)(inputs)       
        #x = LSTM(64, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(x)
        x = Conv1D(filters=32, kernel_size=3,  activation='relu', kernel_initializer=self.initializer)(x)
        #x = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(x)
        x = Conv1D(filters=16, kernel_size=2, activation='relu', kernel_initializer=self.initializer)(x)
        x = Bidirectional(LSTM(32, kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(x)
        x = Bidirectional(LSTM(64, kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(x)
        
        x = MaxPooling1D(pool_size=1, strides=1)(x)
        
        x = Bidirectional(LSTM(32, kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer))(x)
        x = Dense(32, activation='relu', kernel_regularizer=self.l2_reg,  kernel_initializer=self.initializer)(x) 
        attention_x = Dense(32, activation='softmax', kernel_initializer=self.initializer, name='attention_x')(x)
        x = Multiply()([x, attention_x])                
        x = Dense(8, activation='relu', kernel_regularizer=self.l2_reg, name="x_out", kernel_initializer=self.initializer)(x) 
                     
        return x
    
    

    def build_model(self, input_shape):
        
        inputs = Input(shape=input_shape)
        #x = self.build_model_x(inputs)
        
        h = self.build_model_h(inputs)
        
        ave_output = h
        
        outputs = Dense(1)(ave_output)  
        model = Model(inputs=inputs, outputs=outputs)
        self.model = model
        return model        
                
        
        
    def train_model(self, input_shape, X_train, X_test, y_train, y_test ):
    
        
        model = self.build_model(input_shape)

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', tf.keras.metrics.R2Score()])
        model.summary(expand_nested=True,show_trainable=True)
 
        tf.keras.utils.plot_model(model, to_file=self.dot_img_file, 
            show_shapes=True, 
            show_dtype=True,
            show_layer_names=True,
            expand_nested=True,
            show_layer_activations=True,
            show_trainable=True
            )   
 

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
                                initial_epoch=0, epochs=2000, 
                                batch_size=64, callbacks=[
                                    early_stopping,
                                    reduce_lr,
                                    model_checkpoint])

        y_pred = model.predict(X_test)
        return history_out, y_pred, y_test, X_test


def evaluate_model( y_test, y_pred):
    
    correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(y_test, y_pred)
    print(" ")
    print(f"Pred MSE: {mse},  MAE: {mae}, R2: {r2}")
    print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.4f}")
    print(f"Number of Samples: {total}")        
    print(" ")        
    
    return rmse, mse, mae, r2
    
        
        
def plot_training_history(history):
    
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
        'data/Lucky13_EX_3070_oos.csv',  
        'data/Lucky13_EX_3070.csv',  #3
        'data/Lucky13_ALL_oos.csv',  #4
        'data/Lucky13_ALL.csv',  #5
    ]

    #file_path = datafile[13]
    dcnn = DCNN()

    train = True
    run_oos = False
    run_perf = False

    if train:
        
        for i in range(5):
            file_path = datafile[1]
            
            print(f"Loading {file_path}" )
            df = pd.read_csv(file_path)
            df = df.drop(columns=['outputC'])
            
            df = df[(df['RSI'] > 60) & (df['RSI'] < 80)]  #  81%
            #df = df[(df['RSI'] > 60) & (df['RSI'] < 75)]   # 878%
            #df = df[(df['RSI'] > 20) & (df['RSI'] < 40)]  # 84%
            #df = df[(df['RSI'] > 25) & (df['RSI'] < 40)]  # 91%
            
            X = df.drop(columns=['output']).values
            y = df['output'].values
            X = X.reshape(X.shape[0], X.shape[1], 1)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    
            input_shape = (X_train.shape[1], X_train.shape[2])
            #(14, 1)
                    
            history_out, y_pred, y_test, X_test = dcnn.train_model(input_shape, X_train, X_test, y_train, y_test )

            # Load best model and evaluate
            best_model = tf.keras.models.load_model(dcnn.checkpoint_model)
            y_pred = best_model.predict(X_test)
            rmse, mse, mae, r2 = evaluate_model( y_test, y_pred)
            
            model_file = f"dcnn_{mse}_{mae}_model.keras"
            file_path = os.path.join(dcnn.checkpoint_dir, model_file)
            best_model.save(file_path)
            

    
    if run_oos:
        
        file_path = datafile[0]
        df = pd.read_csv(file_path)
        df = df.drop(columns=['outputC'])
        X = df.drop(columns=['output']).values
        y = df['output'].values 
        
        
        model_dir = "saved_models/dcnn/"
        files = os.listdir(model_dir)
        model_to_load = files[0]
        model_xxx= os.path.join(model_dir, model_to_load)
        
        oos_model = tf.keras.models.load_model(model_xxx)
        y_pred = oos_model.predict(X, verbose=0)
        evaluate_model( y, y_pred)

    if run_perf:
        file_path = datafile[0]
        
        model_lower = "dcnn_25.keras"
        model_upper = "dcnn_75.keras"
        saved_model_dir = "saved_models"
            
        loaded_model_lower= os.path.join(saved_model_dir, model_lower)
        loaded_model_upper= os.path.join(saved_model_dir, model_upper)
        perf_model_lower = tf.keras.models.load_model(loaded_model_lower)
        perf_model_upper = tf.keras.models.load_model(loaded_model_upper)
        
        df = pd.read_csv(file_path)
        df = df.drop(columns=['outputC'])
        
        y_pred = []
        y_target = []
        
        for idx, row in df.iterrows():
            
            rsi = row['RSI']
            target = row['output']
            
            if rsi > 20 and rsi < 40:
                
                x_val = row[:-1].values
                x_val = x_val.reshape((1, 14, 1)) 
                y_val = perf_model_lower.predict(x_val)
                y_pred.append(y_val[0])
                y_target.append(target)
                
            if rsi > 60 and rsi < 80:
            
                x_val = row[:-1].values
                x_val = x_val.reshape((1, 14, 1)) 
                y_val = perf_model_upper.predict(x_val, verbose=0)
                y_pred.append(y_val[0])
                y_target.append(target)
                
        evaluate_model(y_target, y_pred)


if __name__ == "__main__":
    run()            
        