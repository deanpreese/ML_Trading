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
        self.trained_dir = 'trained_models/'
       
        self.checkpoint_model = os.path.join(self.checkpoint_dir, 'dcnn_model.keras')
        self.trained_model = os.path.join(self.trained_dir, 'dcnn_model.keras')


        self.drop_out = 0.2
        self.l2_reg = l2(0.02)
        self.initializer = GlorotUniform(seed=42)
        

    def build_model(self, input_shape):
        inputs = Input(shape=input_shape)
        
        """
        using b + y3
        Val MSE: 9.3517, Val MAE: 1.7306, R2: 0.4582243744784792
        Total Wins: 5648, Total Losses: 1803, Win Percentage: 0.758
        Number of Samples: 7451

        """              
        a = Conv1D(filters=64, kernel_size=3, activation='relu', kernel_initializer=self.initializer)(inputs)
        a = LSTM(64, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(a)
        a = Conv1D(filters=32, kernel_size=2, activation='relu', kernel_initializer=self.initializer)(a)
        a = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(a)
        a = Conv1D(filters=64, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(a)
        a = MaxPooling1D(pool_size=1, strides=1)(a)
        ya= Dropout(self.drop_out)(a)
        ya = Bidirectional(LSTM(32,name="BIC", kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(ya)
        ya = Dropout(self.drop_out)(ya)
        ya = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', kernel_initializer=self.initializer)(ya)
        ya = Dropout(self.drop_out)(ya)
        ya = Dense(16, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(ya)
        
        #------

        b = Conv1D(filters=64, kernel_size=2, activation='relu', kernel_initializer=self.initializer)(inputs)
        b = LSTM(64, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(b)
        b = Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer=self.initializer)(b)
        b = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(b)
        b = Conv1D(filters=64, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(b)
        b = MaxPooling1D(pool_size=1, strides=1)(b)
        yb = Dropout(self.drop_out)(b)
        yb = Bidirectional(LSTM(32,name="BIC", kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(yb)
        yb = Dropout(self.drop_out)(yb)
        #yb = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', kernel_initializer=self.initializer)(yb)
        yb = Dropout(self.drop_out)(yb)
        yb = Dense(16, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(yb)
        
        """
        ave_output = Average()([ya, yb])
        Val MSE: 9.4009, Val MAE: 1.7269, R2: 0.45537637866248615
        Total Wins: 5653, Total Losses: 1798, Win Percentage: 0.759
        Number of Samples: 7451

        ave_output = 0.4*yb + 0.6*ya
        Val MSE: 9.4056, Val MAE: 1.7313, R2: 0.45510589571598137
        Total Wins: 5652, Total Losses: 1799, Win Percentage: 0.759
        Number of Samples: 7451      

        ave_output = Average()([ya, ya, yb])
        Val MSE: 9.3818, Val MAE: 1.7219, R2: 0.45647946626513114
        Total Wins: 5651, Total Losses: 1800, Win Percentage: 0.758
        Number of Samples: 7451    

        ave_output = 0.45*yb + 0.55*ya
        Val MSE: 9.5536, Val MAE: 1.7303, R2: 0.44653159753668414
        Total Wins: 5652, Total Losses: 1799, Win Percentage: 0.759
        Number of Samples: 7451
        
        ya
        Val MSE: 9.2932, Val MAE: 1.7292, R2: 0.4616171211504386
        Total Wins: 5647, Total Losses: 1804, Win Percentage: 0.758
        Number of Samples: 7451
        
        yb
        Val MSE: 9.8263, Val MAE: 1.7446, R2: 0.43072936312387267
        Total Wins: 5650, Total Losses: 1801, Win Percentage: 0.758
        Number of Samples: 7451

        """     
        #ave_output = Average()([ya, ya, yb, ya, yb])
        ave_output = yb

        outputs = Dense(1)(ave_output)  
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', tf.keras.metrics.R2Score()])
        model.summary(expand_nested=True,show_trainable=True)
        
            
        dot_img_file = os.path.join(self.checkpoint_dir, 'dcnn.png')
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    
        
        print(" ")
        print(" ----- ")
        print(" ")
        self.model = model
        return model        
                
        
        
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
        'data/new_model_HLC_lucky13.csv', #11
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
        