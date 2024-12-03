import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib 
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Average, Reshape, Concatenate, ConvLSTM1D, Flatten, SeparableConv1D, BatchNormalization, LSTMCell, TimeDistributed, StackedRNNCells, RNN, LayerNormalization, Bidirectional, Add, Dense,  Dropout, MaxPooling1D, LSTM, MultiHeadAttention, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2

from ml_model.model_stats import gen_reg_stats_x, gen_class_stats 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from ml_model.model_stats import gen_reg_stats_x 
tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)


class MODEL_KA_CNN:
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


    def build_model(self, input_shape, model_type):
        
        l2_reg = l2(0.01)
        inputs = Input(shape=input_shape)
                
        input_dim = inputs.shape[1]               
        hidden_units = 16 
        output_units = 16               
        #reshaped_inputs = Reshape((input_dim, 1))(inputs)
        
        reshaped_inputs = inputs
                
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
            y = Conv1D(filters=16, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
            y = Conv1D(filters=16, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(y)
            y = Conv1D(filters=16, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(y)
            x = MaxPooling1D(pool_size=1, strides=1)(y)
            
            x_out = LSTM(hidden_units, return_sequences=False, activation='relu')(x)
            y_out = LSTM(hidden_units, return_sequences=False, activation='relu')(y)
            
            xy_output = Average()([x_out, y_out])
            #xy_output = 0.2*x_out + 0.8*y_out
            #xy_output = x_out
            
            univariate_outputs.append(xy_output)

        concatenated_outputs = Concatenate(axis=1)(univariate_outputs)
        reshaped_attention_input = Reshape((input_dim, hidden_units))(concatenated_outputs)
        attention_output = MultiHeadAttention(num_heads=input_dim//2, key_dim=input_dim//2, kernel_regularizer=l2_reg)(reshaped_attention_input, reshaped_attention_input)
                        
        #attention_output = MultiHeadAttention(num_heads=4, key_dim=8, kernel_regularizer=l2_reg)(reshaped_attention_input, reshaped_attention_input)

        flattened_output = Reshape((-1,))(attention_output)
        dense_output = Dense(output_units, activation='relu')(flattened_output)
    
        sum_output = Add()(univariate_outputs)
        sum_output = Dense(output_units, activation='relu')(sum_output)

        #ave_output = Average()([sum_output, dense_output, sum_output])
        #ave_output = Average()([sum_output, dense_output])
        ave_output = 0.5*sum_output + 0.5*dense_output
                
        
        outputs = Dense(1)(ave_output)
        
        self.model = Model(inputs, outputs)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', tf.keras.metrics.R2Score()])
        self.model.summary()
    
        dot_img_file = os.path.join(self.checkpoint_dir, 'kan_cnn_plot.png')
        tf.keras.utils.plot_model(self.model, to_file=dot_img_file, show_shapes=True)
    
        return self.model  
        

        
    
    def train_model(self, file_path, model_type, epochs):
    
        y_pred = None
        df = pd.read_csv(file_path)
        
        if model_type == "C":
            df = df.drop(columns=['output'])
            X = df.drop(columns=['outputC'])
            y = df['outputC'].values
            
        else:    
            df = df.drop(columns=['outputC'])
            X = df.drop(columns=['output'])
            y = df['output'].values            
                
        
        X = X.values
        
        X = X.reshape(X.shape[0], X.shape[1], 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
        input_shape = (X_train.shape[1], X_train.shape[2])
        #(14, 1)
        
        model = self.build_model(input_shape, model_type)

        if model_type == "R":
            self.model.compile(optimizer=Adam(learning_rate=0.001), 
                loss='mse', metrics=['mae', tf.keras.metrics.R2Score()])

        else:
            c_metrics = ['BinaryAccuracy', 'AUC', 'MeanSquaredError','accuracy', tf.keras.metrics.R2Score()]
            
            self.model.compile(optimizer=Adam(learning_rate=0.001), 
                             loss='binary_crossentropy', 
                             metrics=c_metrics)
                
            
        self.model.summary()
    
        dot_img_file = os.path.join(self.checkpoint_dir, 'kan_cnn_plot.png')
        tf.keras.utils.plot_model(self.model, to_file=dot_img_file, show_shapes=True)

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
                                initial_epoch=0, epochs=epochs, 
                                batch_size=32, callbacks=[
                                    early_stopping,
                                    reduce_lr,
                                    model_checkpoint])
        
        if model_type == "C":
            self.evaluate_classes(model, X_test, y_test)
            
        else:
            y_pred = model.predict(X_test)
            self.evaluate_model(y_pred)
            #self.plot_training_history(history_out)
        
        
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
        
        
    def evaluate_classes(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        perf, correct, total, tn, fp, fn, tp, mse, rmse, mae, r2 = gen_class_stats( y_test, predictions)
        print(f"Val MSE: {mse}, Val MAE: {mae}, R2: {r2}")
        print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.4f}")
        print(f"Number of Samples: {total}")
        print(f"TP {tp}   TN {tn}   FP {fp}   FN {fn}")     

        
def run():

    datafile = [ 
            'data/Lucky13_3070_oos.csv',
            'data/Lucky13_3070.csv',  #1
                          
            'data/NewModel_3070_oos.csv',   
            'data/NewModel_3070.csv',  #3
            'data/NewModel_ALL_oos.csv',   
            'data/NewModel_ALL.csv',  #5

            'data/NewModel_ALL_SPAN2.csv',   #6  
            'data/NewModel_ALL_SPAN3.csv',   #7
            'data/NewModel_ALL_SPAN6.csv',   #8
    ] 

    file_path = datafile[0]
    model = MODEL_KA_CNN()

    model_type = "R"
    train = True
    test = True
    single_item = False

    if train:
        file_path = datafile[1]
        history_out, y_pred = model.train_model(file_path, model_type,10)
        

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
        