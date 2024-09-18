import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib 
import matplotlib.pyplot as plt


from tensorflow.keras.layers import Lambda, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Average, GlobalAveragePooling1D, Reshape, Concatenate, ConvLSTM1D, Flatten, SeparableConv1D, LayerNormalization, Bidirectional, Add, Dense,  Dropout, MaxPooling1D, LSTM, MultiHeadAttention, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2

from ml_model.data_func  import sequence_and_split
from ml_model.model_stats import gen_reg_stats_x 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

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
       
        self.checkpoint_model = os.path.join(self.checkpoint_dir, 'dcnn_x_model.keras')
        self.trained_model = os.path.join(self.trained_dir, 'dcnn_x_model.keras')
        self.dot_img_file = os.path.join(self.checkpoint_dir, 'dcnn_x.png')


        self.drop_out = 0.2
        self.l2_reg = l2(0.01)
        self.initializer = GlorotUniform(seed=42)
        


    def create_feature_model(self, input_shape):
        
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)
        inx = LSTM(16, return_sequences=True, activation='relu')(reshaped_inputs)
        

        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        x = MaxPooling1D(pool_size=1, strides=1)(x)
        smx_out = Dense(1, activation='linear')(x) 
        subx_model = Model(input, smx_out)
        
        return subx_model

    """
    def build_model(self, input_shape):
        
        inputs = Input(shape=input_shape)
        
        lstm_out = LSTM(32, return_sequences=True, kernel_regularizer=self.l2_reg, recurrent_regularizer=self.l2_reg)(inputs)
        #attention = MultiHeadAttention(num_heads=4, key_dim=64, kernel_regularizer=self.l2_reg)(lstm_out, lstm_out)
        #attention = Add()([attention, lstm_out])
        #attention = LayerNormalization()(attention)
        #dp = Dropout(self.drop_out)(attention)
        #dense = Dense(16, activation='relu', kernel_regularizer=self.l2_reg)(dp)
        
        #x = Flatten()(attention)
        x = lstm_out
        num_features = inputs.shape[1]
        feature_outputs = []
        
        for i in range(num_features):
            
            feature_input = Lambda(lambda x: x[:, i:i+1])(x)
            feature_model = self.create_feature_model((1,))
            feature_output = feature_model(feature_input)
            feature_outputs.append(feature_output)

        concatenated_outputs = Concatenate(axis=1)(feature_outputs)
        reshaped_attention_input = Reshape((num_features, 1))(concatenated_outputs)
        
        attention = Dense(num_features, activation='softmax', kernel_initializer=self.initializer, name='attention')(reshaped_attention_input)
        weighted = Multiply()([reshaped_attention_input, attention])
        
        #weighted = Conv1D(32, 1, activation='relu', kernel_initializer=self.initializer)(weighted)
        
        print(f"Weighted {weighted.shape}")
        
        weighted = Reshape((-1,))(weighted)
        print(f"Weighted {weighted.shape}")
        
        aggregated = Dense(32, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(weighted)
        aggregated = Dropout(self.drop_out)(aggregated)
        
        outputs = Dense(1, activation='linear')(aggregated)  
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', tf.keras.metrics.R2Score()])
        model.summary(expand_nested=True,show_trainable=True)
            
        
        tf.keras.utils.plot_model(model, to_file=self.dot_img_file, show_shapes=True)
    
        
        print(" ")
        print(" ----- ")
        print(" ")
        self.model = model
        return model        
        """                
        


    def build_model(self, input_shape):
        inputs = Input(shape=input_shape)
        
        """
        H
        Val MSE: 9.1925, Val MAE: 1.7172, R2: 0.46744909954386704
        Total Wins: 5652, Total Losses: 1799, Win Percentage: 0.759
        Number of Samples: 7451
                
        X        
        Val MSE: 9.2086, Val MAE: 1.7184, R2: 0.4665146637449804
        Total Wins: 5651, Total Losses: 1800, Win Percentage: 0.758
        Number of Samples: 7451    
        
        """        

        h = Conv1D(filters=64, kernel_size=4, activation='relu', kernel_initializer=self.initializer)(inputs) 
        h = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(h)
        h = Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer=self.initializer)(h)
        h = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(h)
        h = Conv1D(filters=16, kernel_size=2, activation='relu', kernel_initializer=self.initializer)(h)
        h = Bidirectional(LSTM(32,name="BIC", kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(h)
        h = Bidirectional(LSTM(16,name="BIC2", kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer))(h)
        h = Dense(8, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(h) 
                
                
                
        x = Conv1D(filters=256, kernel_size=5, activation='relu', kernel_initializer=self.initializer)(inputs)       
        #x = LSTM(64, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(x)
        x = Conv1D(filters=128, kernel_size=4, activation='relu', kernel_initializer=self.initializer)(x)
        #x = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(x)
        x = Conv1D(filters=64, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        #x = Bidirectional(LSTM(32,name="BIC", kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(x)
        #x = Bidirectional(LSTM(16,name="BIC2", kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(x)
        x = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=False, kernel_initializer=self.initializer)(x)
        #x = Dense(16, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(x) 
                
        outputs = Dense(1)(x)  
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', tf.keras.metrics.R2Score()])
        model.summary(expand_nested=True,show_trainable=True)
            
        
        tf.keras.utils.plot_model(model, to_file=self.dot_img_file, show_shapes=True)
    
        
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

        time_steps = 21

        feature_dims, X_train, X_test, y_train, y_test = sequence_and_split(file_path, time_steps)
        
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
        print(f"X_Train  {X_train.shape}")

        #input_layer = Input(shape=(sequence_length, input_dim)) 
        #X_Train  (26072, 7, 14)
        input_shape = (X_train.shape[1], X_train.shape[2])
        
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
        #model.plot_training_history(history_out)

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
        