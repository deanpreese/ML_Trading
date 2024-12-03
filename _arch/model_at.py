import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import pywt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import Lambda, Flatten, GRU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, Average, Conv2D, LeakyReLU, Reshape, Concatenate, Multiply,LayerNormalization 
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Add, Dense, Dropout, MaxPooling1D, LSTM, MultiHeadAttention, Attention

from tensorflow.keras.layers import AdditiveAttention

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2, l1_l2
from ml_model.model_stats import gen_reg_stats_x, gen_class_stats
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from ml_model.data_func import split_three_ways, sequence_and_split_3_ways, create_sequences_XY
from ml_model.k_model_base import K_MODEL_BASE


tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)


class MODEL_AT (K_MODEL_BASE):

    def __init__(self):
        
        self.checkpoint_dir = 'checkpoints/'
        self.trained_dir = 'trained_models/'
        self.model_name = 'model_at'
       
       
        self.checkpoint_model = os.path.join(self.checkpoint_dir, f"{self.model_name}.keras")
        self.trained_model = os.path.join(self.trained_dir, f"{self.model_name}.keras")
        self.model_plot = os.path.join(self.checkpoint_dir, f"{self.model_name}.png")

        self.drop_out = 0.2
        self.l2_reg = l2(0.01)
        self.initializer = GlorotUniform(seed=42)



    def build_model_x(self, inputs):
        
        x = Conv1D(filters=64, kernel_size=3, activation='relu', kernel_initializer=self.initializer)(inputs)       
        #x = LSTM(64, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(x)
        x = Conv1D(filters=32, kernel_size=2,  activation='relu', kernel_initializer=self.initializer)(x)
        #x = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(x)
        x = Conv1D(filters=16, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        x = Bidirectional(LSTM(32, kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(x)
        #x = Bidirectional(LSTM(64, kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(x)
        
        x = MaxPooling1D(pool_size=1, strides=1)(x)
        
        x = Bidirectional(LSTM(32, kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer))(x)
        x = Dense(32, activation='relu', kernel_regularizer=self.l2_reg,  kernel_initializer=self.initializer)(x) 
        attention_x = Dense(32, activation='softmax', kernel_initializer=self.initializer)(x)
        x = Multiply()([x, attention_x])                
        x = Dense(8, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(x) 
                     
        return x
    


 
    def build_model_h(self, inputs):

        """ 
        H
        Val MSE: 9.1925, Val MAE: 1.7172, R2: 0.46744909954386704
        Total Wins: 5652, Total Losses: 1799, Win Percentage: 0.759
        Number of Samples: 7451
        """        

        h = Conv1D(filters=64, kernel_size=3, activation='relu', kernel_initializer=self.initializer)(inputs) 
        #h = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(h)
        h = Conv1D(filters=32, kernel_size=2, activation='relu', kernel_initializer=self.initializer)(h)
        #h = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(h)
        h = Conv1D(filters=16, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(h)
        h = Bidirectional(LSTM(32,kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(h)
        h = Bidirectional(LSTM(32,kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer))(h)

        h = Dense(32, activation='relu', kernel_regularizer=self.l2_reg,  kernel_initializer=self.initializer)(h)         
        attention_h = Dense(32, activation='softmax', kernel_initializer=self.initializer )(h)
        h = Multiply()([h, attention_h])                
        h = Dense(8, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(h)           
        
        return h          
        

    def build_model_z(self, inputs):

        z = Conv1D(filters=64, kernel_size=5, activation='relu', kernel_initializer=self.initializer)(inputs) 
        #h = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(h)
        z = Conv1D(filters=32, kernel_size=4, activation='relu', kernel_initializer=self.initializer)(z)
        #h = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=True, kernel_initializer=self.initializer)(h)
        #z = Conv1D(filters=16, kernel_size=2, activation='relu', kernel_initializer=self.initializer)(z)
        z = Bidirectional(LSTM(32, kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(z)
        z = Bidirectional(LSTM(32,kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer))(z)

        #z = Dense(32, activation='relu', kernel_regularizer=self.l2_reg,  kernel_initializer=self.initializer)(z)         
        #attention_z = Dense(32, activation='softmax', kernel_initializer=self.initializer )(z)
        #z = Multiply()([z, attention_z])                
        z = Dense(8, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(z)           
        
        return z        
                

    # Build the Enhanced Model
    def create_model(self,input_shape):
        
        inputs = Input(shape=input_shape)
        h = self.build_model_h(inputs)
        x = self.build_model_x(inputs)
        z = self.build_model_z(inputs)
        
        h2 = self.build_model_h(inputs)
        x2 = self.build_model_x(inputs)
        z2 = self.build_model_z(inputs)
        
        
        """
        OOS Pred
        Pred MSE: 15.9459,  MAE: 2.3922, R2: 0.11343220697683987
        Total Wins: 2336, Total Losses: 1077, Win Percentage: 0.684400
        Accuracy Score: 0.6844418400234398
        Number of Samples: 3413
        """
        #final_out = 0.45 * h + 0.45 * x + 0.1 * z


        final_out = 0.40 * h + 0.40 * x + 0.025 * h2 + 0.025 * x2 + 0.05 * z
        #final_out = 0.40 * h + 0.40 * x + 0.2 * z

         
        #final_out = z
        output = Dense(1, activation='linear')(final_out)
        
        self.model = Model(inputs, output)
        return self.model


def process_data_file(file_to_load):
    
    print(f"Loading {file_to_load}" )
    df = pd.read_csv(file_to_load)
    
   
    f_list_f = ['SDBB91', 'COMP2', 'COMP3', 'ATR5', 'TV3', 'HourOfDay', 'TV1', 'ZH79X', 'SDKC29C', 
                        'ZL57X', 'COMP0', 'ATR2', 'TV6', 'RSI14', 'RSI9', 'ATR51' ,'output','outputC']   

    f_list_r = ['RSI9', 'ATR2', 'ATR5', 'ATR51', 'RSI14', 'TV3', 'TV6', 'COMP2', 'SDKC29C', 'COMP3', 'output','outputC']
    
    f_list_rx = ['RSI9', 'ATR2', 'ATR5', 'RSI14', 'TV3', 'TV6', 'COMP2', 'output','outputC']
    f_list_rxx = ['RSI9', 'ATR2', 'ATR5', 'RSI14', 'TV3', 'output','outputC']
    
    
    f_list_c = ['RSI9', 'ATR2', 'RSI14', 'TV6', 'TV1', 'ZL57X', 'COMP0', 'HourOfDay', 'SDBB91', 'ZH79X', 'output','outputC']

    f_list_cx = ['RSI9', 'ATR2', 'RSI14', 'TV6', 'TV1', 'ZL57X', 'COMP0', 'ZH79X', 'output','outputC']

    col_filter = f_list_f
    #df = df.drop(columns=['TimeTicks','SeqClose'])
    df=df[col_filter]
        
    #df = df[((df['RSI'] > 20) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 80)]  
    #df = df[((df['RSI'] > 25) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 75)] 
    #df = df[(df['RSI9'] > 60)]  
    #df = df[(df['RSI9'] < 40)]  
    
    X = df.drop(columns=['output','outputC']).values
    y = df['output'].values

    return X, y, df

def main():
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

    file_train = 3
    file_oos = 2
    time_steps = 13

    X_data, y_data, df_data = process_data_file(datafile[file_train])
    X_oos, y_oos, df_oos = process_data_file(datafile[file_oos])
    
    feature_dims, X_train, X_val, X_test, y_train, y_val, y_test  = sequence_and_split_3_ways(df_data, time_steps)
    X_oos, y_oos, = create_sequences_XY(X_oos, y_oos, time_steps=time_steps, step_size=1)
   
    input_shape = (time_steps, feature_dims)

    model_at = MODEL_AT()
    history_out, y_pred = model_at.train_model(input_shape, X_train, X_test, y_train, y_test, X_val, y_val, 100 )

    best_model = tf.keras.models.load_model(model_at.checkpoint_model)
    y_pred = best_model.predict(X_test)
    oos_model = tf.keras.models.load_model(model_at.checkpoint_model)
    y_pred_oos = oos_model.predict(X_oos)
    
    print("-----")
    print(f"Train Shape {X_train.shape}")
    print(f"Val Shape   {X_val.shape}")
    print(f"Test Shape  {X_test.shape}")
    print("-----")
    print("Test Pred")
    rmse, mse, mae, r2 = model_at.evaluate_model( y_test, y_pred)
    
    print("-----")
    print(f"OOS Test Shape {X_oos.shape}")
    print("-----")
    print("OOS Pred")
    rmse_oos, mse_oos, mae_oos, r2_oos = model_at.evaluate_model( y_oos, y_pred_oos)


    model_file = f"{model_at.model_name}_{mse_oos}_{mae_oos}_{r2_oos}_model.keras"
    oos_file_path = os.path.join(model_at.checkpoint_dir, model_file)
    oos_model.save(oos_file_path)
    
    oos_model_plot_file = f"{model_at.model_name}_{mse_oos}_{mae_oos}_{r2_oos}_model.png"
    oos_model_plot_path = os.path.join(model_at.checkpoint_dir, oos_model_plot_file)
    
    tf.keras.utils.plot_model(best_model, to_file=oos_model_plot_path, 
        show_shapes=True, 
        show_dtype=True,
        show_layer_names=True,
        expand_nested=True,
        show_layer_activations=True,
        show_trainable=True
    )   
    
            
if __name__ == "__main__":
    main()