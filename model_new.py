import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import pywt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, Average, Conv2D, LeakyReLU, Reshape, Concatenate, Multiply,LayerNormalization 
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Add, Dense, Dropout, MaxPooling1D, LSTM, MultiHeadAttention, Attention

from tensorflow.keras.layers import AdditiveAttention

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2
from ml_model.model_stats import gen_reg_stats_x, gen_class_stats
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from ml_model.data_func import split_three_ways
from ml_model.k_model_base import K_MODEL_BASE


tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)


class MODEL_NEW (K_MODEL_BASE):
    def __init__(self):
        
        self.checkpoint_dir = 'checkpoints/'
        self.trained_dir = 'trained_models/'
       
        self.checkpoint_model = os.path.join(self.checkpoint_dir, 'model_new.keras')
        self.trained_model = os.path.join(self.trained_dir, 'model_new.keras')
        self.model_plot = os.path.join(self.checkpoint_dir, 'model_new.png')

        self.drop_out = 0.2
        self.l2_reg = l2(0.01)
        self.initializer = GlorotUniform(seed=42)

    def create_feature_model_h(self, inputs, output_dim):

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

        attention_output = MultiHeadAttention(num_heads=4, key_dim=16)(h, h)
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
        h = Dropout(self.drop_out)(attention_output)

        h = Bidirectional(LSTM(32,name="BIC", kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(h)
        h = Bidirectional(LSTM(32,name="BIC2", kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer))(h)
        h = Dense(32, activation='relu', kernel_regularizer=self.l2_reg,  kernel_initializer=self.initializer)(h)         

        attention_h = Dense(32, activation='softmax', kernel_initializer=self.initializer, name='attention_h')(h)
        h = Multiply()([h, attention_h])                
        
        h = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg, name="h_out", kernel_initializer=self.initializer)(h)           
       
        return h   


    def create_feature_model_x(self, inputs, output_dim):
        
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
        
        attention_output = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
        x = Dropout(self.drop_out)(attention_output)

        
        x = Bidirectional(LSTM(32, kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(x)
        x = Bidirectional(LSTM(64, kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(x)
        
        x = MaxPooling1D(pool_size=1, strides=1)(x)
        
        x = Bidirectional(LSTM(32, kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer))(x)
        x = Dense(32, activation='relu', kernel_regularizer=self.l2_reg,  kernel_initializer=self.initializer)(x) 
        attention_x = Dense(32, activation='softmax', kernel_initializer=self.initializer, name='attention_x')(x)
        x = Multiply()([x, attention_x])                
        x = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg, name="x_out", kernel_initializer=self.initializer)(x) 

        return x

    
    def create_feature_model_c1(self, inputs, output_dim):
        """
        Feature Model C1 with Layer Normalization and Dropout
        """
        x = LSTM(32, return_sequences=True, activation='relu')(inputs)
        x = Dropout(self.drop_out)(x)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        x = BatchNormalization()(x)
        x = Dropout(self.drop_out)(x)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        x = BatchNormalization()(x)
        x = Dropout(self.drop_out)(x)
        x = MaxPooling1D(pool_size=1, strides=1)(x)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        x = Dropout(self.drop_out)(x)
        
        out_r = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(x) 
        out_s = Dense(output_dim, activation='sigmoid', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(x) 
        out_t = Dense(output_dim, activation='tanh', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(x) 
        
        x = Average()([out_r,out_s, out_t])
        
        return x

    def create_feature_model_c2(self, inputs, output_dim):
        """
        Feature Model C2 with Layer Normalization and Dropout
        """
        x = LSTM(32, return_sequences=True, activation='relu')(inputs)
        x = Dropout(self.drop_out)(x)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        x = BatchNormalization()(x)
        x = Dropout(self.drop_out)(x)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        x = BatchNormalization()(x)
        x = Dropout(self.drop_out)(x)
        x = MaxPooling1D(pool_size=1, strides=1)(x)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        x = Dropout(self.drop_out)(x)
        
        out_r = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(x) 
        out_s = Dense(output_dim, activation='sigmoid', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(x) 
        out_t = Dense(output_dim, activation='tanh', kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer)(x) 
        
        x = Average()([out_r,out_s, out_t])
        return x    


    def create_feature_model_linear(self, input_shape):
        
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)
        inx = LSTM(32, return_sequences=True, activation='relu')(reshaped_inputs)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        smx_out = Dense(1, activation='linear')(x) 
        subx_model = Model(input, smx_out)
        return subx_model

    def create_feature_model_sigmoid(self, input_shape):
        
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)
        inx = LSTM(32, return_sequences=True, activation='relu')(reshaped_inputs)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        smx_out = Dense(1, activation='sigmoid')(x) 
        subx_model = Model(input, smx_out)
        return subx_model

    def create_feature_model_tanh(self, input_shape):
        
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)
        inx = LSTM(32, return_sequences=True, activation='relu')(reshaped_inputs)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        smx_out = Dense(1, activation='tanh')(x) 
        subx_model = Model(input, smx_out)
        return subx_model


   
    def create_model(self, input_shape ):
        
        inputs = Input(shape=input_shape)
        feature_outputs_linear = []
        feature_outputs_sigmoid = []
        feature_outputs_tanh = []
        
        #concat_dims = 32
        concat_dims = input_shape[0]
        output_dim = 8
        
        model_h = self.create_feature_model_h(inputs, output_dim)
        model_x = self.create_feature_model_x(inputs, output_dim)
        model_c1 = self.create_feature_model_c1(inputs, output_dim)
        model_c2 = self.create_feature_model_c2(inputs, output_dim)
        
        
        concat_dims = input_shape[0]
        
        for i in range(input_shape[0]):
            feature_input = inputs[:, i:i+1]

            fa = self.create_feature_model_linear((1,))
            fax = fa(feature_input)
            feature_outputs_linear.append(fax)
            
            fm1 = self.create_feature_model_sigmoid((1,))
            fm1x = fm1(feature_input)
            feature_outputs_sigmoid.append(fm1x)
            
            fm2 = self.create_feature_model_tanh((1,))
            fm2x = fm2(feature_input)
            feature_outputs_tanh.append(fm2x)
       
       
        concat_lin = Concatenate(axis=1)(feature_outputs_linear)
        concat_sig = Concatenate(axis=1)(feature_outputs_sigmoid)
        concat_tan = Concatenate(axis=1)(feature_outputs_tanh)
        
        xa = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(concat_lin)
        xs = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(concat_sig)
        xt = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(concat_tan)
        
        fc = 0.7 * concat_lin  + 0.15 * concat_sig +  0.15 * concat_tan  
        fc = Dense(32, activation='relu')(fc)
        fc = Dense(output_dim, activation='relu')(fc)
        #x = fc  
        
        concat_filtered = xa + xs + xt  #oos 10 epoch 11.6   1.9  .37   .768   
        cf = 0.4 * fc + 0.6 * concat_filtered  #oos 10 epoch  11.68  1.9497  .36598   .7692
        x = cf
        
        #x = model_h   #oos 10 epoch 12.9   2.05  .2992   .7675
        #x = model_x   #oos 10 epoch 12.8   2.05  .3052   .7675
        #x = model_c1  #oos 10 epoch 11.4   1.9  .37716   .7686
        #x = model_c2  #oos 10 epoch 11.93   1.84  .3526   .7695
        
        #Pred MSE: 11.4941,  MAE: 1.8776, R2: 0.37632320886813575
        #x = model_c1 * 0.125 + model_c2 * 0.125 +  cf * 0.75
        
        #x = model_c1 * 0.3 + model_c2 * 0.3 +  cf * 0.4  #Pred MSE: 11.3626,  MAE: 1.8535, R2: 0.383459340672459                
        #x = model_c1 * 0.35 + model_c2 * 0.35 +  cf * 0.3  # 11.8057,  MAE: 1.9011, R2: 0.35941563047766467   7582
        #x = model_c1 * 0.40 + model_c2 * 0.40 +  cf * 0.2  # Pred MSE: 11.5322,  MAE: 1.8948, R2: 0.37425920186845196   7672    
        
        output = Dense(1, activation='linear')(x)
        self.model = Model(inputs=inputs, outputs=output)
        return self.model
    

def process_data_file(file_to_load):
    
    print(f"Loading {file_to_load}" )
    df = pd.read_csv(file_to_load)
    
    f_list_f = ['SDBB91', 'COMP2', 'COMP3', 'ATR5', 'TV3', 'HourOfDay', 'TV1', 'ZH79X', 'SDKC29C', 
                        'ZL57X', 'COMP0', 'ATR2', 'TV6', 'RSI14', 'RSI9', 'ATR51' ,'output','outputC']   

    f_list_r = ['RSI9', 'ATR2', 'ATR5', 'ATR51', 'RSI14', 'TV3', 'TV6', 'COMP2', 'SDKC29C', 'COMP3', 'output','outputC']
    
    f_list_rx = ['RSI9', 'ATR2', 'ATR5', 'RSI14', 'TV3', 'TV6', 'COMP2', 'output','outputC']
    
    f_list_c = ['RSI9', 'ATR2', 'RSI14', 'TV6', 'TV1', 'ZL57X', 'COMP0', 'HourOfDay', 'SDBB91', 'ZH79X', 'output','outputC']

    f_list_cx = ['RSI9', 'ATR2', 'RSI14', 'TV6', 'TV1', 'ZL57X', 'COMP0', 'ZH79X', 'output','outputC']

    col_filter = f_list_rx
        
    df = df.drop(columns=['TimeTicks','SeqClose'])
    df=df[col_filter]
        
    #df = df[((df['RSI'] > 20) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 80)]  
    #df = df[((df['RSI'] > 25) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 75)] 
    #df = df[(df['RSI9'] > 60)]  
    #df = df[(df['RSI9'] < 40)]  
    
    X = df.drop(columns=['output', 'outputC']).values
    y = df['output'].values

    return X, y

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

    X_data, y_data = process_data_file(datafile[file_train])
    X_oos, y_oos = process_data_file(datafile[file_oos])
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_three_ways(X_data, y_data)
    input_shape = (X_train.shape[1], 1)
    #(14, 1)


    model_new = MODEL_NEW()
    history_out, y_pred = model_new.train_model(input_shape, X_train, X_test, y_train, y_test, X_val, y_val, 10 )

    # Load best model and evaluate
    best_model = tf.keras.models.load_model(model_new.checkpoint_model)
    y_pred = best_model.predict(X_test)

    oos_model = tf.keras.models.load_model(model_new.checkpoint_model)
    y_pred_oos = oos_model.predict(X_oos)
    
    print("-----")
    print(f"Train Shape {X_train.shape}")
    print(f"Val Shape   {X_val.shape}")
    print(f"Test Shape  {X_test.shape}")
    print("-----")
    print("Test Pred")
    rmse, mse, mae, r2 = model_new.evaluate_model( y_test, y_pred)
    
    print("-----")
    print(f"OOS Test Shape {X_oos.shape}")
    print("-----")
    print("OOS Pred")
    rmse_oos, mse_oos, mae_oos, r2_oos = model_new.evaluate_model( y_oos, y_pred_oos)

    model_file = f"model_new_{mse_oos}_{mae_oos}_{r2_oos}_model.keras"
    oos_file_path = os.path.join(model_new.checkpoint_dir, model_file)
    oos_model.save(oos_file_path)
    
    oos_model_plot_file = f"model_new_{mse_oos}_{mae_oos}_{r2_oos}_model.png"
    oos_model_plot_path = os.path.join(model_new.checkpoint_dir, oos_model_plot_file)
    
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