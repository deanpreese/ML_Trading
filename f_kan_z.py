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
from tensorflow.keras.layers import Input, Conv1D, Average, Conv2D, LeakyReLU, Reshape, Concatenate, Multiply 
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

tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)


class F_KAN_Z:
    def __init__(self, epochs=50, batch_size=32):
        
        self.checkpoint_dir = 'checkpoints/'
        self.trained_dir = 'trained_models/'
       
        self.checkpoint_model = os.path.join(self.checkpoint_dir, 'f_kan_z_model.keras')
        self.trained_model = os.path.join(self.trained_dir, 'f_kan_z_model.keras')
        self.model_plot = os.path.join(self.checkpoint_dir, 'f_kan_z_model.png')

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
        
        inx = LSTM(32, return_sequences=True, activation='relu')(inputs)
        #inx = Dropout(self.drop_out)(inx)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
       # x = Dropout(self.drop_out)(x)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        #x = Dropout(self.drop_out)(x)
        x = MaxPooling1D(pool_size=1, strides=1)(x)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        #x = Dropout(self.drop_out)(x)
        c1_out = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg, name="c1_out", kernel_initializer=self.initializer)(x) 
        
        return c1_out


    def create_feature_model_c2(self, inputs, output_dim):
        
        x = LSTM(32, return_sequences=True, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        c2_out = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg, name="c2_out", kernel_initializer=self.initializer)(x) 

        return c2_out    

    def create_feature_model_a(self, input_shape):
        
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)
        inx = LSTM(32, return_sequences=True, activation='relu')(reshaped_inputs)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        smx_out = Dense(1, activation='linear')(x) 
        subx_model = Model(input, smx_out)
        return subx_model

    def create_feature_model_s(self, input_shape):
        
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)
        inx = LSTM(32, return_sequences=True, activation='relu')(reshaped_inputs)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        smx_out = Dense(1, activation='sigmoid')(x) 
        subx_model = Model(input, smx_out)
        return subx_model

    def create_feature_model_t(self, input_shape):
        
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
        feature_outputs = []
        feature_outputs1 = []
        feature_outputs2 = []
        
        att_dim = 64
        concat_dims = 32
        output_dim = 16
        
        model_a = self.create_feature_model_h(inputs, output_dim)
        model_b = self.create_feature_model_x(inputs, output_dim)
        model_c1 = self.create_feature_model_c1(inputs, output_dim)
        model_c2 = self.create_feature_model_c2(inputs, output_dim)
        
        for i in range(input_shape[0]):
            feature_input = inputs[:, i:i+1]

            #if ((i > 10) | (i==4)) : 
            if (i > -1) : 
                fa = self.create_feature_model_a((1,))
                fax = fa(feature_input)
                feature_outputs.append(fax)
                
                fm1 = self.create_feature_model_s((1,))
                fm1x = fm1(feature_input)
                feature_outputs1.append(fm1x)
                
                fm2 = self.create_feature_model_t((1,))
                fm2x = fm2(feature_input)
                feature_outputs2.append(fm2x)
       
        concatenated_outputs = Concatenate(axis=1)(feature_outputs)
        concatenated_outputs1 = Concatenate(axis=1)(feature_outputs1)
        concatenated_outputs2 = Concatenate(axis=1)(feature_outputs2)
        
        xa_out = Dense(concat_dims, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(concatenated_outputs)
        xs_out = Dense(concat_dims, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(concatenated_outputs1)
        xt_out = Dense(concat_dims, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(concatenated_outputs2)
        
        xa = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(xa_out)
        xs = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(xs_out)
        xt = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(xt_out)

        xa1 = Dense(output_dim, activation='softmax', kernel_initializer=self.initializer, name='attention_xa')(xa)
        x1 = Multiply()([xa1, xa])
        xs1 = Dense(output_dim, activation='softmax', kernel_initializer=self.initializer, name='attention_xs')(xs)
        x2 = Multiply()([xs1, xs])
        xt1 = Dense(output_dim, activation='softmax', kernel_initializer=self.initializer, name='attention_xt')(xt)
        x3 = Multiply()([xt1, xt])

        xc_ta = Multiply()([xt1, xa])
        xc_sa = Multiply()([xs1, xa])
        xc_ts = Multiply()([xt1, xs1])

        #s_ave_output = Average()([xa, xs, xt, x1,x2,x3])
        #s_ave_output = Average()([xa, xs, xt, x1,x2,x3])
        s_ave_output = Average(name='final_average')([xa, xs, xt, x1,x2,x3, xc_ta, xc_sa, xc_ts])
        x = s_ave_output
        
        #ave_output = Average()([model_a, model_b, s_ave_output])
        #ave_output = Average()([model_a, model_b, model_c1, model_c2, s_ave_output])
        #xa = Average()([model_a, model_b, model_c1, model_c2, ave_output])
        
        output = Dense(1, activation='linear')(x)
        
        self.model = Model(inputs=inputs, outputs=output)
        return self.model
    

    def train_model(self, input_shape, X_train, X_test, y_train, y_test,  X_val, y_val, epocs ):
        
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
            mode="auto", min_delta=0.000001, cooldown=0, min_lr=0,
            )

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_model, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min'
            )
        
        history_out = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), 
            initial_epoch=0, epochs=epocs, verbose=1, batch_size=64, 
            callbacks=[early_stopping, reduce_lr, model_checkpoint]
            )      
        
        y_pred = self.model.predict(X_test)
        return history_out, y_pred



def evaluate_model( y_test, y_pred):
    
    correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(y_test, y_pred)
    print(" ")
    print(f"Pred MSE: {mse},  MAE: {mae}, R2: {r2}")
    print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.4f}")
    print(f"Number of Samples: {total}")        
    print(" ")        
    
    return rmse, mse, mae, r2
    
def combined_plots(history, y_true, y_pred):
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



def main():
        
    np.random.seed(42)
    tf.random.set_seed(42)
    
    datafile = [ 
            'data/NewModel_3070_oos.csv',   
            'data/NewModel_3070.csv',  #1
            'data/NewModel_ALL_oos.csv',   
            'data/NewModel_ALL.csv',  #3
            'data/NewModel_span3_3070_oos.csv',   
            'data/NewModel_span3_3070.csv',  #5
            
    ]   

    file_train = 1
    file_oos = 0

    best_model_path = ""

    #Lucky13  ALL Cols
    f_13 = ['SDLR310','SDBB91','SDKC91','SDKC9','ROC','ATR54','ATR53','ATR52','ATR51','ATR5','ATR21','ATR2','RSI','STOK1','output','outputC']

    f_list_f = ['SDBB91', 'COMP2', 'COMP3', 'ATR5', 'TV3', 'HourOfDay', 'TV1', 'ZH79X', 'SDKC29C', 
        'ZL57X', 'COMP0', 'ATR2', 'TV6', 'RSI14', 'RSI9', 'ATR51' ,'output','outputC']   

    f_list_r = ['RSI9', 'ATR2', 'ATR5', 'ATR51', 'RSI14', 'TV3', 'TV6', 'COMP2', 'SDKC29C', 'COMP3', 'output','outputC']
    f_list_c = ['RSI9', 'ATR2', 'RSI14', 'TV6', 'TV1', 'ZL57X', 'COMP0', 'HourOfDay', 'SDBB91', 'ZH79X', 'output','outputC']

    f_list_uni = [
                'RSI9', 'RSI14', 'COMP2', 'COMP1', 'COMP0', 'TV5', 'TV6', 
                'RSI91', 'ROC9', 'COMP3', 'STOK5133', 'ROC7', 'STOK714Y', 
                'ROC14', 'RSI141', 'TV2', 'TV3', 'ROC141', 'ROC91', 'TV1', 
                'output','outputC']   

    col_filter = f_list_uni


    file_path = datafile[file_train]
    print(f"Loading {file_path}" )
    data = pd.read_csv(file_path)
    df = data.drop(columns=['TimeTicks','SeqClose'])

    df=df[col_filter]
    
    #df = df[((df['RSI'] > 20) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 80)]  
    #df = df[((df['RSI'] > 25) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 75)] 
    #df = df[(df['RSI9'] > 60)]  
    #df = df[(df['RSI'] < 40)]  
    
    X = df.drop(columns=['output', 'outputC']).values
    y = df['output'].values

    X_train, X_val, X_test, y_train, y_val, y_test = split_three_ways(X, y)
    input_shape = (X_train.shape[1], 1)
    #(14, 1)

    c_kan = F_KAN_Z()
    history_out, y_pred = c_kan.train_model(input_shape, X_train, X_test, y_train, y_test, X_val, y_val, 1000 )

    # Load best model and evaluate
    best_model = tf.keras.models.load_model(c_kan.checkpoint_model)
    y_pred = best_model.predict(X_test)
    rmse, mse, mae, r2 = evaluate_model( y_test, y_pred)
    
    model_file = f"f_kan_z_{mse}_{mae}_{r2}_model.keras"
    file_path = os.path.join(c_kan.checkpoint_dir, model_file)
    best_model.save(file_path)
    
    best_model_path = file_path
    
    plot_file = f"f_kan_z_{mse}_{mae}_{r2}_model.png"
    plot_path = os.path.join(c_kan.checkpoint_dir, plot_file)
    tf.keras.utils.plot_model(best_model, to_file=plot_path, 
        show_shapes=True, 
        show_dtype=True,
        show_layer_names=True,
        expand_nested=True,
        show_layer_activations=True,
        show_trainable=True
    )   
    
    file_path = datafile[file_oos]
    df = pd.read_csv(file_path)
    df = df.drop(columns=['TimeTicks','SeqClose'])
        
    #df = df[((df['RSI'] > 20) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 80)]  
    #df = df[((df['RSI'] > 25) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 75)] 
    #df = df[(df['RSI'] > 60)]  
    #df = df[(df['RSI'] < 40)]  
    
    
    df = data[col_filter]
    
    X = df.drop(columns=['output','outputC']).values
    y = df['output'].values 
        
    oos_model = tf.keras.models.load_model(best_model_path)
    #X_test = sc_temp.fit(X)
    X_test = X
    y_test = y
    
    y_pred = oos_model.predict(X_test)
    evaluate_model( y_test, y_pred)
            
if __name__ == "__main__":
    main()
    