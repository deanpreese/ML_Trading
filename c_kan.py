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
from tensorflow.keras.layers import Input, Conv1D, Average, Conv2D, LeakyReLU, Reshape, Concatenate, Multiply, BatchNormalization, Bidirectional, Add, Dense, Dropout, MaxPooling1D, LSTM, MultiHeadAttention, Attention
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


class C_KANX:
    def __init__(self, epochs=50, batch_size=32):
        
        self.checkpoint_dir = 'checkpoints/'
        self.trained_dir = 'trained_models/'
       
        self.checkpoint_model = os.path.join(self.checkpoint_dir, 'c_kan_model.keras')
        self.trained_model = os.path.join(self.trained_dir, 'c_kan_model.keras')
        self.model_plot = os.path.join(self.checkpoint_dir, 'c_kan_model.png')

        self.drop_out = 0.2
        self.l2_reg = l2(0.01)
        self.initializer = GlorotUniform(seed=42)


    def create_feature_model(self, input_shape):
        
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)
        inx = LSTM(32, return_sequences=True, activation='relu')(reshaped_inputs)
        #inx = Dropout(self.drop_out)(inx)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
       # x = Dropout(self.drop_out)(x)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        #x = Dropout(self.drop_out)(x)
        x = MaxPooling1D(pool_size=1, strides=1)(x)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        #x = Dropout(self.drop_out)(x)
        smx_out = Dense(1, activation='linear')(x) 
        subx_model = Model(input, smx_out)
        return subx_model


    def create_feature_model2(self, input_shape):
        
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)
        x = LSTM(32, return_sequences=True, activation='relu')(reshaped_inputs)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        smx_out = Dense(1, activation='linear')(x) 
        subx_model = Model(input, smx_out)
        return subx_model

   
    def create_model(self, input_shape ):
        
        inputs = Input(shape=input_shape)
        feature_outputs = []
        
        for i in range(input_shape[0]):
            feature_input = inputs[:, i:i+1]
            
            if i % 2 == 0:
                fm = self.create_feature_model((1,))
                fmx = fm(feature_input)
                feature_outputs.append(fmx)
                
                fm2 = self.create_feature_model2((1,))
                fm2x = fm2(feature_input)
                #feature_outputs.append(fm2x)
                
            else:
                fm = self.create_feature_model((1,))
                fmx = fm(feature_input)
                feature_outputs.append(fmx)
                
                fm2 = self.create_feature_model2((1,))
                fm2x = fm2(feature_input)
                #feature_outputs.append(fm2x)
                
       
        concatenated_outputs = Concatenate(axis=1)(feature_outputs)
        rs = Reshape((len(feature_outputs), 1))(concatenated_outputs)
        x = Dense(len(feature_outputs), activation='softmax', kernel_initializer=self.initializer, name='attention')(rs)
        x = Multiply()([rs, x])
        
        x = Dense(32, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(x)
    
        #attention2 = Dense(32, activation='softmax', kernel_initializer=self.initializer, name='attention2')(weighted)
        #weighted = Multiply()([weighted, attention2])
        
        #x = Conv1D(32, 2, activation='relu', kernel_initializer=self.initializer)(x)
        #x = LSTM(32, kernel_regularizer=self.l2_reg, activation='relu', return_sequences=False, kernel_initializer=self.initializer)(x)
        #x = Bidirectional(LSTM(32,name="BIC", kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(x)
        
        x = Reshape((-1,))(x)
        aggregated = Dense(64, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(x)
        aggregated = Dropout(self.drop_out)(aggregated)
        
        output = Dense(1, activation='linear')(aggregated)
        
        self.model = Model(inputs=inputs, outputs=output)
        return self.model

    

    def train_model(self, input_shape, X_train, X_test, y_train, y_test,  X_val, y_val ):
        
        self.create_model(input_shape )

        self.model.compile(optimizer=Adam(learning_rate=0.001), 
                loss='mse', metrics=['mae', tf.keras.metrics.R2Score()])
        self.model.summary()
        
        tf.keras.utils.plot_model(self.model, to_file=self.model_plot, 
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
                    verbose=0,
                    save_best_only=True, 
                        save_weights_only=False, mode='min')
        
        
        history_out = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                                initial_epoch=0, epochs=2000, verbose=1,
                                batch_size=64, callbacks=[
                                    early_stopping,
                                    reduce_lr,
                                    model_checkpoint])      
        
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
    
    datafile = [ 
        'data/Lucky13_3070_oos.csv',   
        'data/Lucky13_3070.csv',  #1
        'data/Lucky13_EX_3070_oos.csv',  
        'data/Lucky13_EX_3070.csv',  #3
        'data/Lucky13_ALL_oos.csv',  #4
        'data/Lucky13_ALL.csv',  #5
    ]


    train = True
    run_oos = False

    sc_temp = None

    if train:

        for i in range(1):

            np.random.seed(42)
            tf.random.set_seed(42)

            file_path = datafile[1]

            print(f"Loading {file_path}" )
            df = pd.read_csv(file_path)
            df = df.drop(columns=['outputC'])
                
            #Lucky13  ALL Cols
            f_13 = ['SDLR310','SDBB91','SDKC91','SDKC9','ROC','ATR54','ATR53','ATR52','ATR51','ATR5','ATR21','ATR2','RSI','STOK1','output','outputC']

            #df = df[(df['RSI'] > 60) & (df['RSI'] < 80)]  #  81%
            #df = df[(df['RSI'] > 60) & (df['RSI'] < 75)]   # 878%
            #df = df[(df['RSI'] > 20) & (df['RSI'] < 40)]  # 84%
            #df = df[(df['RSI'] > 25) & (df['RSI'] < 40)]  # 91%
            df = df[((df['RSI'] > 20) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSIseq'] < 80)]  
            
            X = df.drop(columns=['output']).values
            y = df['output'].values

            X_train, X_val, X_test, y_train, y_val, y_test = split_three_ways(X, y)

            #scaler = StandardScaler()
            #X_train = scaler.fit_transform(X_train)
            #X_val = scaler.transform(X_val)
            #X_test = scaler.transform(X_test)

            input_shape = (X_train.shape[1], 1)
            #(14, 1)

            c_kan = C_KANX()
            history_out, y_pred = c_kan.train_model(input_shape, X_train, X_test, y_train, y_test, X_val, y_val )

            # Load best model and evaluate
            best_model = tf.keras.models.load_model(c_kan.checkpoint_model)
            y_pred = best_model.predict(X_test)
            rmse, mse, mae, r2 = evaluate_model( y_test, y_pred)
            
            model_file = f"c_kan_{mse}_{mae}_model.keras"
            file_path = os.path.join(c_kan.checkpoint_dir, model_file)
            best_model.save(file_path)
            
            #sc_temp = scaler
            #scaler_file = f"c_kan_scaler_{mse}_{mae}_model.sc"
            #sc_file_path = os.path.join(c_kan.checkpoint_dir, scaler_file)
            #joblib.dump(scaler, sc_file_path) 
            
        
    if run_oos:
        file_path = datafile[0]
        df = pd.read_csv(file_path)
        df = df.drop(columns=['outputC'])
        X = df.drop(columns=['output']).values
        y = df['output'].values 
            
        oos_model = tf.keras.models.load_model(c_kan.checkpoint_model)
        X_test = sc_temp.transform(X)
        y_test = y
        
        y_pred = oos_model.predict(X_test)
        evaluate_model( y_test, y_pred)
            


if __name__ == "__main__":
    main()
    