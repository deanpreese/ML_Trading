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

from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error,r2_score, root_mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score



tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)


class FEAT_KAN:
    def __init__(self, epochs=50, batch_size=32):
        
        self.checkpoint_dir = 'checkpoints/'
        self.trained_dir = 'trained_models/'
       
        self.checkpoint_model = os.path.join(self.checkpoint_dir, 'f_kan_z_model.keras')
        self.trained_model = os.path.join(self.trained_dir, 'f_kan_z_model.keras')
        self.model_plot = os.path.join(self.checkpoint_dir, 'f_kan_z_model.png')

        self.drop_out = 0.2
        self.l2_reg = l2(0.01)
        self.initializer = GlorotUniform(seed=42)



    def create_feature_model_v1(self, input_shape):
        
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)
        inx = LSTM(32, return_sequences=True, activation='relu')(reshaped_inputs)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
        #x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        smx_out = Dense(8, activation='relu')(x) 
        subx_model = Model(input, smx_out)
        return subx_model

    def create_feature_model_v2(self, input_shape):
        
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)
        inx = LSTM(32, return_sequences=True, activation='relu')(reshaped_inputs)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        x = LSTM(32, return_sequences=False, activation='relu')(x)
        smx_out = Dense(8, activation='relu')(x) 
        subx_model = Model(input, smx_out)
        return subx_model


   
    def create_model(self, input_shape ):
        
        inputs = Input(shape=input_shape)
        feature_outputs = []
        
        concat_dims = 64
        output_dim = 16
        
        for i in range(input_shape[0]):
            feature_input = inputs[:, i:i+1]
            fa = self.create_feature_model_v2((1,))
            fax = fa(feature_input)
            feature_outputs.append(fax)
       
        concatenated_outputs = Concatenate(axis=1)(feature_outputs)
        
        xa = Dense(concat_dims, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(concatenated_outputs)
        #xa = Dense(output_dim, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(xa)
        
        output = Dense(1, activation='linear')(xa)
        self.model = Model(inputs=inputs, outputs=output)
        return self.model
    

    def train_model(self, input_shape, X_train, X_test, y_train, y_test,  X_val, y_val, epocs ):
        
        self.create_model(input_shape )

        self.model.compile(optimizer=Adam(learning_rate=0.001), 
                loss='mse', metrics=['mae', tf.keras.metrics.R2Score()])
        self.model.summary()
        
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, verbose=1,
            mode="auto", min_delta=0.000001, cooldown=0, min_lr=0,
            )

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
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
    


def main():
    
    datafile = [ 
                'data/Lucky13_3070_oos.csv',   
                'data/Lucky13_3070.csv',  #1
                'data/Model_M_1_3070_oos.csv', 
                'data/Model_M_1_3070.csv' #3
        ] 
    
    file_train = 3


    np.random.seed(42)
    tf.random.set_seed(42)

    file_path = datafile[file_train]

    lucky_13_columns = [
        "SDLR310", "SDBB91", "SDKC91", "SDKC9", "ROC", "ATR54", "ATR53", "ATR52", 
        "ATR51", "ATR5", "ATR21", "ATR2", "RSI", "STOK1", "output", "outputC"
    ]
    
    model_m_1_all = [
    #"Year", "Month", "Day", "DayOfWeek", "HourOfDay", "MinOfHour",
    #"SeqClose", "RSIRAW", 
    "SDBB9L", "SDBB9U", "SDKC7U", "SDKC7L",
    "ROC14", "ROC9", "ATR5", "ATR2", "RSI14", "RSI14Avg", "RSIH14", "RSIL14",
    "RSI9", "RSI9Avg", "RSIH9", "RSIL9", "STOK721", "STOK513", "STOK721D", "STOK513D",
    "TV1", "TV2", "TV3", "TV4", "TV5", "TV6", "COMP0", "COMP1", "COMP2", "COMP3"
    ]
    

    column_results = []
    #column_list = data_13_x
    column_list = model_m_1_all
    
    for i in range(len(column_list)):
        
        print(f"Loading {file_path}" )
        df = pd.read_csv(file_path)
        y = df['output']
        
        print(column_list[i])
        X = df[[column_list[i]]]
    
        X_train, X_val, X_test, y_train, y_val, y_test = split_three_ways(X, y)

        input_shape = (X_train.shape[1], 1)
        #(14, 1)

        print(X_train.shape)
        print(X_test.shape)
        print(X_val.shape)
        
        epocs = 10
    
        c_kan = FEAT_KAN()
        history_out, y_pred = c_kan.train_model(input_shape, X_train, X_test, y_train, y_test, X_val, y_val, epocs )

        # Load best model and evaluate
        best_model = tf.keras.models.load_model(c_kan.checkpoint_model)
        y_pred = best_model.predict(X_test)
        
        rmse = root_mean_squared_error(y_test, y_pred)
        mse = rmse **2.0
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results_out = f"{column_list[i]}   {rmse}   {mse}  {mae}  {r2}"
        print(results_out)
        
        column_results.append(results_out)


    with open('column_values.txt', 'a') as f:
        for param in column_results:
            f.write(f"{param}\n")
    
if __name__ == "__main__":
    main()
    


"""
model M 1

Ind	RMSE	MSE	MAE	R2
RSI9	2.600893798	6.764648549	1.45318178	0.2474
COMP0	2.616377132	6.845429298	1.459901933	0.2384
COMP1	2.6173631	6.850589596	1.467248789	0.2379
COMP2	2.623929718	6.885007167	1.450373155	0.2340
RSI14	2.64027553	6.971054874	1.455140136	0.2245
RSI9Avg	2.650076691	7.022906468	1.459279378	0.2187
RSIH9	2.690845264	7.240648237	1.533185889	0.1945
RSIL9	2.692056358	7.247167433	1.499268698	0.1938
TV5	2.703086822	7.306678365	1.513283643	0.1871
TV6	2.70425494	7.312994781	1.478992586	0.1864
RSIH14	2.704279713	7.313128768	1.499084087	0.1864
RSIL14	2.71083781	7.348641631	1.486517786	0.1825
TV2	2.73916742	7.503038156	1.509466276	0.1653
COMP3	2.740339925	7.509462903	1.507601961	0.1646
TV3	2.749168813	7.557929163	1.517337792	0.1592
STOK513	2.769937941	7.672556199	1.491499561	0.1464
TV1	2.771054663	7.678743946	1.514862054	0.1457
RSI14Avg	2.77703421	7.711919003	1.526581432	0.1420
STOK721	2.804029889	7.862583616	1.534335963	0.1253
SDKC7L	2.807741671	7.883413289	1.542681627	0.1230
SDKC7U	2.815117062	7.924884072	1.539233315	0.1184
SDBB9U	2.833062914	8.026245475	1.537378399	0.1071
SDBB9L	2.833106917	8.026494802	1.55632324	0.1071
STOK513D	2.84122359	8.07255149	1.566609683	0.1019
ROC14	2.848050354	8.111390821	1.57370402	0.0976
STOK721D	2.880839827	8.299238107	1.577234285	0.0767
ROC9	2.888512747	8.343505892	1.598751761	0.0718
TV4	2.914939315	8.496871213	1.582834815	0.0547
ATR5	2.985708133	8.914453055	1.641601483	0.0083
ATR2	2.989038755	8.934352682	1.648500993	0.0061

"""    