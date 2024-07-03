import pandas as pd
import numpy as np
import tensorflow as tf
import random

from tensorflow.keras.layers import Input, LSTM, Concatenate, Reshape, Flatten, Dense, LeakyReLU, Dropout, MultiHeadAttention
from tensorflow.keras.layers import   BatchNormalization, Layer,  Attention, Bidirectional, TimeDistributed, Conv1D, Conv2D, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import MeanSquaredError, BinaryCrossentropy, BinaryAccuracy, AUC  
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

tf.config.set_visible_devices([], 'GPU')

def set_seeds(seed=42):
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

# Function to create sequences
def create_sequences(data_in, seq_length_in):
    print("create_sequences")
    xs = [data_in.iloc[i:i + seq_length_in, :-1].values for i in range(len(data_in) - seq_length_in)]
    ys = data_in.iloc[seq_length_in:, -1].values
    return np.array(xs), np.array(ys)

# Function to normalize sequences
def normalize_sequences(sequences_in):
    print("normalize_sequences")
    scalers_out = {}
    for i in range(sequences_in.shape[0]):
        #scalers_out[i] = MinMaxScaler((-1,1))
        #scalers_out[i] = MinMaxScaler((0,1))
        scalers_out[i] = MinMaxScaler()
        #scalers_out[i] = StandardScaler() 
        sequences_in[i] = scalers_out[i].fit_transform(sequences_in[i])
    return sequences_in, scalers_out

# Function to reverse scaling
def reverse_scaling(preds_in, scalers_in, seq_length_in, feature_dim_in):
    reversed_preds = []
    for i in range(len(preds_in)):
        temp_input = np.zeros((seq_length_in, feature_dim_in))
        temp_input[:, -1] = preds_in[i]
        reversed_pred = scalers_in[i].inverse_transform(temp_input)
        reversed_preds.append(reversed_pred[0, -1])
    return np.array(reversed_preds)


def sequence_and_normalize(data_in, seq_length_in, split_in):
    feature_dim = data_in.shape[1] - 1
    X, y = create_sequences(data_in, seq_length_in)
    X, scalers = normalize_sequences(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_in, random_state=42)
    
    return feature_dim, scalers, X_train, X_val, y_train, y_val

# Transformer-based Model
def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(num_heads=8, key_dim=input_shape[-1])(x, x)
    x = Dropout(0.1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model


def build_modelY(input_dim,  lay1, lay2, lay3, sequence_length):
    
    init = RandomNormal(stddev=0.02)
    dropout_rate=0.5
    
    input_layer = Input(shape=(sequence_length, input_dim))
    print("Input Shape:", input_layer.shape)

    reshaped_input = Reshape((sequence_length, input_dim, 1))(input_layer)
    conv1 = TimeDistributed(Conv1D(filters=32, kernel_size=7, activation='relu', padding='same'))(reshaped_input)
    conv1 = Flatten()(conv1)  
    conv1 = Reshape((sequence_length, -1))(conv1)

    x = LSTM(lay1, return_sequences=True, kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.l2(0.01))(conv1)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
       
    x = MultiHeadAttention(num_heads=2, key_dim=15,kernel_regularizer=tf.keras.regularizers.l2(0.01))(x, x)
    x = Dropout(dropout_rate)(x)
   
    #x = LSTM(lay2, return_sequences=True, kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Bidirectional(LSTM(lay2,return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)))(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Attention()([x, x])    
    
    x = LSTM(lay3, return_sequences=False, kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
        
    #x = Dense(1, activation='tanh')(x)
    x = Dense(1)(x)
    return Model(input_layer, x)


def build_modelX(input_dim,  lay1, lay2, lay3, sequence_length):
   
    init = RandomNormal(stddev=0.02)
   
    dropout_rate=0.5
    
    input_layer = Input(shape=(sequence_length, input_dim))
    x = LSTM(lay1, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_layer)
    x = Dropout(dropout_rate)(x)
    x = LSTM(lay2, return_sequences=True)(x)
    x = Dropout(0.2)(x) 
    x = LSTM(lay3, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.2)(x) 
    x = Dense(lay3//2, kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

    return Model(input_layer, x)

# CNN Model
def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1))
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model

# Ensemble Technique
def ensemble_predictions(preds_transformer, preds_cnn, preds_lstm, preds_lstm_2, weights):
    min_length = min(len(preds_transformer), len(preds_cnn), len(preds_lstm), len(preds_lstm_2))
    preds_transformer = preds_transformer[:min_length]
    preds_cnn = preds_cnn[:min_length]
    preds_lstm = preds_lstm[:min_length]
    preds_lstm_2 = preds_lstm_2[:min_length]
    combined_preds = (weights[0] * preds_transformer + weights[1] * preds_cnn + weights[2] * preds_lstm + weights[3] * preds_lstm_2) / sum(weights)
    return combined_preds

# Evaluation
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse


def run():
    set_seeds(42)
    
    data = pd.read_csv('data/buildSeqInd_Lucky13_5M_ALL.csv')
    data = data.drop(columns=['outputC'])

    drop_cols = [
        #'STOK1',
        #'RSI',
        #'ATR2',
        'ATR21',
        #'ATR3',
        'ATR31', 
        'ATR32',
        'ATR34',   
        #'ROC',     
        'SDKC9',   
        'SDKC91',  
        'SDBB91',  
        #'SDLR310'
    ]
    data = data.drop(columns=drop_cols)

    # Set sequence length
    seq_length = 25
    split = 0.2

    feature_dim_out, scalers_out, X_train, X_test, y_train, y_test = sequence_and_normalize(data, seq_length, split)

    # Print shapes to verify
    print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    lstm_model_2 = build_modelY(feature_dim_out,  8, 64, 32, seq_length)
    optimizer_2 = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.05)
    r_metrics_2 = ['MeanSquaredError','BinaryAccuracy', 'AUC']
    lstm_model_2.compile(optimizer=optimizer_2, loss='mse',metrics=r_metrics_2)
    lstm_model_2.summary()
    history_lstm_2 = lstm_model_2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping])


    lstm_model = build_modelX(feature_dim_out,  8, 64, 32, seq_length)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.05)
    r_metrics = ['MeanSquaredError','BinaryAccuracy', 'AUC']
    lstm_model.compile(optimizer=optimizer, loss='mse',metrics=r_metrics)
    lstm_model.summary()
    history_lstm = lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping])

    # Train Transformer Model
    transformer_model = build_transformer_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    history_transformer = transformer_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=2)

    # Train CNN Model
    cnn_model = build_cnn_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    history_cnn = cnn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=2)

    # Make predictions
    preds_transformer = transformer_model.predict(X_test).reshape(-1)
    preds_cnn = cnn_model.predict(X_test).reshape(-1)
    preds_lstm = lstm_model.predict(X_test).reshape(-1)
    preds_lstm_2 = lstm_model_2.predict(X_test).reshape(-1)

    # Ensure predictions are of the same length
    min_length = min(len(preds_transformer), len(preds_cnn), len(preds_lstm), len(y_test))
    preds_transformer = preds_transformer[:min_length]
    preds_cnn = preds_cnn[:min_length]
    preds_lstm = preds_lstm[:min_length]
    preds_lstm_2 = preds_lstm_2[:min_length]
    y_test = y_test[:min_length]

    # Ensemble predictions
    ensemble_preds = ensemble_predictions(preds_transformer, preds_cnn, preds_lstm, preds_lstm_2, weights=[0.4, 0.3, 0.3, 0.2 ])

    # Evaluate the ensemble model
    mse_ensemble = evaluate_model(y_test, ensemble_preds)
    print(f'Ensemble MSE: {mse_ensemble:.4f}')

    # Plot training history
    def plot_history(history, model_name):
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'{model_name} Training and Validation Loss')
        plt.legend()
        plt.show()

    plot_history(history_transformer, 'Transformer Model')
    plot_history(history_cnn, 'CNN Model')
    plot_history(history_lstm, 'LSTM Model')
    plot_history(history_lstm_2, 'LSTM Model')

if __name__ == "__main__":
    run()