import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, Dense, LayerNormalization, MultiHeadAttention, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

tf.config.set_visible_devices([], 'GPU')

def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:i+sequence_length, :-1]  # all features except output
        y = data[i+sequence_length, -1]  # output
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def prepare_data(filepath, sequence_length):
    data = pd.read_csv(filepath)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    train_size = int(len(data_scaled) * 0.8)
    train_data = data_scaled[:train_size]
    test_data = data_scaled[train_size:]
    
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)
    
    return X_train, y_train, X_test, y_test, scaler

# TFT Model Design
def tft_model(sequence_length, num_features):
    inputs = Input(shape=(sequence_length, num_features))
    
    # LSTM Encoder
    lstm_out = LSTM(128, return_sequences=True)(inputs)
    lstm_out = LSTM(128, return_sequences=True)(lstm_out)
    
    # Temporal Attention Layer
    attention = MultiHeadAttention(num_heads=4, key_dim=128)(lstm_out, lstm_out)
    attention = Add()([attention, lstm_out])
    attention = LayerNormalization()(attention)
    
    # Position-wise Feed-Forward Networks
    dense = Dense(256, activation='relu')(attention)
    dense = Dense(128, activation='relu')(dense)
    
    # Output Layer
    output = Dense(1)(dense[:, -1, :])  # Predicting the output for the last time step
    
    model = Model(inputs, output)
    return model

# Model Training
def train_model(X_train, y_train, X_val, y_val, sequence_length, num_features):
    model = tft_model(sequence_length, num_features)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    return model, history

# Evaluation and Plotting
def evaluate_model(model, X_test, y_test):
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')
    
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_predictions(model, X_test, y_test, scaler, num_features):
    predictions = model.predict(X_test)
    
    # Prepare data for inverse scaling
    test_data_with_predictions = np.concatenate((X_test[:, -1, :], predictions), axis=1)
    y_test_with_features = np.concatenate((X_test[:, -1, :], y_test.reshape(-1, 1)), axis=1)
    
    predictions_rescaled = scaler.inverse_transform(test_data_with_predictions)[:, -1]
    y_test_rescaled = scaler.inverse_transform(y_test_with_features)[:, -1]
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_rescaled, label='Actual')
    plt.plot(predictions_rescaled, label='Predicted')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.show()


# Main Function
def main():
    
    datafile = [ 
                'data/Lucky13_3070_oos.csv',   
                'data/Lucky13_3070.csv',  #1
                'data/ndata_diff_lucky13_3070_oos.csv', 
                'data/ndata_diff_lucky13_3070.csv', #3
                'data/ndata_lucky_13_lag_3070_oos.csv', 
                'data/ndata_lucky13_lag_3070.csv', #5
                'new_model_Z_lucky13_3070_oos.csv',
                'new_model_Z_lucky13_3070.csv' #7,
                'data/Lucky13_3070_oos_3.csv',   
                'data/Lucky13_3070_3.csv',  #8
                'data/Lucky13_3070_oos_5.csv',   
                'data/Lucky13_3070_5.csv',  #10

        ]
    #dtx = pd.read_csv(datafile[10])
    sequence_length = 7
   
    # Prepare data
    X_train, y_train, X_test, y_test, scaler = prepare_data(datafile[1], sequence_length)
    
    # Split validation data from training data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    
    num_features = X_train.shape[2]
    
    # Train model
    model, history = train_model(X_train, y_train, X_val, y_val, sequence_length, num_features)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Plot predictions
    plot_predictions(model, X_test, y_test, scaler, num_features)

if __name__ == "__main__":
    main()