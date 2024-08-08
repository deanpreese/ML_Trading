import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, Dense, LayerNormalization, MultiHeadAttention, Add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Data Preparation
def create_sequences(data, target_col_index, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:i+sequence_length, :-2]  # all features except target columns
        y = data[i+sequence_length, target_col_index]  # target column
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def prepare_data(df, target_col='output', sequence_length=30):
    X = df
    features = X.drop(columns=['output', 'outputC'])
    target = df[target_col]
    
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
    scaled_df[target_col] = target.values

    train_size = int(len(scaled_df) * 0.8)
    train_data = scaled_df[:train_size]
    test_data = scaled_df[train_size:]
    
    target_col_index = scaled_df.columns.get_loc(target_col)
    
    X_train, y_train = create_sequences(train_data.values, target_col_index, sequence_length)
    X_test, y_test = create_sequences(test_data.values, target_col_index, sequence_length)
    
    return X_train, y_train, X_test, y_test, scaler

# TFT Model Design
def tft_model(sequence_length, num_features):
    inputs = Input(shape=(sequence_length, num_features))
    
    # Reduced LSTM size for performance improvement
    lstm_out = LSTM(64, return_sequences=True)(inputs)
    lstm_out = LSTM(64, return_sequences=True)(lstm_out)
    
    # MultiHeadAttention layer with reduced key_dim for performance
    attention = MultiHeadAttention(num_heads=4, key_dim=64)(lstm_out, lstm_out)
    attention = Add()([attention, lstm_out])
    attention = LayerNormalization()(attention)
    
    # Added Dropout layers for regularization
    dense = Dense(128, activation='relu')(attention)
    dense = Dropout(0.3)(dense)
    dense = Dense(64, activation='relu')(dense)
    dense = Dropout(0.3)(dense)
    
    output = Dense(1)(dense[:, -1, :])  # Predicting the target for the last time step
    
    model = Model(inputs, output)
    return model

# Model Training
def train_model(X_train, y_train, X_val, y_val, sequence_length, num_features):
    model = tft_model(sequence_length, num_features)
    model.compile(optimizer='adam', loss='mse')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    return model, history

# Evaluation and Plotting
def evaluate_model(model, X_test, y_test):
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')
    
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_predictions(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    
    y_test_rescaled = scaler.inverse_transform(np.hstack([X_test[:, -1, :], y_test.reshape(-1, 1)]))[:, -1]
    predictions_rescaled = scaler.inverse_transform(np.hstack([X_test[:, -1, :], predictions]))[:, -1]
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_rescaled, label='Actual')
    plt.plot(predictions_rescaled, label='Predicted')
    plt.title('Actual vs Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Main Function
def main():
    datafile = [
        'data/Lucky13_3070_oos.csv',
        'data/Lucky13_3070.csv',
        'data/ndata_diff_lucky13_3070_oos.csv',
        'data/ndata_diff_lucky13_3070.csv',
        'data/ndata_lucky_13_lag_3070_oos.csv',
        'data/ndata_lucky13_lag_3070.csv',
        'new_model_Z_lucky13_3070_oos.csv',
        'new_model_Z_lucky13_3070.csv',
        'data/Lucky13_3070_oos_3.csv',
        'data/Lucky13_3070_3.csv',
        'data/Lucky13_3070_oos_5.csv',
        'data/Lucky13_3070_5.csv'
    ]
    
    dtx = pd.read_csv(datafile[1])
    sequence_length = 21
    num_features = 13

    X_train, y_train, X_test, y_test, scaler = prepare_data(dtx, target_col='output', sequence_length=sequence_length)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    model, history = train_model(X_train, y_train, X_val, y_val, sequence_length, num_features)
    
    evaluate_model(model, X_test, y_test)
    
    plot_training_history(history)
    
    plot_predictions(model, X_test, y_test, scaler)

if __name__ == "__main__":
    main()
