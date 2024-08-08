import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, Dense, LayerNormalization, MultiHeadAttention, Add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
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

def tft_model(sequence_length, num_features):
    l2_reg = tf.keras.regularizers.l2(0.01)
    
    inputs = Input(shape=(sequence_length, num_features))
    
    lstm_out = LSTM(16, return_sequences=True, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg)(inputs)
    lstm_out = LSTM(32, return_sequences=True, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg)(lstm_out)
    attention = MultiHeadAttention(num_heads=4, key_dim=64, kernel_regularizer=l2_reg)(lstm_out, lstm_out)
    attention = Add()([attention, lstm_out])
    attention = LayerNormalization()(attention)
    dense = Dense(128, activation='relu', kernel_regularizer=l2_reg)(attention)
    dense = Dropout(0.3)(dense)
    dense = Dense(64, activation='relu', kernel_regularizer=l2_reg)(dense)
    dense = Dropout(0.3)(dense)
    
    output = Dense(1, kernel_regularizer=l2_reg)(dense[:, -1, :])
    
    model = Model(inputs, output)
    return model
# Model Training
def train_model(X_train, y_train, X_val, y_val, sequence_length, num_features):
    model = tft_model(sequence_length, num_features)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
    return model, history

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test).flatten()  # Flatten predictions to match y_test shape
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Calculate wins and losses
    wins = np.sum((predictions > 0) & (y_test > 0))
    losses = np.sum((predictions <= 0) & (y_test <= 0))
    
    print(f'Test MSE: {mse}')
    print(f'Test R2: {r2}')
    print(f'Wins: {wins} Losses: {losses}  Percent: {wins/(wins+losses)} ')
    
    return predictions


def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_predictions(y_test, predictions, X_test, scaler):
    # Reshape predictions to be 2D (shape: (7430, 1)) to match the shape of the features in X_test
    predictions = predictions.reshape(-1, 1)
    
    # Prepare the full array for inverse scaling
    y_test_rescaled = scaler.inverse_transform(np.hstack([np.zeros((y_test.shape[0], X_test.shape[2])), y_test.reshape(-1, 1)]))[:, -1]
    predictions_rescaled = scaler.inverse_transform(np.hstack([np.zeros((predictions.shape[0], X_test.shape[2])), predictions]))[:, -1]
    
    # Plot the results
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
    datafile = 'data/Lucky13_3070.csv'
    dtx = pd.read_csv(datafile)
    sequence_length = 7
    num_features = 13

    X_train, y_train, X_test, y_test, scaler = prepare_data(dtx, target_col='output', sequence_length=sequence_length)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    model, history = train_model(X_train, y_train, X_val, y_val, sequence_length, num_features)
    
    predictions = evaluate_model(model, X_test, y_test)
    
    plot_training_history(history)
    plot_predictions(y_test, predictions, X_test, scaler)

if __name__ == "__main__":
    main()
