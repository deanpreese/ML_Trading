import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, Dense, LayerNormalization, MultiHeadAttention, Add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from multiprocessing import Pool

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

# TFT Model Design with Regularization
def tft_model(sequence_length, num_features):
    l2_reg = regularizers.l2(0.01)
    
    inputs = Input(shape=(sequence_length, num_features))
    
    lstm_out = LSTM(64, return_sequences=True, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg)(inputs)
    lstm_out = LSTM(64, return_sequences=True, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg)(lstm_out)
    
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

# Function to train a single TFT model
def train_single_tft_model(seed, X_train, y_train, X_val, y_val, sequence_length, num_features):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    model = tft_model(sequence_length, num_features)
    model.compile(optimizer='adam', loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    return model

# Function to predict with ensemble of models
def ensemble_predict(models, X_test):
    predictions = np.array([model.predict(X_test).flatten() for model in models])
    return np.mean(predictions, axis=0)

# Evaluation and Plotting
def evaluate_model(predictions, y_test):
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f'Test MSE: {mse}')
    print(f'Test R2: {r2}')
    
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
    y_test_rescaled = scaler.inverse_transform(np.hstack([np.zeros((y_test.shape[0], X_test.shape[2])), y_test.reshape(-1, 1)]))[:, -1]
    predictions_rescaled = scaler.inverse_transform(np.hstack([np.zeros((predictions.shape[0], X_test.shape[2])), predictions.reshape(-1, 1)]))[:, -1]
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_rescaled, label='Actual')
    plt.plot(predictions_rescaled, label='Predicted')
    plt.title('Actual vs Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Main function to train multiple models in parallel and combine their predictions
def main():
    
    num_models = 3
    datafile = 'data/Lucky13_3070.csv'
    dtx = pd.read_csv(datafile)
    sequence_length = 7
    num_features = 13

    X_train, y_train, X_test, y_test, scaler = prepare_data(dtx, target_col='output', sequence_length=sequence_length)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Generate random seeds for each model
    seeds = np.random.randint(0, 10000, num_models)
    
    # Train models in parallel
    with Pool(num_models) as pool:
        models = pool.starmap(train_single_tft_model, [(seed, X_train, y_train, X_val, y_val, sequence_length, num_features) for seed in seeds])
    
    # Combine predictions from all models
    predictions = ensemble_predict(models, X_test)
    
    # Evaluate combined predictions
    evaluate_model(predictions, y_test)
    
    # Plot predictions
    plot_predictions(y_test, predictions, X_test, scaler)

if __name__ == "__main__":
    main()
