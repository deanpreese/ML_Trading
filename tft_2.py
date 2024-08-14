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
        x = data[i:i + sequence_length, :-2]  # All features except target columns
        y = data[i + sequence_length, target_col_index]  # Target column
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def prepare_data(df, target_col='output', sequence_length=30):
    features = df.drop(columns=['output', 'outputC'])
    target = df[target_col]

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
    scaled_df[target_col] = target.values

    train_size = int(len(scaled_df) * 0.8)
    train_data, test_data = scaled_df[:train_size], scaled_df[train_size:]
    
    target_col_index = scaled_df.columns.get_loc(target_col)
    
    X_train, y_train = create_sequences(train_data.values, target_col_index, sequence_length)
    X_test, y_test = create_sequences(test_data.values, target_col_index, sequence_length)
    
    return X_train, y_train, X_test, y_test, scaler

# TFT Model Design with Regularization and Custom Neurons
def tft_model(sequence_length, num_features, lstm_units=[64,64], dense_units=[128, 64]):
    l2_reg = regularizers.l2(0.01)
    
    inputs = Input(shape=(sequence_length, num_features))
    
    lstm_out = LSTM(lstm_units[0], return_sequences=True, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg)(inputs)
    lstm_out = LSTM(lstm_units[1], return_sequences=True, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg)(lstm_out)
    
    attention = MultiHeadAttention(num_heads=4, key_dim=lstm_units[1], kernel_regularizer=l2_reg)(lstm_out, lstm_out)
    attention = Add()([attention, lstm_out])
    attention = LayerNormalization()(attention)
    
    dense = Dense(dense_units[0], activation='relu', kernel_regularizer=l2_reg)(attention)
    dense = Dropout(0.3)(dense)
    dense = Dense(dense_units[1], activation='relu', kernel_regularizer=l2_reg)(dense)
    dense = Dropout(0.3)(dense)
    
    output = Dense(1, kernel_regularizer=l2_reg)(dense[:, -1, :])
    
    model = Model(inputs, output)
    return model

# Function to train a single TFT model
def train_single_tft_model(seed, X_train, y_train, X_val, y_val, sequence_length, num_features, lstm_units, dense_units):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    model = tft_model(sequence_length, num_features, lstm_units, dense_units)
    model.compile(optimizer='adam', loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    return model

# Function to predict with ensemble of models
def ensemble_predict(models, X_test):
    predictions = np.array([model.predict(X_test).flatten() for model in models])
    return np.mean(predictions, axis=0)

# Evaluation function for individual models
def evaluate_individual_model(predictions, y_test, model_id):
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Calculate wins and losses
    #wins = np.sum((predictions > 0) & (y_test > 0))
    #losses = np.sum((predictions <= 0) & (y_test <= 0))
    #win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    wins = 0
    losses = 0
    
    for i in range(len(y_test)):
        if (predictions[i] > 0 and y_test[i] > 0) or (predictions[i] < 0 and y_test[i] < 0):
            wins += 1
        elif (predictions[i] > 0 and y_test[i] < 0) or (predictions[i] < 0 and y_test[i] > 0):
            losses += 1
        elif (predictions[i] == 0 and y_test[i] == 0):
            wins += 1
        elif (predictions[i] == 0 and y_test[i] != 0):
            losses += 1
    
    total_samples = wins + losses
    win_percentage = (wins / total_samples) 

    print(f'Model {model_id} Test MSE: {mse}')
    print(f'Model {model_id} Test R2: {r2}')
    print(f'Model {model_id} Wins: {wins}')
    print(f'Model {model_id} Losses: {losses}')
    print(f'Model {model_id} Win Rate: {win_percentage:.2%}')

    return mse, r2, wins, losses

# Evaluation and Plotting for the ensemble
def evaluate_ensemble(predictions, y_test):
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    wins = np.sum((predictions > 0) & (y_test > 0)) + np.sum((predictions < 0) & (y_test < 0))
    losses = np.sum((predictions < 0) & (y_test > 0))  + np.sum((predictions > 0) & (y_test < 0))  
    total = wins+losses
    win_percentage = (wins / total) 
    
    print(f'Ensemble Test MSE: {mse}')
    print(f'Ensemble Test R2: {r2}')
    print(f'Ensemble Wins: {wins}')
    print(f'Ensemble Losses: {losses}')
    print(f'Ensemble Win Rate: {win_percentage:.2%}')
    
    
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Corrected plot_predictions function
def plot_predictions(y_test, predictions, scaler):
    # Get the original number of features the scaler was fitted on
    n_features = scaler.min_.shape[0]
    
    # Create placeholders for inverse transform with the correct number of features
    y_test_placeholder = np.zeros((y_test.shape[0], n_features))
    predictions_placeholder = np.zeros((predictions.shape[0], n_features))
    
    # Set the last column (or the target column) to the y_test and predictions values
    y_test_placeholder[:, -1] = y_test
    predictions_placeholder[:, -1] = predictions
    
    # Inverse transform the placeholders
    y_test_rescaled = scaler.inverse_transform(y_test_placeholder)[:, -1]
    predictions_rescaled = scaler.inverse_transform(predictions_placeholder)[:, -1]
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_rescaled, label='Actual')
    plt.plot(predictions_rescaled, label='Predicted')
    plt.title('Actual vs Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Main function to train multiple models in parallel and combine their predictions
def main(lstm_units_list, dense_units_list):
    datafile = 'data/Lucky13_3070.csv'
    dtx = pd.read_csv(datafile)
    sequence_length = 21
    num_features = 13

    X_train, y_train, X_test, y_test, scaler = prepare_data(dtx, target_col='output', sequence_length=sequence_length)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    num_models = len(lstm_units_list)
    seeds = np.random.randint(0, 10000, num_models)
    
    # Train models in parallel
    with Pool(num_models) as pool:
        models = pool.starmap(train_single_tft_model, [
            (seeds[i], X_train, y_train, X_val, y_val, sequence_length, num_features, lstm_units_list[i], dense_units_list[i])
            for i in range(num_models)
        ])
    
    # Evaluate each model individually
    individual_results = []
    for i, model in enumerate(models):
        predictions = model.predict(X_test).flatten()
        results = evaluate_individual_model(predictions, y_test, i + 1)
        individual_results.append(results)
    
    # Combine predictions from all models
    ensemble_predictions = ensemble_predict(models, X_test)
    
    # Evaluate combined predictions
    evaluate_ensemble(ensemble_predictions, y_test)
    
    # Plot ensemble predictions
    plot_predictions(y_test, ensemble_predictions, scaler)  # Updated call with three arguments

if __name__ == "__main__":

    # Define LSTM and Dense units for each model in the ensemble
    lstm_units_list = [
        [64,64], 
        #[128, 64],
        #[32,16],
        ]  # Example of varying LSTM units for each model
    
    dense_units_list = [
        [128, 64], 
        #[256, 128], 
        # [128, 32]
        ]  
       
    # Varying Dense units for each model
    
    main(lstm_units_list, dense_units_list)
