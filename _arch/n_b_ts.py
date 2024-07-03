import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Layer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#tf.config.set_visible_devices([], 'GPU')

# Data loading
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocessing
def preprocess_data(data, feature_columns, target_column):
    scaler = MinMaxScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    data[target_column] = scaler.fit_transform(data[target_column].values.reshape(-1, 1))
    return data, scaler

# Create sequences
def create_sequences(data, feature_columns, target_column, seq_length=60):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i+seq_length][feature_columns].values
        target = data.iloc[i+seq_length][target_column]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Custom layer for forecast initialization
class ForecastInitLayer(Layer):
    def __init__(self, forecast_length):
        super(ForecastInitLayer, self).__init__()
        self.forecast_length = forecast_length

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return tf.zeros((batch_size, self.forecast_length))

# N-Beats Block
class NBeatsBlock(Layer):
    def __init__(self, units, thetas_dim, backcast_length, forecast_length):
        super(NBeatsBlock, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.fc1 = Dense(units, activation='relu')
        self.fc2 = Dense(units, activation='relu')
        self.fc3 = Dense(units, activation='relu')
        self.theta_b = Dense(thetas_dim)
        self.theta_f = Dense(forecast_length)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        theta_b = self.theta_b(x)
        theta_f = self.theta_f(x)
        backcast = theta_b[:, :self.backcast_length]
        forecast = theta_f
        #print(f"Block backcast shape: {backcast.shape}, forecast shape: {forecast.shape}")
        return backcast, forecast

# Build N-Beats model
def build_nbeats_model(input_shape, stack_types, nb_blocks_per_stack, forecast_length, backcast_length, thetas_dim):
    inputs = Input(shape=input_shape)
    backcast = inputs
    forecast = ForecastInitLayer(forecast_length)(inputs)  # Initialize forecast tensor with compatible shape

    for stack_type in stack_types:
        for _ in range(nb_blocks_per_stack):
            block = NBeatsBlock(units=256, thetas_dim=thetas_dim, backcast_length=backcast_length, forecast_length=forecast_length)
            backcast, block_forecast = block(backcast)
            forecast = Add()([forecast, block_forecast])

    model = Model(inputs=inputs, outputs=forecast)
    #print(f"Model output shape: {model.output.shape}")
    return model

# Plot results
def plot_results(actuals, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()

# Main function to run the entire script
def main():
    data = load_data("data/buildSeqInd_Lucky13_5M_ALL.csv")
    data = data.drop(columns=['outputC'])
    
    # Adjust these based on your dataset
    feature_columns = list(data.columns[:-1])
    target_column = data.columns[-1]
    
    data, scaler = preprocess_data(data, feature_columns, target_column)
    
    # Split data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    
    # Create sequences
    seq_length = 60
    train_sequences, train_targets = create_sequences(train_data, feature_columns, target_column, seq_length)
    test_sequences, test_targets = create_sequences(test_data, feature_columns, target_column, seq_length)
    
    # Hyperparameters
    input_shape = (seq_length, len(feature_columns))
    forecast_length = 1
    backcast_length = seq_length
    thetas_dim = 2 * seq_length

    # Build and compile the N-Beats model
    stack_types = ['trend', 'seasonality']
    nb_blocks_per_stack = 3
    model = build_nbeats_model(input_shape, stack_types, nb_blocks_per_stack, forecast_length, backcast_length, thetas_dim)
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(train_sequences, train_targets, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping])
    
    # Evaluate the model on test data
    test_predictions = model.predict(test_sequences)
    
    # Inverse transform the predictions and targets
    test_targets = scaler.inverse_transform(test_targets.reshape(-1, 1)).reshape(-1)
    test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1)).reshape(-1)
    
    # Plot the results
    plot_results(test_targets, test_predictions)

if __name__ == "__main__":
    main()
