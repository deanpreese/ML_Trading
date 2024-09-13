import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
from ml_model.model_stats import gen_reg_stats_x 

class LiquidNeuralNetwork(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size, additional_hidden_layers=1, dropout_rate=0.5):
        super(LiquidNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Define layers
        self.fc1 = tf.keras.layers.Dense(hidden_size, activation=None)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        
        # Additional hidden layers with Dropout
        self.hidden_layers = [
            (tf.keras.layers.Dense(hidden_size, activation=None), tf.keras.layers.Dropout(dropout_rate))
            for _ in range(additional_hidden_layers)
        ]
        
        self.fc2 = tf.keras.layers.Dense(output_size, activation=None)
        
        # Initialize dynamic weights to match input size
        self.dynamic_weight = tf.Variable(tf.random.normal([input_size], dtype=tf.float32), trainable=True)
    
    def call(self, inputs, training=False):
        # Apply dynamic adaptation to the weights
        dynamic_layer = tf.tanh(self.fc1(inputs * self.dynamic_weight))
        dynamic_layer = self.dropout1(dynamic_layer, training=training)
        
        # Pass through additional hidden layers with Dropout
        for layer, dropout in self.hidden_layers:
            dynamic_layer = tf.tanh(layer(dynamic_layer))
            dynamic_layer = dropout(dynamic_layer, training=training)
        
        output = self.fc2(dynamic_layer)
        return output
    
    def adapt(self, inputs):
        # Update the dynamic weights based on input, ensuring dtype matches
        mean_inputs = tf.reduce_mean(inputs, axis=0)
        mean_inputs = tf.cast(mean_inputs, dtype=tf.float32)
        self.dynamic_weight.assign_add(0.01 * mean_inputs)


# Function to load data from a CSV file
def load_data(filepath):
    df = pd.read_csv(filepath)
    features = df.drop(columns=['output'])
    target = df['output']
    return features.values, target.values

# Function to train the model with early stopping and aggressive monitoring
def train_model(model, optimizer, X_train, y_train, X_val, y_val, epochs=100, patience=10):
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        start = time.time()
        
        with tf.GradientTape() as tape:
            predictions = model(X_train, training=True)
            loss = tf.reduce_mean(tf.square(y_train - predictions))  # Calculate MSE
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # Adapt the model
        model.adapt(X_train)
        
        # Validation
        val_predictions = model(X_val, training=False)
        val_loss = tf.reduce_mean(tf.square(y_val - val_predictions))  # Calculate validation MSE
        
        train_losses.append(loss.numpy())
        val_losses.append(val_loss.numpy())
        
        end = time.time()
        t = round(end-start, 2)
        
        print(f'Epoch [{epoch+1}/{epochs}]  Elapsed Time {t} Seconds    Loss: {loss.numpy():.4f}, Val Loss: {val_loss.numpy():.4f}')
        
        # Early stopping based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
    
    return train_losses, val_losses
    

# Function to evaluate the model and calculate win/loss statistics
def evaluate_model(model, X_test, y_test):
    outputs = model(X_test, training=False)
    y_pred = outputs.numpy()
    
    correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(y_test, y_pred)
    print(f"Test MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R2: {r2}")
    print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.4f}")
    print(f"Number of Samples: {total}")
        
    return mse


# Function to load and scale data from a CSV file
def load_and_scale_data(filepath):
    df = pd.read_csv(filepath)
    features = df.drop(columns=['output'])
    target = df['output']
    
    # Scale features to range [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, target.values

def main():
    # Configuration
    hidden_size = 32
    output_size = 1
    learning_rate = 0.0005
    epochs = 150
    patience = 10
    additional_hidden_layers = 1
    dropout_rate = 0.5
    
    filepath = 'data/Lucky13_3070.csv'
    
    # Load and scale data
    X, y = load_and_scale_data(filepath)
    input_size = X.shape[1]
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Initialize model and optimizer
    model = LiquidNeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size, 
                                additional_hidden_layers=additional_hidden_layers, dropout_rate=dropout_rate)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Train the model
    train_model(model, optimizer, X_train, y_train, X_val, y_val, epochs=epochs, patience=patience)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
