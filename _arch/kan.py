import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Add
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

#tf.config.set_visible_devices([], 'GPU')


def load_and_prepare_data(file_path):

    df = pd.read_csv(file_path)
    
    # Drop the binary output 'outputC'
    df = df.drop(columns=['outputC'])
    
    # Separate features and continuous target
    X = df.drop(columns=['output']).values
    y = df['output'].values
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    input_dim = X_train.shape[1]
    
    return X_train, X_test, y_train, y_test, input_dim

def build_enhanced_kolmogorov_arnold_network(input_dim, hidden_units, output_dim):
    inputs = Input(shape=(input_dim,))
    
    # Univariate functions
    univariate_outputs = []
    for i in range(input_dim):
        x = Lambda(lambda x: x[:, i:i+1])(inputs)
        x = Dense(hidden_units, activation='relu')(x)
        x = Dense(hidden_units, activation='relu')(x)
        univariate_outputs.append(x)
    
    # Summation of univariate outputs
    sum_output = Add()(univariate_outputs)
    
    # Additional layers
    sum_output = Dense(hidden_units, activation='relu')(sum_output)
    
    # Parallel network for feature interaction
    interaction_layer = Dense(hidden_units, activation='relu')(inputs)
    interaction_layer = Dense(hidden_units, activation='relu')(interaction_layer)
    
    # Combine original sum_output with interaction layer
    combined_output = Add()([sum_output, interaction_layer])
    
    # Final output layer
    outputs = Dense(output_dim)(combined_output)
    
    # Build and compile the model
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    
    return model



def build_kolmogorov_arnold_network(input_dim, hidden_units, output_dim):
    """
    Builds the Kolmogorov-Arnold Network architecture.
    
    Parameters:
    - input_dim: int, number of input features
    - hidden_units: int, number of units in the hidden layers of the univariate functions
    - output_dim: int, number of output units (typically 1 for regression)
    
    Returns:
    - model: keras.Model, compiled Kolmogorov-Arnold Network model
    """
    inputs = Input(shape=(input_dim,))
    
    # List to store outputs of each univariate function
    univariate_outputs = []
    
    # Create univariate functions
    for i in range(input_dim):
        # Dense layers for the i-th univariate function
        x = Lambda(lambda x: x[:, i:i+1])(inputs)
        x = Dense(hidden_units, activation='relu')(x)
        #x = Dense(hidden_units*2, activation='relu')(x)
        x = Dense(hidden_units, activation='relu')(x)
        univariate_outputs.append(x)
    
    # Sum all univariate outputs
    sum_output = Add()(univariate_outputs)
    
    # Final output layer
    outputs = Dense(output_dim)(sum_output)
    
    # Build and compile the model
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    model.summary()
    
    return model

def train_model(model, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
    """
    Trains the Kolmogorov-Arnold Network model with early stopping.
    
    Parameters:
    - model: keras.Model, the model to be trained
    - X_train: np.ndarray, training features
    - y_train: np.ndarray, training target values
    - epochs: int, number of training epochs
    - batch_size: int, size of the training batches
    - validation_split: float, fraction of the training data to be used as validation data
    
    Returns:
    - history: keras.callbacks.History, the history object containing training details
    """
    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_split=validation_split, callbacks=[early_stopping])
    return history

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set and computes extensive statistics including correct win/loss logic.
    
    Parameters:
    - model: keras.Model, the trained model
    - X_test: np.ndarray, testing features
    - y_test: np.ndarray, testing target values
    
    Returns:
    - metrics: dict, containing MSE, MAE, R2, Total Wins, Total Losses, Win Percentage, and the number of samples
    """
    y_pred = model.predict(X_test)
    
    # Ensure y_pred has the correct shape
    if y_pred.shape != y_test.shape:
        y_pred = np.reshape(y_pred, y_test.shape)
    
    print(f"y_pred.shape: {y_pred.shape}, y_test.shape: {y_test.shape}")
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Win/Loss calculation
    wins = 0
    losses = 0
    
    wins = np.sum((y_pred > 0) & (y_test > 0)) + np.sum((y_pred < 0) & (y_test < 0))
    losses = np.sum((y_pred < 0) & (y_test > 0))  + np.sum((y_pred > 0) & (y_test < 0))  
    total = wins+losses
    win_percentage = (wins / total) 
    
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'Total Wins': wins,
        'Total Losses': losses,
        'Win Percentage': win_percentage,
        'Samples': total
    }
    
    return metrics

def plot_training_history(history):
    """
    Plots the training and validation loss and MAE over epochs.
    
    Parameters:
    - history: keras.callbacks.History, the history object returned by model training
    """
    plt.figure(figsize=(12, 6))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    # Plot training & validation MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.show()

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

    file_path = datafile[1]
    
    
    # Load and prepare the data
    X_train, X_test, y_train, y_test, input_dim = load_and_prepare_data(file_path)
    
    # Define model parameters
    hidden_units = 128  # Adjust this based on your data
    output_dim = 1  # Single continuous output
    
    # Build the Kolmogorov-Arnold Network model
    #model = build_kolmogorov_arnold_network(input_dim, hidden_units, output_dim)
    model = build_enhanced_kolmogorov_arnold_network(input_dim, hidden_units, output_dim)
    
    # Train the model
    history = train_model(model, X_train, y_train)
    
    # Plot the training history
    plot_training_history(history)
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Test MSE: {metrics['MSE']}, Test MAE: {metrics['MAE']}, R2: {metrics['R2']}")
    print(f"Total Wins: {metrics['Total Wins']}, Total Losses: {metrics['Total Losses']}, Win Percentage: {metrics['Win Percentage']:.2f}%")
    print(f"Number of Samples: {metrics['Samples']}")

if __name__ == "__main__":
    main()
