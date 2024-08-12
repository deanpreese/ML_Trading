import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time

#tf.config.set_visible_devices([], 'GPU')


class LiquidNeuralNetwork(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super(LiquidNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Define layers
        self.fc1 = tf.keras.layers.Dense(hidden_size, activation=None)
        self.fc2 = tf.keras.layers.Dense(output_size, activation=None)
        
        #self.fc1 = tf.keras.layers.Dense(hidden_size)
        #self.fc2 = tf.keras.layers.Dense(output_size)
        
        
        # Initialize dynamic weights to match input size
        self.dynamic_weight = tf.Variable(tf.random.normal([input_size], dtype=tf.float32), trainable=True)
    
    def call(self, inputs):
        # Apply dynamic adaptation to the weights
        dynamic_layer = tf.tanh(self.fc1(inputs * self.dynamic_weight))
        output = self.fc2(dynamic_layer)
        return output
    
    def adapt(self, inputs):
        # Update the dynamic weights based on input, ensuring dtype matches
        mean_inputs = tf.reduce_mean(inputs, axis=0)
        mean_inputs = tf.cast(mean_inputs, dtype=tf.float32)  # Cast to float32
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
            predictions = model(X_train)
            loss = tf.reduce_mean(tf.square(y_train - predictions))  # Correct way to calculate MSE
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # Adapt the model
        model.adapt(X_train)
        
        # Validation
        val_predictions = model(X_val)
        val_loss = tf.reduce_mean(tf.square(y_val - val_predictions))  # Correct way to calculate validation MSE
        
        train_losses.append(loss.numpy())
        val_losses.append(val_loss.numpy())
        
        end = time.time()
        t = round(end-start,2)
        
        print(f'{patience_counter} Epoch [{epoch+1}/{epochs}]  Elapsed Time {t} Seconds    Loss: {loss.numpy():.4f}, Val Loss: {val_loss.numpy():.4f}')
        
        # Early stopping based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
    
    """
    # Plotting the losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    """
    
    return train_losses, val_losses
    

# Function to evaluate the model and calculate win/loss statistics
def evaluate_model(model, X_test, y_test, margin=0.01):
    outputs = model(X_test)
    mse = mean_squared_error(y_test, outputs.numpy())

    targets = y_test    

    wins = 0
    losses = 0
    
    print("Evaluating ....")
    
    for i in range(len(outputs)):
        
        predict = outputs[i]
        target = targets[i]
        
        if (predict > 0 and target > 0) or (predict < 0 and target < 0):
            wins += 1
        elif (predict > 0 and target < 0) or (predict < 0 and target > 0):
            losses += 1
        elif (predict == 0 and target == 0):
            wins += 1
        elif (predict == 0 and target != 0):
            losses += 1
    
    total_samples = wins + losses
    win_percentage = (wins / total_samples) * 100
    
    print(f"Total: {total_samples}, Wins: {wins}, Losses: {losses}, Win Percentage: {win_percentage:.2f}%")
        
    
    # Calculate wins, losses, and percent winning with a margin
    #wins = np.sum((predictions.numpy() > y_test ) & (predictions.numpy() < y_test ))
    #losses = np.sum((predictions.numpy() < y_test ) | (predictions.numpy() > y_test ))
    #total = wins + losses
    #percent_winning = (wins / total) * 100 if total > 0 else 0#
    
    #print(f"Total: {total}, Wins: {wins}, Losses: {losses}, Win Percentage: {percent_winning:.2f}%")
    return mse


# Main function
def main():
    # Configuration
    hidden_size = 32
    output_size = 1
    learning_rate = 0.0009
    epochs = 150
    patience = 10
    
    filepath = 'data/Lucky13_3070.csv'
    #filepath = 'data/Lucky13_D.csv'
    
    # Load data
    X, y = load_data(filepath)
    input_size = X.shape[1]  # Dynamically set input_size based on data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Initialize model and optimizer
    model = LiquidNeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Train the model
    train_model(model, optimizer, X_train, y_train, X_val, y_val, epochs=epochs, patience=patience)
    
    # Evaluate the model
    mse = evaluate_model(model, X_test, y_test)
    print(f'Mean Squared Error on Test Data: {mse:.4f}')

if __name__ == '__main__':
    main()
