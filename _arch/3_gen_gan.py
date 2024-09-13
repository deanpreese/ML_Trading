import numpy as np
import pandas as pd
import tensorflow as tf
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

tf.config.set_visible_devices([], 'GPU')

def set_seeds(seed=42):
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

# Function to create sequences
def create_sequences(feature_dfs, target_df, sequence_length):
    sequences = []
    targets = []
    for i in range(len(target_df) - sequence_length):
        sequence = [df.iloc[i:i + sequence_length].values for df in feature_dfs]
        sequences.append(np.concatenate(sequence, axis=1))
        targets.append(target_df.iloc[i + sequence_length].values)
    return np.array(sequences), np.array(targets)

# Function to create the generator model
def create_generator(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(input_shape[0] * input_shape[1]))  # Output flattened sequence
    model.add(Reshape(input_shape))  # Reshape to the original input shape
    return model

# Function to create the discriminator model
def create_discriminator(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Function to create the GAN model
def create_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential([generator, discriminator])
    return model

# Function to train the GAN model
def train_gan(generator, discriminator, gan, sequences, targets, epochs=1000, batch_size=64):
    d_losses = []
    g_losses = []
    accuracy = []

    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, sequences.shape[0], batch_size)
        real_sequences = sequences[idx]
        real_targets = targets[idx]
        
        noise = np.random.normal(0, 1, (batch_size, sequences.shape[1], sequences.shape[2]))
        generated_sequences = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_sequences, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_sequences, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        d_losses.append(d_loss)
        
        # Train generator
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        g_losses.append(g_loss)
        
        # Discriminator accuracy
        predictions = discriminator.predict(real_sequences)
        acc = np.mean((predictions > 0.5) == np.ones((batch_size, 1)))
        accuracy.append(acc)
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}/{epochs}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}, Accuracy: {acc}")
    
    return d_losses, g_losses, accuracy



# Function to evaluate the GAN model
def evaluate_gan(generator, sequences, targets):
    noise = np.random.normal(0, 1, (sequences.shape[0], sequences.shape[1], sequences.shape[2]))
    predictions = generator.predict(noise)
    
    # Flatten predictions and targets
    predictions_flat = predictions.reshape(predictions.shape[0] * predictions.shape[1], -1)
    targets_flat = np.tile(targets, (predictions.shape[1], 1)).reshape(-1, targets.shape[-1])
    
    # Ensure predictions_flat and targets_flat have the same number of features
    predictions_flat = predictions_flat[:, :targets_flat.shape[1]]
    
    mse = mean_squared_error(targets_flat, predictions_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_flat, predictions_flat)
    
    wins = np.sum((predictions_flat > 0) & (targets_flat > 0))
    losses = np.sum((predictions_flat < 0) & (targets_flat > 0))
    
    return wins, losses, mse, rmse, r2

# Function to plot results
def plot_results(d_losses, g_losses, accuracy, targets, predictions):
    predictions_flat = predictions.reshape(predictions.shape[0] * predictions.shape[1], -1)
    targets_flat = np.tile(targets, (predictions.shape[1], 1)).reshape(-1, targets.shape[-1])
    
    # Ensure predictions_flat and targets_flat have the same number of features
    predictions_flat = predictions_flat[:, :targets_flat.shape[1]]

    plt.figure(figsize=(12, 6))
    
    plt.subplot(3, 1, 1)
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.legend()
    plt.title('Losses')
    
    plt.subplot(3, 1, 2)
    plt.plot(accuracy, label='Discriminator Accuracy')
    plt.legend()
    plt.title('Discriminator Accuracy')
    
    plt.subplot(3, 1, 3)
    plt.plot(targets_flat.flatten(), label='Actual')
    plt.plot(predictions_flat.flatten(), label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted')
    
    plt.tight_layout()
    plt.show()



set_seeds(42)

# Example usage
# Assuming feature_dfs is a list of three feature dataframes and target_df is the target dataframe
#feature_dfs = [pd.DataFrame(np.random.randn(1000, 10)) for _ in range(3)]
#target_df = pd.DataFrame(np.random.randn(1000, 1))

training_data_path = 'data/IND_LSTM_ALL.csv'
base = pd.read_csv(training_data_path)
base = base.drop(columns=['outputC'])

g1 = base.drop(columns=['SDLR9', 'SDLR310', 'SDKC9', 'ATR3', 'TSI', 'STOD15657'])
g2 = base.drop(columns=['SD79', 'SDLR921', 'SDBB20', 'ATR3', 'STOD7217', 'REMA'])
g3 = base.drop(columns=['SD813', 'SDLR921', 'SDBB9', 'ADX', 'ATR2', 'STOD7217'])

output = base['output'].values.reshape(-1, 1)
target_df = pd.DataFrame(output)

feature_dfs = [g1, g2, g3]


print(" ")
print(f"g1 number of features: {g1.shape[0]}   {g1.shape[1]}")
print(f"g2 number of features: {g2.shape[0]}   {g2.shape[1]}")
print(f"g3 number of features: {g3.shape[0]}   {g3.shape[1]}")
print(f"output number of features: {target_df.shape[0]}   {target_df.shape[1]}")
print(" ")

sequences, targets = create_sequences(feature_dfs, target_df, sequence_length=13)

input_shape = (sequences.shape[1], sequences.shape[2])

generator = create_generator(input_shape)
discriminator = create_discriminator(input_shape)
gan = create_gan(generator, discriminator)

discriminator.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
gan.compile(optimizer=Adam(), loss='binary_crossentropy')

d_losses, g_losses, accuracy = train_gan(generator, discriminator, gan, sequences, targets, epochs=100, batch_size=64)

wins, losses, mse, rmse, r2 = evaluate_gan(generator, sequences, targets)
print(f"Wins: {wins}, Losses: {losses}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")

predictions = generator.predict(sequences)
plot_results(d_losses, g_losses, accuracy, targets, predictions)
