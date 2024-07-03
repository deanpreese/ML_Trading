import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

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
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(input_shape[1]))  # Output features for each time step
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
            print(" ")
            print(f"Epoch {epoch}/{epochs}  Accuracy: {acc}")
            converted_values = [float(value) for value in d_loss]
            print("Discriminator Loss:", *converted_values)
            converted_values = [float(value) for value in g_loss]
            print("Generator Loss:", *converted_values)
            print(" ")
    
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
    losses = np.sum((predictions_flat <= 0) & (targets_flat > 0))
    
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


# -------------------------------------------------------------------------

training_data_path = 'data/IND_LSTM_ALL.csv'
base = pd.read_csv(training_data_path)
base = base.drop(columns=['outputC'])

df1=base
#df1 = df1.drop(columns=['SD79'])
df1 = df1.drop(columns=['SD813'])
df1 = df1.drop(columns=['SD921'])
#df1 = df1.drop(columns=['SDLR9'])
#df1 = df1.drop(columns=['SDLR921'])
df1 = df1.drop(columns=['SDLR2155'])
df1 = df1.drop(columns=['SDLR310'])
#df1 = df1.drop(columns=['SDBB9'])
df1 = df1.drop(columns=['SDBB20'])
df1 = df1.drop(columns=['SDKC9'])
#df1 = df1.drop(columns=['ADX'])
#df1 = df1.drop(columns=['ROC'])
#df1 = df1.drop(columns=['ATR3'])
df1 = df1.drop(columns=['ATR2'])
#df1 = df1.drop(columns=['RSI'])
df1 = df1.drop(columns=['TSI'])
#df1 = df1.drop(columns=['STOD7217'])
df1 = df1.drop(columns=['STOK7217'])
#df1 = df1.drop(columns=['STOD15657'])
#df1 = df1.drop(columns=['STOK15657'])
df1 = df1.drop(columns=['REMA'])
df1 = df1.drop(columns=['output'])

df2 = base
#df2 = df2.drop(columns=['SD79'])
#df2 = df2.drop(columns=['SD813'])
#df2 = df2.drop(columns=['SD921'])
#df2 = df2.drop(columns=['SDLR9'])
df2 = df2.drop(columns=['SDLR921'])
df2 = df2.drop(columns=['SDLR2155'])
#df2 = df2.drop(columns=['SDLR310'])
df2 = df2.drop(columns=['SDBB9'])
df2 = df2.drop(columns=['SDBB20'])
#df2 = df2.drop(columns=['SDKC9'])
df2 = df2.drop(columns=['ADX'])
#df2 = df2.drop(columns=['ROC'])
df2 = df2.drop(columns=['ATR3'])
#df2 = df2.drop(columns=['ATR2'])
df2 = df2.drop(columns=['RSI'])
#df2 = df2.drop(columns=['TSI'])
df2 = df2.drop(columns=['STOD7217'])
#df2 = df2.drop(columns=['STOK7217'])
df2 = df2.drop(columns=['STOD15657'])
#df2 = df2.drop(columns=['STOK15657'])
df2 = df2.drop(columns=['REMA'])
df2 = df2.drop(columns=['output'])


df3 = base
#df3 = df3.drop(columns=['SD79'])
#df3 = df3.drop(columns=['SD813'])
#df3 = df3.drop(columns=['SD921'])
#df3 = df3.drop(columns=['SDLR9'])
df3 = df3.drop(columns=['SDLR921'])
df3 = df3.drop(columns=['SDLR2155'])
#df3 = df3.drop(columns=['SDLR310'])
df3 = df3.drop(columns=['SDBB9'])
df3 = df3.drop(columns=['SDBB20'])
#df3 = df3.drop(columns=['SDKC9'])
df3 = df3.drop(columns=['ADX'])
#df3 = df3.drop(columns=['ROC'])
df3 = df3.drop(columns=['ATR3'])
#df3 = df3.drop(columns=['ATR2'])
df3 = df3.drop(columns=['RSI'])
#df3 = df3.drop(columns=['TSI'])
df3 = df3.drop(columns=['STOD7217'])
#df3 = df3.drop(columns=['STOK7217'])
df3 = df3.drop(columns=['STOD15657'])
#df3 = df3.drop(columns=['STOK15657'])
df3 = df3.drop(columns=['REMA'])
df3 = df3.drop(columns=['output'])


output_df = pd.read_csv(training_data_path)
output = output_df['output'].values.reshape(-1, 1)


feature_dfs = [df1,df2,df3]
df1 = feature_dfs[0]
df2 = feature_dfs[1]
df3 = feature_dfs[2]

target_df = pd.DataFrame(output)

print(" ")
print(f"df1 number of features: {df1.shape[0]}   {df1.shape[1]}")
print(f"df2 number of features: {df2.shape[0]}   {df2.shape[1]}")
print(f"df3 number of features: {df3.shape[0]}   {df3.shape[1]}")
print(f"output features: {output.shape[0]}   {output.shape[1]}")
print(" ")

if df1.shape[1] != df2.shape[1]:
    print("ERROR: df1 and df2 must have the same number of features")
    exit()


# Example usage
# Assuming feature_dfs is a list of three feature dataframes and target_df is the target dataframe
#feature_dfs = [pd.DataFrame(np.random.randn(1000, 10)) for _ in range(3)]
#target_df = pd.DataFrame(np.random.randn(1000, 1))



sequences, targets = create_sequences(feature_dfs, target_df, sequence_length=10)
input_shape = (sequences.shape[1], sequences.shape[2])
generator = create_generator(input_shape)
discriminator = create_discriminator(input_shape)
gan = create_gan(generator, discriminator)

discriminator.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
gan.compile(optimizer=Adam(), loss='binary_crossentropy')

d_losses, g_losses, accuracy = train_gan(generator, discriminator, gan, sequences, targets, epochs=20, batch_size=64)

wins, losses, mse, rmse, r2 = evaluate_gan(generator, sequences, targets)
print(f"Wins: {wins}, Losses: {losses}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")

predictions = generator.predict(sequences)
plot_results(d_losses, g_losses, accuracy, targets, predictions)
