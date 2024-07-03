import numpy as np
import pandas as pd
import tensorflow as tf
import random
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, LeakyReLU, BatchNormalization, Input, Dropout, TimeDistributed, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

tf.config.set_visible_devices([], 'GPU')

# Define the self-attention layer
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.theta = Dense(channels // 8)
        self.phi = Dense(channels // 8)
        self.g = Dense(channels // 2)
        self.o = Dense(channels)
        self.gamma = self.add_weight(shape=[1], initializer='zeros', trainable=True)

    def call(self, inputs):
        theta = self.theta(inputs)
        phi = self.phi(inputs)
        g = self.g(inputs)

        attention_map = tf.matmul(theta, phi, transpose_b=True)
        attention_map = tf.nn.softmax(attention_map, axis=-1)

        attention = tf.matmul(attention_map, g)
        attention = self.o(attention)

        return self.gamma * attention + inputs


# Function to create the generator with LSTM and self-attention
def create_generator(latent_dim, seq_length, feature_dim):
    model = Sequential()
    model.add(Dense(seq_length * feature_dim, activation=LeakyReLU(alpha=0.2), input_dim=latent_dim))
    model.add(Reshape((seq_length, feature_dim)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(SelfAttention(feature_dim))
    model.add(LSTM(256, return_sequences=True))
    model.add(BatchNormalization(momentum=0.8))
    model.add(SelfAttention(256))
    model.add(TimeDistributed(Dense(feature_dim, activation='tanh')))
    return model

# Function to create the discriminator with LSTM and self-attention
def create_discriminator(seq_length, feature_dim):
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(seq_length, feature_dim)))
    model.add(Dropout(0.4))
    model.add(LSTM(128))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Function to combine generator and discriminator into a GAN
def create_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    return gan

# Function to create sequences
def create_sequences(features, target, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)




def train_gan(generator, discriminator, gan, X_train, epochs=5000, batch_size=64, patience=10 , min_delta=0.001):
    half_batch = batch_size // 2
    history = {'d_loss': [], 'g_loss': [], 'd_acc': []}
    best_g_loss = np.inf
    patience_counter = 0

    # Define optimizers
    generator_optimizer = Adam(0.0002, 0.5)
    discriminator_optimizer = Adam(0.0002, 0.5)

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_samples = X_train[idx]
        noise = np.random.normal(0, 1, (half_batch, generator.input_shape[1]))
        generated_samples = generator.predict(noise)

        with tf.GradientTape() as tape:
            real_output = discriminator(real_samples, training=True)
            fake_output = discriminator(generated_samples, training=True)
            d_loss_real = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output)
            d_loss_fake = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        if grads:
            discriminator_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
        else:
            print("No gradients computed for discriminator.")

        d_acc_real = tf.keras.metrics.binary_accuracy(tf.ones_like(real_output), real_output)
        d_acc_fake = tf.keras.metrics.binary_accuracy(tf.zeros_like(fake_output), fake_output)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)

        history['d_loss'].append(d_loss.numpy().mean())
        history['d_acc'].append(d_acc.numpy().mean())

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, generator.input_shape[1]))

        with tf.GradientTape() as tape:
            fake_output = discriminator(generator(noise, training=True), training=True)
            g_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output)

        grads = tape.gradient(g_loss, generator.trainable_variables)
        if grads:
            generator_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        else:
            print("No gradients computed for generator.")

        history['g_loss'].append(g_loss.numpy().mean())
        #g_loss_value = g_loss if not isinstance(g_loss, list) else g_loss[0]
        g_loss_value = g_loss.numpy().mean()
       
        if g_loss_value < best_g_loss - min_delta:
            best_g_loss = g_loss_value
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break            
            
        if epoch % 2 == 0:
            print(" ")
            print(f"Epoch: {epoch}/{epochs}  Accuracy: {100*d_acc[0]}")
            print(f"Patience: {patience_counter}")
            print("Generator Loss:", g_loss.numpy().mean())
            print("Discriminator Loss:", d_loss.numpy().mean())            
            print(" ")            

    return history


def evaluate_gan(generator, X_test, y_test):
    print(f"X_test shape: {X_test.shape}")
    
    # Generate noise based on the latent dimension
    noise = np.random.normal(0, 1, (X_test.shape[0], latent_dim))
    predictions = generator.predict(noise)
    print(f"Predictions shape: {predictions.shape}")
    
    # Extract the last time step prediction correctly
    last_time_step_predictions = predictions[:, -1, 0]  # Using only the first feature
    print(f"Last time step predictions shape: {last_time_step_predictions.shape}")
    
    # Ensure y_test has a compatible shape
    y_test_reshaped = y_test[:, -1, 0].reshape(-1)
    print(f"Reshaped y_test shape: {y_test_reshaped.shape}")

    # Flatten the arrays to make them compatible with sklearn metrics
    y_test_flat = y_test_reshaped.flatten()
    predictions_flat = last_time_step_predictions.flatten()

    print(f"Flattened y_test shape: {y_test_flat.shape}")
    print(f"Flattened predictions shape: {predictions_flat.shape}")

    mse = mean_squared_error(y_test_flat, predictions_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_flat, predictions_flat)

    wins = np.sum((predictions_flat > 0) & (y_test_flat > 0))
    losses = len(y_test_flat) - wins

    print(f"MSE: {mse}, RMSE: {rmse}, R2: {r2}, Wins: {wins}, Losses: {losses}")
    return predictions_flat, y_test_flat



def plot_results(history, y_test, predictions):
    plt.figure(figsize=(10, 5))

    # Plot history of Discriminator loss vs Generator loss
    plt.subplot(1, 2, 1)
    plt.plot(history['d_loss'], label='Discriminator Loss')
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.legend()
    plt.title('Loss History')

    # Plot actual vs predicted
    plt.subplot(1, 2, 2)
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted')

    plt.show()




if __name__ == "__main__":
    
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    
    # Generate dummy data
    #features = np.random.rand(1000, 10)
    #target = np.random.rand(1000, 1)

    #train_file = pd.read_csv('data/sm13_3070.csv')
    #train_file =  pd.read_csv('data/IND_LSTM_ALL.csv')
    train_file = pd.read_csv('data/buildSeqInd_Lucky13_5M_ALL.csv')
    data = train_file.drop(columns=['outputC'])

    features = data.drop(columns=['output']).to_numpy()
    target = data['output'].values.reshape(-1, 1)

    print(" ")
    print(f"features: {features.shape[0]}   {features.shape[1]}")
    print(f"output number of features: {target.shape}")
    print(" ")

    # Create sequences
    seq_length = 13
    X, y = create_sequences(features, target, seq_length)

    # Ensure the target shape matches the generator's output
    y = y[:, np.newaxis, :]  # Add a new axis to match the expected shape (samples, 1, feature_dim)

    # Reshape the sequences for the LSTM GAN
    X = X.reshape((X.shape[0], seq_length, features.shape[1]))
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    latent_dim = 100  # Latent dimension size

    # Create generator and discriminator
    generator = create_generator(latent_dim, seq_length, features.shape[1])
    discriminator = create_discriminator(seq_length, features.shape[1])

    # Print model summaries to check input/output shapes
    generator.summary()
    discriminator.summary()

    # Compile GAN
    gan = create_gan(generator, discriminator)
    gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

    # Train GAN with early stopping
    history = train_gan(generator, discriminator, gan, X, epochs=100, batch_size=64, patience=10, min_delta=0.00001)

    # Evaluate GAN
    print("Evaluating GAN")
    predictions, y_test_flat = evaluate_gan(generator, X, y)

    # Plot results
    #plot_results(history, y_test_flat, predictions)

