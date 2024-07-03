import numpy as np
import pandas as pd
import tensorflow as tf
import random
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

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

# Function to create sequences
def create_sequences(features, target, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

# Function to create the generator with self-attention
def create_generator(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation=LeakyReLU(alpha=0.2)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(SelfAttention(128))
    model.add(Dense(256, activation=LeakyReLU(alpha=0.2)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(SelfAttention(256))
    model.add(Dense(512, activation=LeakyReLU(alpha=0.2)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(SelfAttention(512))
    model.add(Dense(output_dim, activation='tanh'))
    return model

# Function to create the discriminator with self-attention
def create_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim, activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Function to combine generator and discriminator into a GAN
def create_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(generator.input_shape[1],))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    return gan

# Function to train the GAN
def train_gan(generator, discriminator, gan, X_train, epochs=10000, batch_size=64, patience=10, min_delta=0.001):
    best_g_loss = np.inf
    patience_counter = 0
    half_batch = batch_size // 2
    history = {'d_loss': [], 'g_loss': [], 'd_acc': []}
    
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_samples = X_train[idx]
        noise = np.random.normal(0, 1, (half_batch, generator.input_shape[1]))
        generated_samples = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_samples, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_samples, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
        d_acc = 0.5 * np.add(d_loss_real[1], d_loss_fake[1])
        history['d_loss'].append(d_loss)
        history['d_acc'].append(d_acc)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, generator.input_shape[1]))
        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch(noise, valid_y)
        history['g_loss'].append(g_loss)

        g_loss_value = g_loss if not isinstance(g_loss, list) else g_loss[0]
            
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
            print(f"Epoch: {epoch}/{epochs}  Accuracy: {100*d_acc}")
            print(f"Patience: {patience_counter}")
            print("Generator Loss:", *g_loss)
            print("Discriminator Loss:", d_loss)            
            print(" ")            
            

    return history

# Function to evaluate the GAN
def evaluate_gan(generator, X_test, y_test):
    predictions = generator.predict(X_test)
    print(f"Predictions shape: {predictions.shape}")
    print(f"y_test shape: {y_test.shape}")
    predictions = predictions[:, :y_test.shape[1]]  # Ensure predictions match the shape of y_test
    print(f"Reshaped predictions shape: {predictions.shape}")
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    wins = np.sum((predictions > 0) & (y_test > 0))
    losses = len(y_test) - wins

    print(f"MSE: {mse}, RMSE: {rmse}, R2: {r2}, Wins: {wins}, Losses: {losses}")
    return predictions

# Function to plot the results
def plot_results(history, y_test, predictions):
   
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
    
    # Plot history of Discriminator loss vs Generator loss
    ax1.plot(history['d_loss'], label='Discriminator Loss')
    ax1.plot(history['g_loss'], label='Generator Loss')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Loss History')

    # Plot actual vs predicted
    
    ax2.plot(y_test.flatten(), label='Actual')
    ax2.plot(predictions.flatten(), label='Predicted')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Actual vs Predicted')

    plt.show()
   
# Example of using the functions
if __name__ == "__main__":
    
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    
    # Generate dummy data
    #features = np.random.rand(1000, 10)
    #target = np.random.rand(1000, 1)


    #train_file = pd.read_csv('data/sm13_3070.csv')
    train_file =  pd.read_csv('data/IND_LSTM_ALL.csv')
    #train_file = pd.read_csv('data/buildSeqInd_Lucky13_5M_ALL.csv')
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

    # Flatten the sequences for the GAN
    X = X.reshape((X.shape[0], -1))
    y = y.reshape((y.shape[0], -1))  # Ensure y has the same shape for comparison

    # Create generator and discriminator
    generator = create_generator(input_dim=X.shape[1], output_dim=X.shape[1])  # Ensure output dimension matches X
    discriminator = create_discriminator(input_dim=X.shape[1])

    # Compile discriminator
    discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

    # Create GAN
    gan = create_gan(generator, discriminator)
    gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

    # Train GAN
    history = train_gan(generator, discriminator, gan, X, epochs=100, batch_size=64 , patience=10, min_delta=0.00001)

    # Evaluate GAN
    predictions = evaluate_gan(generator, X, y)

    # Plot results
    #plot_results(history, y, predictions)
