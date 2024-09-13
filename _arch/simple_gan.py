import numpy as np
import pandas as pd
import tensorflow as tf
import random
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, LSTM, Concatenate, Reshape, Flatten, Dense, LeakyReLU, Dropout, MultiHeadAttention
from tensorflow.keras.layers import   BatchNormalization, Layer,  Attention, Bidirectional, TimeDistributed, Conv1D

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

# Disable GPU usage
tf.config.set_visible_devices([], 'GPU')

class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_q = self.add_weight(name='W_q',
                                   shape=(input_shape[-1], input_shape[-1]),
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.W_k = self.add_weight(name='W_k',
                                   shape=(input_shape[-1], input_shape[-1]),
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.W_v = self.add_weight(name='W_v',
                                   shape=(input_shape[-1], input_shape[-1]),
                                   initializer='glorot_uniform',
                                   trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        q = tf.tensordot(inputs, self.W_q, axes=1)
        k = tf.tensordot(inputs, self.W_k, axes=1)
        v = tf.tensordot(inputs, self.W_v, axes=1)
        
        attn_weights = tf.nn.softmax(tf.matmul(q, k, transpose_b=True))
        attn_output = tf.matmul(attn_weights, v)
        
        return attn_output


def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def create_sequences(data_in, seq_length_in):
    """Create sequences from the input data."""
    print("create_sequences")
    xs = [data_in.iloc[i:i + seq_length_in, :-1].values for i in range(len(data_in) - seq_length_in)]
    ys = data_in.iloc[seq_length_in:, -1].values
    return np.array(xs), np.array(ys)

def normalize_sequences(sequences_in):
    """Normalize sequences using MinMaxScaler."""
    scalers_out = {}
    print("normalize_sequences")
    for i in range(sequences_in.shape[0]):
        scalers_out[i] = MinMaxScaler((0, 1))
        sequences_in[i] = scalers_out[i].fit_transform(sequences_in[i])
    return sequences_in, scalers_out

def normalize_targets(targets_in):
    """Normalize targets using MinMaxScaler."""
    scaler = MinMaxScaler((0, 1))
    print("normalize_targets")
    targets_out = scaler.fit_transform(targets_in.reshape(-1, 1)).flatten()
    return targets_out, scaler

def reverse_scaling(preds_in, scalers_in, seq_length_in, feature_dim_in):
    """Reverse the scaling of the predictions."""
    reversed_preds = []
    for i in range(len(preds_in)):
        temp_input = np.zeros((seq_length_in, feature_dim_in))
        temp_input[:, -1] = preds_in[i]
        reversed_pred = scalers_in[i].inverse_transform(temp_input)
        reversed_preds.append(reversed_pred[0, -1])
    return np.array(reversed_preds)

def sequence_and_normalize(data_in, seq_length_in):
    """Create and normalize sequences, and split the data into training and validation sets."""
    feature_cnt = data_in.shape[1] - 1
    X, y = create_sequences(data_in, seq_length_in)
    X, scalers_x = normalize_sequences(X)
    y, scaler_y = normalize_targets(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return feature_cnt, scalers_x, scaler_y, X_train, X_val, y_train, y_val

def gradient_penalty(discriminator, real_data, fake_data, batch_size):
    epsilon = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)
    gradients = tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp


def build_generator(timesteps, n_features, latent_dim, layer1, layer2, layer3):
    """Define the Generator model."""
    input_layer = Input(shape=(latent_dim,))
    x =  Dense(timesteps * n_features, activation="relu")(input_layer)
    x =  Reshape((timesteps, n_features))(x)
    x =  LSTM(layer1, return_sequences=True)(x)
    x =  LSTM(layer2, return_sequences=True)(x)
    output_layer =  TimeDistributed(Dense(n_features))(x)
    
    print("Input Shape:", input_layer.shape)
    print("Generator Output Shape:", output_layer.shape)
    
    return Model(input_layer, output_layer)
    
def build_generatorD(timesteps, n_features, latent_dim, layer1, layer2, layer3):
    init = RandomNormal(stddev=0.02)
    
    input_layer = Input(shape=(latent_dim,))
    x = Dense(timesteps * n_features, kernel_initializer=init )(input_layer)
    x = Reshape((timesteps, n_features))(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = SelfAttention()(x)

    x = Dense(layer1, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = SelfAttention()(x)
    
    x = Dense(layer2, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = SelfAttention()(x)    
    
    x = Dense(layer3, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    output_layer =  TimeDistributed(Dense(n_features))(x)
    return Model(input_layer, output_layer)    
    
    


def build_generatorX(timesteps, n_features, latent_dim, layer1, layer2, layer3):
    
    init = RandomNormal(stddev=0.02)
    
    input_layer = Input(shape=(latent_dim,))
    x =  Dense(timesteps * n_features, activation="relu")(input_layer)
    x =  Reshape((timesteps, n_features))(x)
    
    conv_r = Reshape((timesteps, n_features, 1))(x)
    conv1 = TimeDistributed(Conv1D(filters=32, kernel_size=n_features, activation='relu', padding='same'))(conv_r)
    conv1 = Flatten()(conv1)  
    conv1 = Reshape((timesteps, -1))(conv1)

    #y = Bidirectional(LSTM(layer2 ,return_sequences=True, kernel_initializer=init))(conv1)
    y = LSTM(layer2 , return_sequences=True, kernel_initializer=init)(conv1)
    y = LeakyReLU(negative_slope=0.2)(y)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)
   
    x = LSTM(layer1, return_sequences=True, kernel_initializer=init)(x)
    x = MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
   
    #x = Bidirectional(LSTM(layer2 ,return_sequences=True, kernel_initializer=init))(x)
    x = LSTM(layer2, return_sequences=True, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = MultiHeadAttention(num_heads=3, key_dim=32)(x, y)
    
    x = LSTM(layer3, return_sequences=True)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
     
    #x =  TimeDistributed(Dense(n_features, activation='tanh'))(x)   
    x =  TimeDistributed(Dense(n_features))(x)   
    return Model(input_layer, x)
   

def build_discriminator(timesteps, n_features, layer1, layer2, layer3):
    
    init = RandomNormal(stddev=0.02)
    input_layer = Input(shape=(timesteps, n_features))
    x = Dense(layer1, kernel_initializer=init)(input_layer)
    x = Dense(layer2, kernel_initializer=init)(x)
    x = Dense(layer3, kernel_initializer=init)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(input_layer, x)        
    

def build_discriminatorD(timesteps, n_features, layer1, layer2, layer3):
    
    init = RandomNormal(stddev=0.02)
    input_layer = Input(shape=(timesteps, n_features))
    
    x = Dense(layer1, kernel_initializer=init)(input_layer)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = SelfAttention()(x)
    
    x = Dense(layer2, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = SelfAttention()(x)
    
    x = Dense(layer3, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = SelfAttention()(x)
    
    x = Dense(1, activation='sigmoid')(x)
    return Model(input_layer, x)    
    

def build_discriminatorX(timesteps, n_features, layer3,layer2, layer1):
    
    init = RandomNormal(stddev=0.02)
    input_layer = Input(shape=(timesteps, n_features))
    
    reshaped_input = Reshape((timesteps, n_features, 1))(input_layer)
    conv1 = TimeDistributed(Conv1D(filters=32, kernel_size=7, activation='relu', padding='same'))(reshaped_input)
    conv1 = Flatten()(conv1)  
    conv1 = Reshape((timesteps, -1))(conv1)

    x = LSTM(layer1, return_sequences=True, kernel_initializer=init)(input_layer)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Bidirectional(LSTM(layer2 ,return_sequences=True))(x)
    #x = LSTM(layer2, return_sequences=True, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = LSTM(layer3, return_sequences=True, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Attention()([x, x])
    #x = Flatten()(x)
    
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(layer3)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = Dense(1, activation='sigmoid')(x)
    return Model(input_layer, x)
    
    
def create_gan(generator, discriminator, latent_dim, g_lr=0.001, d_lr=0.001):
    """Create and compile the GAN model."""

    #discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=d_lr))
    discriminator.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=d_lr))

    gan_input = Input(shape=(latent_dim,))
    generated_sequence = generator(gan_input)
    discriminator.trainable = False
    gan_output = discriminator(generated_sequence)
    gan_model = Model(gan_input, gan_output)
    #gan_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=g_lr))
    gan_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=g_lr))
    gan_model.summary()
    
    return gan_model


def train_model(generator, discriminator, gan_model, X_train, latent_dim, batch_size, epochs, steps_per_epoch, patience_epochs, min_delta=0.001, gp_weight=10.0):
    """Train the GAN model."""
    best_g_loss = np.inf
    patience_counter = 0

    history = {'d_loss': [], 'g_loss': []}

    # Initialize optimizers
    gen_optimizer = Adam(0.0002, beta_1=0.5)
    disc_optimizer = Adam(0.0002, beta_1=0.5)

    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            noise = tf.random.normal(shape=(batch_size, latent_dim))
            fake_sequences = generator(noise, training=False)
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_sequences = X_train[idx]

            with tf.GradientTape() as disc_tape:
                d_loss_real = tf.reduce_mean(discriminator(real_sequences, training=True))
                d_loss_fake = tf.reduce_mean(discriminator(fake_sequences, training=True))
                gp = gradient_penalty(discriminator, real_sequences, fake_sequences, batch_size)
                d_loss = d_loss_fake - d_loss_real + gp_weight * gp
                #print(f"    Discriminator Losses: Real {d_loss_real.numpy()}, Fake {d_loss_fake.numpy()}, GP {gp.numpy()}")  # Debugging

            d_gradients = disc_tape.gradient(d_loss, discriminator.trainable_variables)
            if d_gradients and all([grad is not None for grad in d_gradients]):
                disc_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
                #print("     Applied discriminator gradients")  # Debugging
            
            noise = tf.random.normal(shape=(batch_size, latent_dim))
            with tf.GradientTape() as gen_tape:
                fake_sequences = generator(noise, training=True)
                g_loss = -tf.reduce_mean(discriminator(fake_sequences, training=True))
                #print(f"         Generator Loss: {g_loss.numpy()}")  # Debugging

            g_gradients = gen_tape.gradient(g_loss, generator.trainable_variables)
            if g_gradients and all([grad is not None for grad in g_gradients]):
                gen_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
                #print("          Applied generator gradients")  # Debugging
            

            history['d_loss'].append(d_loss.numpy())
            history['g_loss'].append(g_loss.numpy())

        print(f"Epoch {epoch}/{epochs} - Patience Epochs: {patience_counter}  -  Discriminator Loss: {d_loss.numpy()}, Generator Loss: {g_loss.numpy()}")

        g_loss_value = np.mean(history['g_loss'])
        
        if g_loss_value < best_g_loss - min_delta:
            best_g_loss = g_loss_value
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_epochs:
            print(f"Early stopping at epoch {epoch}")
            break

    return history


 
def train_model_X(generator, discriminator, gan_model, X_train, latent_dim, batch_size, epochs, steps_per_epoch, patience_epochs, min_delta=0.001):
    """Train the GAN model."""
    best_g_loss = np.inf
    patience_counter = 0

    history = {'d_loss': [], 'g_loss': []}

    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            noise = tf.random.normal(shape=(batch_size, latent_dim))
            fake_sequences = generator.predict(noise)
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_sequences = X_train[idx]

            d_loss_real = discriminator.train_on_batch(real_sequences, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(fake_sequences, np.zeros((batch_size, 1)))
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            g_loss = gan_model.train_on_batch(noise, np.ones((batch_size, 1)))

            history['d_loss'].append(d_loss)
            history['g_loss'].append(g_loss)

        print(f"Epoch {epoch}/{epochs} - Patience Epochs: {patience_counter}  -  Discriminator Loss: {d_loss}, Generator Loss: {g_loss[0]}")

        g_loss_value = np.mean(history['g_loss'])
        
        if g_loss_value < best_g_loss - min_delta:
            best_g_loss = g_loss_value
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_epochs:
            print(f"Early stopping at epoch {epoch}")
            break


    return history

def evaluate_model(generator, discriminator, X_test, latent_dim, history, sample_index=0):
    """Evaluate the model and plot results."""
    noise = tf.random.normal(shape=(len(X_test), latent_dim))
    generated_data = generator.predict(noise)
    real_output = discriminator.predict(X_test)
    fake_output = discriminator.predict(generated_data)

    ks_test = ks_2samp(X_test.flatten(), generated_data.flatten(),method='exact')
    print(f"KS test statistic: {ks_test.statistic} {ks_test.pvalue}  {ks_test.statistic_location}  {ks_test.statistic_sign} ")
    print(f"Discriminator's confidence on real data (mean): {np.mean(real_output)}")
    print(f"Discriminator's confidence on generated data (mean): {np.mean(fake_output)}")
    rmse = np.sqrt(mean_squared_error(X_test.flatten(), generated_data.flatten()))
    mae = mean_absolute_error(X_test.flatten(), generated_data.flatten())
    print(f"Root Mean Squared Error (RMSE) between real and generated data: {rmse}")
    print(f"Mean Absolute Error (MAE) between real and generated data: {mae}")


def plot_results(generator, discriminator, X_test, latent_dim, history, sample_index=0):
    """Evaluate the model and plot results."""
    noise = tf.random.normal(shape=(len(X_test), latent_dim))
    generated_data = generator.predict(noise)

    fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    
    axs[0, 0].plot(X_test[sample_index], label='Actual', linestyle='--', marker='o')
    axs[0, 0].plot(generated_data[sample_index], label='Generated', linestyle='--', marker='x')
    axs[0, 0].set_title('Actual vs. Predicted Sequence')
    axs[0, 0].set_xlabel('Timesteps')
    axs[0, 0].set_ylabel('Normalized Values')
    
    actual_softmax = tf.nn.softmax(X_test[sample_index], axis=-1)
    generated_softmax = tf.nn.softmax(generated_data[sample_index], axis=-1)
    
    axs[0, 1].plot(actual_softmax, label='Actual Softmax', linestyle='--', marker='o')
    axs[0, 1].plot(generated_softmax, label='Generated Softmax', linestyle='--', marker='x')
    axs[0, 1].set_title('Softmax of Actual vs. Predicted Sequence')
    axs[0, 1].set_xlabel('Timesteps')
    axs[0, 1].set_ylabel('Softmax Values')
    
    avg_actual = np.mean(X_test, axis=0)
    avg_generated = np.mean(generated_data, axis=0)
    
    axs[1, 1].plot(avg_actual, label='Averaged Actual', linestyle='--', marker='o')
    axs[1, 1].plot(avg_generated, label='Averaged Generated', linestyle='--', marker='x')
    axs[1, 1].set_title('Averaged Actual vs. Predicted Sequence')
    axs[1, 1].set_xlabel('Timesteps')
    axs[1, 1].set_ylabel('Normalized Values')
    
    axs[1, 0].plot(history['d_loss'], label='Discriminator Loss')
    axs[1, 0].plot(history['g_loss'], label='Generator Loss')
    axs[1, 0].set_title('Training Loss')
    axs[1, 0].set_xlabel('Steps')
    axs[1, 0].set_ylabel('Loss')

    plt.show()

def run():
    set_seeds(42)

    # Paths to data and models
    training_data_path = 'data/buildSeqInd_Lucky13_5M_ALL.csv'
    # Load and preprocess the data
    training_data = pd.read_csv(training_data_path)
    training_data = training_data.drop(columns=['outputC'])
    
    #training_data = training_data.drop(columns=['STOK1'])
    #training_data = training_data.drop(columns=['RSI'])
    #training_data = training_data.drop(columns=['ATR2'])
    training_data = training_data.drop(columns=['ATR21'])
    #training_data = training_data.drop(columns=['ATR3'])
    training_data = training_data.drop(columns=['ATR31']) 
    training_data = training_data.drop(columns=['ATR32']) 
    training_data = training_data.drop(columns=['ATR34'])   
    #training_data = training_data.drop(columns=['ROC'])     
    #training_data = training_data.drop(columns=['SDKC9'])   
    training_data = training_data.drop(columns=['SDKC91'])  
    training_data = training_data.drop(columns=['SDBB91'])  
    #training_data = training_data.drop(columns=['SDLR310'])

    # Define constants
    timesteps_in = 25  # Number of timesteps in each sequence
    latent_dim_in = 50  # Dimension of the latent space
    batch_size_in = 32
    epochs_in = 100
    steps_per_epoch_in = 3
    patience_in = 10
    
    g_lay1 = 16
    g_lay2 = 128
    g_lay3 = 512
    
    d_lay1 = 512
    d_lay2 = 16
    d_lay3 = 8
    
    g_learn = 0.0001
    d_learn = 0.0001


    cnt_features, x_scalers, y_scaler, X_train, X_val, y_train, y_val = sequence_and_normalize(training_data, timesteps_in)
    out_dim = cnt_features

    generator = build_generatorD(timesteps_in, cnt_features, latent_dim_in, g_lay1, g_lay2, g_lay3)
    discriminator = build_discriminatorX(timesteps_in, cnt_features, d_lay1, d_lay2, d_lay3)
    gan_model = create_gan(generator, discriminator, latent_dim_in, g_learn, d_learn)

    # Train the model
    history = train_model(generator, discriminator, gan_model, X_train, latent_dim_in, batch_size_in, epochs_in, steps_per_epoch_in, patience_in)

    # Evaluate the model
    evaluate_model(generator, discriminator, X_val, latent_dim_in, history, sample_index=0)

if __name__ == "__main__":
    run()
