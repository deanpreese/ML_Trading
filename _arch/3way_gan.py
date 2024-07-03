import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LeakyReLU, Input, LSTM, concatenate
from tensorflow.keras.optimizers import Adam
import os
import joblib
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

def set_seeds(seed=42):
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


# Function to create sequences
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        sequences.append(seq)
    return np.array(sequences)

# Function to create the Generator model
def create_generator(input_shape, units):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units, return_sequences=True))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(input_shape[1]))
    return model

# Function to create the Discriminator model
def create_discriminator(input_shape, units):
    combined_input = Input(shape=input_shape)
    x = LSTM(units)(combined_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(units // 2)(x)
    x = LeakyReLU(alpha=0.2)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(combined_input, output)
    return model

# Function to create and compile the GAN model
def create_gan(generators, discriminator, input_shape):
    discriminator.trainable = False
    gan_inputs = [Input(shape=(input_shape[0], input_shape[1])) for _ in generators]
    gen_outputs = [gen(gan_input) for gen, gan_input in zip(generators, gan_inputs)]
    combined_output = concatenate(gen_outputs, axis=-1)  # Combine along the feature dimension
    gan_output = discriminator(combined_output)
    gan = Model(gan_inputs, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    gan.summary()
    return gan

# Training steps defined as @tf.function
@tf.function
def train_discriminator(discriminator, real_samples, real_labels, fake_samples, fake_labels):
    with tf.GradientTape() as tape:
        real_loss = discriminator(real_samples, training=True)
        fake_loss = discriminator(fake_samples, training=True)
        d_loss_real = tf.keras.losses.binary_crossentropy(real_labels, real_loss)
        d_loss_fake = tf.keras.losses.binary_crossentropy(fake_labels, fake_loss)
        d_loss = 0.5 * tf.add(d_loss_real, d_loss_fake)
    grads = tape.gradient(d_loss, discriminator.trainable_variables)
    if grads:
        discriminator.optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
    return d_loss

@tf.function
def train_generator(gan, noise, valid_y):
    with tf.GradientTape() as tape:
        g_loss = gan(noise, training=True)
        g_loss = tf.keras.losses.binary_crossentropy(valid_y, g_loss)
    grads = tape.gradient(g_loss, gan.trainable_variables)
    if grads:
        gan.optimizer.apply_gradients(zip(grads, gan.trainable_variables))
    return g_loss

def train_gan(generators, discriminator, gan, X_trains, y_train, epochs=1000, batch_size=64, patience=10):
    half_batch = int(batch_size / 2)
    history = {"d_loss": [], "acc": [], "g_loss": []}
    best_g_loss = np.inf
    patience_counter = 0

    for epoch in range(epochs):
        idx = np.random.randint(0, X_trains[0].shape[0], half_batch)
        real_samples = [X[idx] for X in X_trains]
        real_samples = np.concatenate(real_samples, axis=2)
        real_labels = np.ones((half_batch, 1))
        
        noise = [np.random.normal(0, 1, (half_batch, X.shape[1], X.shape[2])) for X in X_trains]
        fake_samples = [gen.predict(n) for gen, n in zip(generators, noise)]
        combined_fake_samples = np.concatenate(fake_samples, axis=2)
        fake_labels = np.zeros((half_batch, 1))
        
        d_loss = train_discriminator(discriminator, real_samples, real_labels, combined_fake_samples, fake_labels)
        
        noise = [np.random.normal(0, 1, (batch_size, X.shape[1], X.shape[2])) for X in X_trains]
        valid_y = np.ones((batch_size, 1))
        g_loss = train_generator(gan, noise, valid_y)
        
        history['acc'].append(100 * d_loss[1])
        history['d_loss'].append(d_loss[0])
        history['g_loss'].append(g_loss[0])

        g_loss_value = g_loss if not isinstance(g_loss, list) else g_loss[0]
        g_loss_value = tf.keras.backend.get_value(g_loss_value[0])  # Convert to scalar
        
        if g_loss_value < best_g_loss:
            best_g_loss = g_loss_value
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter > patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 2 == 0:
            print(f"{epoch} D loss: {d_loss[0]}, acc.: {100 * d_loss[1]} G loss: {g_loss_value}")

    return history

# Function to save models
def save_models(generators, discriminator, gan, path="models"):
    if not os.path.exists(path):
        os.makedirs(path)
    
    for i, gen in enumerate(generators):
        gen.save(os.path.join(path, f'generator_{i}.keras'))
    
    discriminator.save(os.path.join(path, 'discriminator.keras'))
    gan.save(os.path.join(path, 'gan.keras'))

    print(f"Models saved to {path}")

# Function to load models
def load_models(path="models"):
    generators = []
    for i in range(3):  # Assuming 3 generators
        generators.append(load_model(os.path.join(path, f'generator_{i}.keras')))
    
    discriminator = load_model(os.path.join(path, 'discriminator.keras'))
    gan = load_model(os.path.join(path, 'gan.keras'))

    print(f"Models loaded from {path}")
    return generators, discriminator, gan

# Function to save scalers
def save_scalers(scaler_X, scaler_y, path="models"):
    joblib.dump(scaler_X, os.path.join(path, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(path, 'scaler_y.pkl'))
    print(f"Scalers saved to {path}")

# Function to load scalers
def load_scalers(path="models"):
    scaler_X = joblib.load(os.path.join(path, 'scaler_X.pkl'))
    scaler_y = joblib.load(os.path.join(path, 'scaler_y.pkl'))
    print(f"Scalers loaded from {path}")
    return scaler_X, scaler_y

# Function to evaluate the model
def evaluate_gan(generators, X_tests, y_test, scaler_y):
    """Evaluate the GAN model."""
    y_preds = [gen.predict(X_test) for gen, X_test in zip(generators, X_tests)]
    combined_y_pred = np.concatenate(y_preds, axis=2)
    avg_y_pred = np.mean(combined_y_pred, axis=2, keepdims=True)
    
    print("avg_y_pred shape:", avg_y_pred.shape)
    print("First 5 elements of avg_y_pred before reshaping:", avg_y_pred[:5])
    
    y_pred_reshaped = avg_y_pred.reshape(-1, avg_y_pred.shape[-1])
    
    print("y_pred_reshaped shape:", y_pred_reshaped.shape)
    
    y_pred_rescaled = scaler_y.inverse_transform(y_pred_reshaped)
    
    y_test_reshaped = y_test.reshape(-1, y_test.shape[-1])
    y_test_rescaled = scaler_y.inverse_transform(y_test_reshaped)
    
    y_test_flat = y_test_rescaled.flatten()
    y_pred_flat = y_pred_rescaled.flatten()
    
    min_length = min(len(y_test_flat), len(y_pred_flat))
    y_test_flat = y_test_flat[:min_length]
    y_pred_flat = y_pred_flat[:min_length]
    
    print("First 5 elements of y_test_rescaled:", y_test_rescaled[:5])
    print("First 5 elements of y_pred_rescaled:", y_pred_rescaled[:5])
    print("First 5 elements of y_test_flat:", y_test_flat[:5])
    print("First 5 elements of y_pred_flat:", y_pred_flat[:5])
    
    mse = mean_squared_error(y_test_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    
    return y_test_rescaled, y_pred_rescaled, mse, rmse, y_test_flat, y_pred_flat

# Function to plot results
def plot_results(y_test_flat, y_pred_flat, history_in):
    """Plot the results and training history."""
    predictions = y_pred_flat
    actual = y_test_flat

    ups, dwns = 0, 0
    colors = []

    for i in range(len(predictions)):
        if ((predictions[i] < 0 and actual[i] > 0) or (predictions[i] > 0 and actual[i] < 0)):
            colors.append('red')
            dwns += 1
        elif ((predictions[i] > 0 and actual[i] > 0) or (predictions[i] < 0 and actual[i] < 0)):
            colors.append('green')
            ups += 1
        else:
            colors.append('black')

    print("ups:", ups, " dwns:", dwns)

    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predictions)

    print(f"MSE: {mse}, RMSE: {rmse}, R2: {r2}")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 9))

    ax1.plot(history_in['d_loss'], label='Discriminator Loss')
    ax1.plot(history_in['g_loss'], label='Generator Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(history_in['acc'], label='Discriminator Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()

    ax3.plot(actual, label='Actual')
    ax3.plot(predictions, label='Predicted')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Output')
    ax3.legend()

    plt.show()

# ------------------------------------------------------------------------------------------------
# Data Loading and Preprocessing
# ------------------------------------------------------------------------------------------------

set_seeds(42)

training_data_path = 'data/IND_LSTM_ALL.csv'
base = pd.read_csv(training_data_path)
base = base.drop(columns=['outputC'])

g1 = base.drop(columns=['SDLR9', 'SDLR310', 'SDKC9', 'ATR3', 'TSI', 'STOD15657'])
g2 = base.drop(columns=['SD79', 'SDLR921', 'SDBB20', 'ATR3', 'STOD7217', 'REMA'])
g3 = base.drop(columns=['SD813', 'SDLR921', 'SDBB9', 'ADX', 'ATR2', 'STOD7217'])

output = base['output'].values.reshape(-1, 1)

print(" ")
print(f"g1 number of features: {g1.shape[0]}   {g1.shape[1]}")
print(f"g2 number of features: {g2.shape[0]}   {g2.shape[1]}")
print(f"g3 number of features: {g3.shape[0]}   {g3.shape[1]}")
print(f"output number of features: {output.shape[0]}   {output.shape[1]}")
print(" ")

assert g1.shape[1] == g2.shape[1] == g3.shape[1] 
assert output.shape[1] == 1

# Create sequences
sequence_length = 13
X_seq1 = create_sequences(g1, sequence_length)
X_seq2 = create_sequences(g2, sequence_length)
X_seq3 = create_sequences(g3, sequence_length)
y_seq = create_sequences(output, sequence_length)

# Split data into training and testing sets
X_train1, X_test1, y_train, y_test = train_test_split(X_seq1, y_seq, test_size=0.2, random_state=42)
X_train2, X_test2, _, _ = train_test_split(X_seq2, y_seq, test_size=0.2, random_state=42)
X_train3, X_test3, _, _ = train_test_split(X_seq3, y_seq, test_size=0.2, random_state=42)

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled1 = scaler_X.fit_transform(X_train1.reshape(-1, X_train1.shape[2])).reshape(X_train1.shape)
X_test_scaled1 = scaler_X.transform(X_test1.reshape(-1, X_test1.shape[2])).reshape(X_test1.shape)
X_train_scaled2 = scaler_X.fit_transform(X_train2.reshape(-1, X_train2.shape[2])).reshape(X_train2.shape)
X_test_scaled2 = scaler_X.transform(X_test2.reshape(-1, X_test2.shape[2])).reshape(X_test2.shape)
X_train_scaled3 = scaler_X.fit_transform(X_train3.reshape(-1, X_train3.shape[2])).reshape(X_train3.shape)
X_test_scaled3 = scaler_X.transform(X_test3.reshape(-1, X_test3.shape[2])).reshape(X_test3.shape)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

# Create and compile the models
input_shape = (X_train_scaled1.shape[1], X_train_scaled1.shape[2])

generator1 = create_generator(input_shape, 9)
generator2 = create_generator(input_shape, 15)
generator3 = create_generator(input_shape, 50)

discriminator = create_discriminator((input_shape[0], input_shape[1] * 3), 100)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.00002, 0.5), metrics=['accuracy'])

gan = create_gan([generator1, generator2, generator3], discriminator, input_shape)

# Train the GAN model with early stopping
history = train_gan([generator1, generator2, generator3], discriminator, gan, 
                    [X_train_scaled1, X_train_scaled2, X_train_scaled3], y_train_scaled, 
                    epochs=100, batch_size=32, patience=10)

# Evaluate the model
y_test_rescaled, y_pred_rescaled, mse, rmse, y_test_flat, y_pred_flat = evaluate_gan(
    [generator1, generator2, generator3], 
    [X_test_scaled1, X_test_scaled2, X_test_scaled3], 
    y_test_scaled, scaler_y
)
print(f"MSE: {mse}, RMSE: {rmse}")

# Plot the results
plot_results(y_test_flat, y_pred_flat, history)

# Save the models and scalers
# save_models([generator1, generator2, generator3], discriminator, gan)
# save_scalers(scaler_X, scaler_y)
