import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Dropout, Reshape, Flatten, Attention, Concatenate
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from hmmlearn.hmm import GaussianHMM
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


tf.config.set_visible_devices([], 'GPU')

def print_pretty_results(results):
    print("     ")
    print(f"MSE: {results['MSE']:.4f}")
    print(f"RMSE: {results['RMSE']:.4f}")
    print(f"R2: {results['R2']:.4f}")
    print(f"Wins: {results['Wins']}")
    print(f"Losses: {results['Losses']}")
    print(f"% Correct: {results['% Correct']:.2%}")
    print("     ")

def set_seeds(seed=42):
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def create_ar_features(data, lag):
    ar_features = []
    targets = []
    for i in range(lag, len(data)):
        ar_features.append(data[i-lag:i].flatten())
        targets.append(data[i])
    return np.array(ar_features), np.array(targets)



def create_generator(input_dim, conditioning_dim, lay1, lay2, lay3):
    init = RandomNormal(stddev=0.02)
    input_layer = Input(shape=(input_dim + conditioning_dim,))
    
    x = Dense(lay1, kernel_initializer=init)(input_layer)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
        
    x = Dense(lay2, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Reshape((lay2, 1))(x)
    attention = Attention()([x, x])
    x = Flatten()(attention)
    
    x = Dense(lay3, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    output_layer = Dense(input_dim, activation='tanh')(x)
    return Model(input_layer, output_layer)

def create_discriminator(input_dim, conditioning_dim, lay1, lay2, lay3):
    init = RandomNormal(stddev=0.02)
    input_layer = Input(shape=(input_dim + conditioning_dim,))
    
    x = Dense(lay1, kernel_initializer=init)(input_layer)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(lay2, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(lay3, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Reshape((lay3, 1))(x)
    attention = Attention()([x, x])
    x = Flatten()(attention)
    
    x = Dense(1, activation='sigmoid')(x)
    return Model(input_layer, x)

def create_gan(generator, discriminator, input_dim, conditioning_dim):
    discriminator.trainable = False
    noise_input = Input(shape=(input_dim,))
    condition_input = Input(shape=(conditioning_dim,))
    gan_input = tf.keras.layers.Concatenate()([noise_input, condition_input])
    x = generator(gan_input)
    gan_output = discriminator(tf.keras.layers.Concatenate()([x, condition_input]))
    return Model([noise_input, condition_input], gan_output)

def gradient_penalty(discriminator, real_samples, fake_samples, conditioning_samples, batch_size):
    alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
    interpolated = real_samples + alpha * (fake_samples - real_samples)
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        d_interpolated = discriminator(tf.concat([interpolated, conditioning_samples], axis=1))
    gradients = tape.gradient(d_interpolated, [interpolated])[0]
    grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
    gradient_penalty = tf.reduce_mean((grad_l2 - 1.0) ** 2)
    return gradient_penalty

def train_gan(generator, discriminator, gan, x_data, y_data, epochs, batch_size, conditioning_dim, patience=10, min_delta=0.001):
    history = {'d_loss': [], 'g_loss': [], 'd_acc': []}
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    best_g_loss = np.inf
    patience_counter = 0

    for epoch in range(epochs):
        for _ in range(1):
            idx = np.random.randint(0, x_data.shape[0], batch_size)
            real_samples = x_data[idx]
            real_conditions = y_data[idx]

            noise = np.random.normal(0, 1, (batch_size, x_data.shape[1]))
            gen_input = [noise, real_conditions]
            generated_samples = generator.predict(np.concatenate(gen_input, axis=1))
            
            d_loss_real = discriminator.train_on_batch(np.concatenate([real_samples, real_conditions], axis=1), valid)
            d_loss_fake = discriminator.train_on_batch(np.concatenate([generated_samples, real_conditions], axis=1), fake)
            
            gp = gradient_penalty(discriminator, real_samples, generated_samples, real_conditions, batch_size)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) + 10 * gp  

        for _ in range(1):
            noise = np.random.normal(0, 1, (batch_size, x_data.shape[1]))
            gen_input = [noise, real_conditions]
            g_loss = gan.train_on_batch(gen_input, valid)

        history['d_loss'].append(d_loss[0])
        history['g_loss'].append(g_loss)
        history['d_acc'].append(100 * d_loss_real[1])

        if epoch % 2 == 0:
            print(" ")
            print(f"Epoch {epoch}/{epochs}  Patience: {patience_counter}  Accuracy: {100 * d_loss_real[1]}")
            converted_values = [float(value) for value in d_loss]
            print("Discriminator Loss:", *converted_values)
            converted_values = [float(value) for value in g_loss]
            print("Generator Loss:", *converted_values)
            print(" ")

        g_loss_value = np.mean(history['g_loss'])
        
        if g_loss_value < best_g_loss - min_delta:
            best_g_loss = g_loss_value
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    return history

def evaluate_gan(generator, x_test, y_test):
    noise = np.random.normal(0, 1, (x_test.shape[0], x_test.shape[1]))
    gen_input = [noise, y_test]
    generated_samples = generator.predict(np.concatenate(gen_input, axis=1))

    mse = mean_squared_error(x_test, generated_samples)
    rmse = sqrt(mse)
    r2 = r2_score(x_test, generated_samples)

    wins = np.sum(np.logical_and(generated_samples > 0, x_test > 0))
    losses = abs(x_test.shape[0] - wins)
    per_c = wins/(wins + losses)

    return {"MSE": mse, "RMSE": rmse, "R2": r2, "Wins": wins, "Losses": losses, "% Correct": per_c}

def plot_results(history, x_test, y_test, generator):
    plt.figure(figsize=(10, 5))
    
    noise = np.random.normal(0, 1, (x_test.shape[0], x_test.shape[1]))
    gen_input = [noise, y_test]
    generated_samples = generator.predict(np.concatenate(gen_input, axis=1))
    
    plt.subplot(1, 1,1)
    plt.scatter( x_test, generated_samples, alpha=0.5)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.title('Actual vs Predicted')

    plt.show()

if __name__ == "__main__":
    set_seeds(42)

    data = pd.read_csv("data/buildSeqInd_Lucky13_5M_ALL.csv")
        
    data = data.drop(columns=['ATR21', 'ATR31', 'ATR32', 'ATR34', 'SDKC91', 'SDBB91', 'outputC', 'outputX'])
    X_data = data.drop(columns=['output']).to_numpy()    
    y_data = data['output'].values.reshape(-1, 1)
        
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.4, random_state=42)

    lgb_model = LGBMRegressor()
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_train).reshape(-1, 1).astype(np.float32)


    # Define lag order
    lag_order = 3

    # Create autoregressive features for training and validation data
    X_train_ar, y_train_ar = create_ar_features(X_train, lag_order)
    X_val_ar, y_val_ar = create_ar_features(X_val, lag_order)

    # Train a linear regression model to capture the autoregressive relationships
    ar_model = LinearRegression()
    ar_model.fit(X_train_ar, y_train_ar)

    # Predict the autoregressive components
    X_train_ar_pred = ar_model.predict(X_train_ar)
    X_val_ar_pred = ar_model.predict(X_val_ar)

    # Train the GHMM model with autoregressive features
    ghmm_model = GaussianHMM(n_components=4, covariance_type="full", n_iter=200, random_state=42, verbose=True)
    ghmm_model.fit(X_train_ar_pred)

    # Generate conditional inputs using GHMM
    conditional_input_train = ghmm_model.predict(X_train_ar_pred)
    conditional_input_train = conditional_input_train.reshape(-1, 1)

    conditional_input_val = ghmm_model.predict(X_val_ar_pred)
    conditional_input_val = conditional_input_val.reshape(-1, 1)

    # Adjust input dimensions
    input_dim = X_train.shape[1]
    conditioning_dim = conditional_input_train.shape[1]

    # Create and compile models
    generator = create_generator(input_dim, conditioning_dim, 16, 25, 512)
    discriminator = create_discriminator(input_dim, conditioning_dim, 256, 128, 16)

    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

    gan = create_gan(generator, discriminator, input_dim, conditioning_dim)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5))
    gan.summary()

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Train GAN
    history = train_gan(generator, discriminator, gan, X_train[lag_order:], conditional_input_train, epochs=1000, batch_size=32, conditioning_dim=conditioning_dim, patience=7, min_delta=0.001)

    # Evaluate GAN
    evaluation = evaluate_gan(generator, X_val[lag_order:], conditional_input_val)
    print_pretty_results(evaluation)

    # Plot results
    plot_results(history, X_val[lag_order:], conditional_input_val, generator)
