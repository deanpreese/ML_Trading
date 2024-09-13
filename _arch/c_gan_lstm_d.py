import numpy as np
import pandas as pd
import tensorflow as tf
import random
from tensorflow.keras.layers import Input, LSTM, Concatenate, Reshape, Flatten, Dense, LeakyReLU, Dropout, BatchNormalization, Layer, Attention, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

#tf.config.set_visible_devices([], 'GPU')

def print_pretty_results(results):
    print("     ")
    print(f"MSE: {results['MSE']:.4f}")
    print(f"RMSE: {results['RMSE']:.4f}")
    print(f"R2: {results['R2']:.4f}")
    print(f"Wins: {results['Wins']}")
    print(f"Losses: {results['Losses']}")
    print(f"% Correct: {results['% Correct']:.2%}")
    print("     ")

# Function to create sequences from data
def create_sequences(data, sequence_length):
    return np.array([data[i:i + sequence_length] for i in range(len(data) - sequence_length + 1)])

# Function to create the generator model with LSTM
def create_generator(input_dim, conditioning_dim, lay1, lay2, lay3, sequence_length):
    init = RandomNormal(stddev=0.02)
    input_layer = Input(shape=(sequence_length, input_dim + conditioning_dim))
    
    x = LSTM(lay1, return_sequences=True, kernel_initializer=init)(input_layer)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
   
    #x = Attention()([x, x])
   
    #x = Bidirectional(LSTM(lay2 ,return_sequences=True))(x)
    x = LSTM(lay2, return_sequences=True, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
        
    x = Attention()([x, x])
    
    x = LSTM(lay3, return_sequences=True)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(input_dim, activation='tanh')(x)
    
    return Model(input_layer, x)


# Function to create the discriminator model with Dense layers
def create_discriminator(input_dim, conditioning_dim, lay1, lay2, lay3, sequence_length):
    init = RandomNormal(stddev=0.02)
    input_layer = Input(shape=(sequence_length, input_dim + conditioning_dim))
    
    x = Flatten()(input_layer)

    slope = 0.2
    dropout = 0.6
    
    x = Dense(lay1, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=slope)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    
    #x = Reshape((lay1, 1))(x)
    #attention = Attention()([x, x])
    #x = Flatten()(attention)    
    
    x = Dense(lay2, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=slope)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    
    #x = Reshape((lay2, 1))(x)
    #attention = Attention()([x, x])
    #x = Flatten()(attention)    
    
    x = Dense(lay3, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=slope)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    
    x = Reshape((lay3, 1))(x)
    attention = Attention()([x, x])
    x = Flatten()(attention)
    
    #x = BatchNormalization()(x)
    #x = Dropout(dropout)(x)    
    
    x = Dense(1, activation='sigmoid')(x)
    return Model(input_layer, x)

# Function to create the GAN model with Dense discriminator
def create_gan(generator, discriminator, input_dim, conditioning_dim, sequence_length):
    discriminator.trainable = False
    noise_input = Input(shape=(sequence_length, input_dim))
    condition_input = Input(shape=(sequence_length, conditioning_dim))
    gan_input = Concatenate()([noise_input, condition_input])
    generated_sequence = generator(gan_input)
    gan_output = discriminator(Concatenate()([generated_sequence, condition_input]))
    return Model([noise_input, condition_input], gan_output)

# Function to set seeds for reproducibility
def set_seeds(seed=42):
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

# Optimized gradient penalty function
@tf.function
def gradient_penalty(discriminator, real_samples, fake_samples, conditioning_samples):
    batch_size = tf.shape(real_samples)[0]
    alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0, dtype=tf.float32)
    real_samples = tf.cast(real_samples, tf.float32)
    fake_samples = tf.cast(fake_samples, tf.float32)
    conditioning_samples = tf.cast(conditioning_samples, tf.float32)
    
    interpolated = real_samples + alpha * (fake_samples - real_samples)
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        interpolated_input = tf.concat([interpolated, conditioning_samples], axis=-1)
        d_interpolated = discriminator(interpolated_input)
    gradients = tape.gradient(d_interpolated, [interpolated])[0]
    grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
    return tf.reduce_mean((grad_l2 - 1.0) ** 2)

# Function to train the GAN model
def train_gan(generator, discriminator, gan, x_data, y_data, epochs, batch_size, conditioning_dim, sequence_length, patience=10, min_delta=0.001):
    history = {'d_loss': [], 'g_loss': [], 'd_acc': []}
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    best_g_loss = np.inf
    patience_counter = 0

    # Create sequences from the data
    x_data = create_sequences(x_data, sequence_length)
    y_data = create_sequences(y_data, sequence_length)

    for epoch in range(epochs):
        for _ in range(1):  # Train discriminator more times
            idx = np.random.randint(0, x_data.shape[0], batch_size)
            real_samples = x_data[idx]
            real_conditions = y_data[idx]

            noise = np.random.normal(0, 1, (batch_size, sequence_length, x_data.shape[2]))
            gen_input = np.concatenate([noise, real_conditions], axis=-1)
            
            generated_samples = generator.predict(gen_input, verbose=0)

            real_input = np.concatenate([real_samples, real_conditions], axis=-1)
            fake_input = np.concatenate([generated_samples, real_conditions], axis=-1)

            d_loss_real = discriminator.train_on_batch(real_input, valid)
            d_loss_fake = discriminator.train_on_batch(fake_input, fake)
            
            gp = gradient_penalty(discriminator, real_samples, generated_samples, real_conditions)
            d_loss = 0.4 * np.add(d_loss_real, d_loss_fake) + 10 * gp  # Gradient penalty coefficient

        for _ in range(1):  # Train generator more times
            noise = np.random.normal(0, 1, (batch_size, sequence_length, x_data.shape[2]))
            gen_input = [noise, real_conditions]
            g_loss = gan.train_on_batch(gen_input, valid)

        history['d_loss'].append(d_loss)
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

        if epoch > 5 and d_loss_real[1] < 0.5:
            patience_counter += 100

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return history

def evaluate_gan(generator, x_test, y_test, conditioning_dim, sequence_length):
    # Create sequences from the test data
    x_test = create_sequences(x_test, sequence_length)
    y_test = create_sequences(y_test, sequence_length)
    
    noise = np.random.normal(0, 1, (x_test.shape[0], sequence_length, x_test.shape[2]))
    gen_input = np.concatenate([noise, y_test], axis=-1)
    generated_samples = generator.predict(gen_input, verbose=0)

    print("Shape of x_test:", x_test.shape)
    print("Shape of generated_samples:", generated_samples.shape)

    mse = mean_squared_error(x_test[:, -1, :], generated_samples[:, -1, :])
    rmse = sqrt(mse)
    r2 = r2_score(x_test[:, -1, :], generated_samples[:, -1, :])

    # Extract the last value of each sequence for comparison
    y_test_last = y_test[:, -1, :]
    generated_last = generated_samples[:, -1, :]

    # Count wins and losses based on the conditions specified
    wins = np.sum(np.logical_or(
        np.logical_and(y_test_last > 0, generated_last > 0),
        np.logical_and(y_test_last < 0, generated_last < 0)
    ))
    
    losses = abs(y_test.shape[0] - wins)
    per_c = wins/(wins + losses)

    return {"MSE": mse, "RMSE": rmse, "R2": r2, "Wins": wins, "Losses": losses, "% Correct": per_c}

    
# Function to evaluate the GAN model
def evaluate_ganX(generator, x_test, y_test, conditioning_dim, sequence_length):
    # Create sequences from the test data
    x_test = create_sequences(x_test, sequence_length)
    y_test = create_sequences(y_test, sequence_length)
    
    noise = np.random.normal(0, 1, (x_test.shape[0], sequence_length, x_test.shape[2]))
    gen_input = np.concatenate([noise, y_test], axis=-1)
    generated_samples = generator.predict(gen_input, verbose=1)

    #print("Shape of x_test:", x_test.shape)
    #print("Shape of generated_samples:", generated_samples.shape)

    mse = mean_squared_error(x_test[:, -1, :], generated_samples[:, -1, :])
    rmse = sqrt(mse)
    r2 = r2_score(x_test[:, -1, :], generated_samples[:, -1, :])

    wins = np.sum(np.logical_and(generated_samples[:, -1, :] > 0, x_test[:, -1, :] > 0))
    losses = x_test.shape[0] - wins

   # Calculate accuracy
    accuracy = wins / y_test.shape[0]

    return {"MSE": mse, "RMSE": rmse, "R2": r2, "Wins": wins, "Losses": losses, "Accuracy": accuracy}

# Function to plot results
def plot_results(history, x_test, y_test, generator, conditioning_dim, sequence_length):
    # Create sequences from the test data
    x_test = create_sequences(x_test, sequence_length)
    y_test = create_sequences(y_test, sequence_length)
    
    plt.figure(figsize=(10, 5))
    
    # Plot discriminator and generator loss
    plt.subplot(1, 2, 1)
    plt.plot(history['d_loss'], label='Discriminator Loss')
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.legend()
    plt.title('Losses')
    
    # Plot discriminator accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['d_acc'], label='Discriminator Accuracy')
    plt.legend()
    plt.title('Discriminator Accuracy')

    plt.figure(figsize=(10, 5))
    # Plot actual vs predicted
    noise = np.random.normal(0, 1, (x_test.shape[0], sequence_length, x_test.shape[2]))
    gen_input = np.concatenate([noise, y_test], axis=-1)
    generated_samples = generator.predict(gen_input, verbose=0)
    
    plt.scatter(x_test[:, -1, :], generated_samples[:, -1, :], alpha=0.5)
    plt.plot([x_test[:, -1, :].min(), x_test[:, -1, :].max()], [x_test[:, -1, :].min(), x_test[:, -1, :].max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')

    plt.show()


def run_multi(input_dim_in, conditioning_dim_in, x_data_in, y_pred_in):
  
    time_step_list = [7,9,11,13,15,17,19,21,23,25,27,31,35, 40,50,60]

    g_layer_list1 = [16, 32, 64,  8, 16, 4, 64, 128, 256, 32, 64, 32]
    g_layer_list2 = [32, 64, 128, 16 , 32, 8, 32, 64, 128, 16, 32, 8]
    g_layer_list3 = [64, 128, 256, 32 , 64, 16, 8,  16, 32, 8, 16, 4]
    
    d_layer_list1 = [64, 128, 256, 32, 64, 32, 16, 32, 64,  8, 16, 4]
    d_layer_list2 = [32, 64, 128, 16, 32, 8, 32, 64, 128, 16 , 32, 8]
    d_layer_list3 = [8,  16, 32, 8, 16, 4, 64, 128, 256, 32 , 64, 16 ]


    set_seeds(42)
    results_list = []
    
    for seq_length in time_step_list:
            for x in range(len(g_layer_list1)):
        
                generator = create_generator(input_dim_in, conditioning_dim_in, g_layer_list1[x], g_layer_list2[x], g_layer_list3[x], seq_length)
                discriminator = create_discriminator(input_dim_in, conditioning_dim_in, d_layer_list1[x], d_layer_list2[x], d_layer_list3[x], seq_length) 
                    
                discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

                gan = create_gan(generator, discriminator, input_dim_in, conditioning_dim_in, seq_length)
                gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
                gan.summary()

                # Train GAN with discriminator
                history = train_gan(generator, discriminator, gan, x_data_in, y_pred_in, 
                    epochs=1000, batch_size=32, conditioning_dim=conditioning_dim_in, 
                    sequence_length=seq_length, patience=10, min_delta=0.001)

                # Evaluate GAN with discriminator
                evaluation = evaluate_gan(generator, x_data_in, y_pred_in, conditioning_dim_in, seq_length)
                evaluation['g_layer1'] = g_layer_list1[x]
                evaluation['g_layer2'] = g_layer_list2[x]
                evaluation['g_layer3'] = g_layer_list3[x]
                evaluation['d_layer1'] = d_layer_list1[x]
                evaluation['d_layer2'] = d_layer_list2[x]
                evaluation['d_layer3'] = d_layer_list3[x]
                evaluation['seq_length'] = seq_length                 
                 
                results_list.append(evaluation)
                print_pretty_results(evaluation)

    
    df = pd.DataFrame(results_list)
    df.to_csv('results.csv', index=False)
    print(df)        
    
    

# Sample Usage
if __name__ == "__main__":
    set_seeds(42)

    # Example data loading (replace with your actual dataset)
    train_file = pd.read_csv("data/buildSeqInd_Lucky13_5M_ALL.csv")
    data = train_file.drop(columns=['outputC'])

    x_data = data.drop(columns=['output']).to_numpy().astype(np.float32)
    y_data = data['output'].values.reshape(-1, 1).astype(np.float32)
    
    X_train, X_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    
    
    lgb_params = {
        'n_estimators': 250,
        'objective': 'regression',
        'min_child_samples': 7,
        'subsample': 1,
        'num_leaves': 35,
        'colsample_bytree': 1,
        'random_state': 0,
        'n_jobs': -1,
        'learning_rate': 0.01,
        'verbose': 1,
    }

    lgb_model = LGBMRegressor(**lgb_params)
    lgb_model2 = LGBMRegressor()
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_train).reshape(-1, 1).astype(np.float32)
    lgb_model2.fit(X_train, y_train)
    y_pred_lgb2 = lgb_model.predict(X_train).reshape(-1, 1).astype(np.float32)

    xgb_params = {
        'max_depth': 6,
        'learning_rate': 0.07,
        'n_estimators': 125,
        'tree_method': "hist",
        'eval_metric': "mae",
        "verbosity": 2
    }

    xgb_model = XGBRegressor(**xgb_params)
    xgb_model2 = XGBRegressor()
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_train).reshape(-1, 1).astype(np.float32)
    xgb_model2.fit(X_train, y_train)
    y_pred_xgb2 = xgb_model.predict(X_train).reshape(-1, 1).astype(np.float32)
    y_pred = (y_pred_lgb + y_pred_xgb + y_pred_lgb2 + y_pred_xgb2)/4
    
    # Create and compile models
    input_dim = x_data.shape[1]
    conditioning_dim = y_pred.shape[1]
    
    
    
    rm = False
        
    
    if rm:
        run_multi(input_dim, conditioning_dim, x_data, y_pred)
    else:
        
        sequence_length = 13
        
        #generator = create_generator(input_dim, conditioning_dim, 8, 16, 32, sequence_length)
        generator = create_generator(input_dim, conditioning_dim, 32, 64, 512, sequence_length)
        
        #discriminator = create_discriminator(input_dim, conditioning_dim, 128, 64, 32, sequence_length)
        discriminator = create_discriminator(input_dim, conditioning_dim, 256, 128, 16, sequence_length)    
                    
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

        gan = create_gan(generator, discriminator, input_dim, conditioning_dim, sequence_length)
        gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
        gan.summary()

        scaler = MinMaxScaler()
        #scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Train GAN with discriminator
        
        history = train_gan(generator, discriminator, gan, X_train, y_pred, 
            epochs=1000, batch_size=32, conditioning_dim=conditioning_dim, 
            sequence_length=sequence_length, patience=10, min_delta=0.001)

        # Evaluate GAN with discriminator
        evaluation = evaluate_gan(generator, X_val, y_val, conditioning_dim, sequence_length)
        print_pretty_results(evaluation)


        # Plot results
        #plot_results(history, x_data, y_pred, generator, conditioning_dim, sequence_length)
