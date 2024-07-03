import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.config.set_visible_devices([], 'GPU')

def set_seeds(seed=42):
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def preprocess_data(df1, df2, output, seq_length):
    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()
    scaler_output = MinMaxScaler()

    scaled_df1 = scaler1.fit_transform(df1)
    scaled_df2 = scaler2.fit_transform(df2)
    scaled_output = scaler_output.fit_transform(output)

    X1, X2, y = [], [], []

    for i in range(len(df1) - seq_length):
        X1.append(scaled_df1[i:i+seq_length])
        X2.append(scaled_df2[i:i+seq_length])
        y.append(scaled_output[i+seq_length])

    return np.array(X1), np.array(X2), np.array(y), scaler_output



def build_generator(latent_dim, seq_length, n_features1, n_features2, lay1, lay2, lay3):
    input_noise = layers.Input(shape=(latent_dim,))
    input_features1 = layers.Input(shape=(seq_length, n_features1))
    input_features2 = layers.Input(shape=(seq_length, n_features2))

    noise_dense = layers.Dense(seq_length)(input_noise)
    noise_reshape = layers.Reshape((seq_length, 1))(noise_dense)

    x = layers.Concatenate()([noise_reshape, input_features1, input_features2])
    x = layers.LSTM(lay1, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01),kernel_initializer='glorot_uniform')(x)
    
    if lay3 > 0:
        x = layers.LSTM(lay2, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01),kernel_initializer='glorot_uniform')(x)
        x = layers.LSTM(lay3, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.01),kernel_initializer='glorot_uniform')(x)
    else:        
        x = layers.LSTM(lay2, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.01),kernel_initializer='glorot_uniform')(x)
        
    output = layers.Dense(1)(x)

    return tf.keras.Model([input_noise, input_features1, input_features2], output)



def build_discriminator(seq_length, n_features1, n_features2, lay1, lay2, lay3):
    input_features1 = layers.Input(shape=(seq_length, n_features1))
    input_features2 = layers.Input(shape=(seq_length, n_features2))
    input_target = layers.Input(shape=(1,))

    target_dense = layers.Dense(seq_length)(input_target)
    target_reshape = layers.Reshape((seq_length, 1))(target_dense)

    x = layers.Concatenate()([input_features1, input_features2, target_reshape])
    x = layers.LSTM(lay1, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01),kernel_initializer='glorot_uniform')(x)
    
    if lay3 > 0:
        x = layers.LSTM(lay2, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01),kernel_initializer='glorot_uniform')(x)
        x = layers.LSTM(lay3, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.01),kernel_initializer='glorot_uniform')(x)
    else:   
        x = layers.LSTM(lay2, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.01),kernel_initializer='glorot_uniform')(x)                    
        
    output = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model([input_features1, input_features2, input_target], output)



def compile_gan(generator, discriminator, latent_dim, seq_length, n_features1, n_features2, lr_g=0.0002, lr_d=0.0002):
    discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_d), loss='binary_crossentropy', metrics=['accuracy'])
    
    discriminator.trainable = False
    input_noise = layers.Input(shape=(latent_dim,))
    input_features1 = layers.Input(shape=(seq_length, n_features1))
    input_features2 = layers.Input(shape=(seq_length, n_features2))
    
    generated_output = generator([input_noise, input_features1, input_features2])
    gan_output = discriminator([input_features1, input_features2, generated_output])
    
    gan = tf.keras.Model([input_noise, input_features1, input_features2], gan_output)
    gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_g), loss='binary_crossentropy')
    
    gan.summary()
    return gan



def train_gan(generator, discriminator, gan, X1, X2, y, latent_dim, seq_length, epochs=1000, batch_size=32, patience=10, min_delta=0.001):
    best_g_loss = np.inf
    patience_counter = 0

    # Initialize history dictionary
    history = {'d_loss_real': [], 'd_loss_fake': [], 'd_loss': [], 'g_loss': [], 'acc': []}

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X1.shape[0], batch_size)
        real_X1, real_X2, real_y = X1[idx], X2[idx], y[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_y = generator.predict([noise, real_X1, real_X2])
        
        d_loss_real = discriminator.train_on_batch([real_X1, real_X2, real_y], np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch([real_X1, real_X2, fake_y], np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch([noise, real_X1, real_X2], np.ones((batch_size, 1)))
        
        # Since g_loss is a list, extract the loss value
        g_loss_value = g_loss if not isinstance(g_loss, list) else g_loss[0]
        
        # Record losses
        history['acc'].append(100 * d_loss[1])
        history['d_loss'].append(d_loss[0])
        history['g_loss'].append(g_loss_value)

        if epoch % 2 == 0:
            print(f"{epoch} D loss: {d_loss[0]}, acc.: {100 * d_loss[1]} G loss: {g_loss_value}")

        # Early Stopping
        if g_loss_value < best_g_loss - min_delta:
            best_g_loss = g_loss_value
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return history




def evaluate_and_plot(generator, X1, X2, y, scaler_output, latent_dim, seq_length, history_in):
    noise = np.random.normal(0, 1, (X1.shape[0], latent_dim))
    generated_y = generator.predict([noise, X1, X2])
    
    predictions = scaler_output.inverse_transform(generated_y.reshape(-1, 1)).reshape(generated_y.shape)
    actual = scaler_output.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)

    # Classification accuracy for color coding
    ups = 0
    dwns = 0
    colors = []

    for i in range(len(predictions)):
        if ((predictions[i][0] < 0 and actual[i] > 0) or (predictions[i][0] > 0 and actual[i] < 0)):
            colors.append('red')
            dwns += 1
        elif ((predictions[i][0] > 0 and actual[i] > 0) or (predictions[i][0] < 0 and actual[i] < 0)):
            colors.append('green')
            ups += 1
        else:
            colors.append('black')

    print("ups:", ups, " dwns:", dwns)

    #mse = mean_squared_error(actual.flatten(), predictions.flatten())
    #rmse = np.sqrt(mse)
    #r2 = r2_score(actual.flatten(), predictions.flatten())

    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predictions)

    print(f"MSE: {mse}, RMSE: {rmse}, R2: {r2}")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 9))

    # Plot training and validation loss
    ax1.plot(history_in['d_loss'], label='Discriminator Loss')
    ax1.plot(history_in['g_loss'], label='Generator Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    ax2.plot(history_in['acc'], label='Discriminator Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    ax3.plot(actual, label='Actual')
    ax3.plot(predictions, label='Predicted')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Output')
    ax3.legend()

    plt.show()



set_seeds(42)

training_data_path = 'data/IND_LSTM_ALL.csv'
split1 = pd.read_csv(training_data_path)
split1 = split1.drop(columns=['outputC'])

#split1 = split1.drop(columns=['SD79'])
split1 = split1.drop(columns=['SD813'])
split1 = split1.drop(columns=['SD921'])
#split1 = split1.drop(columns=['SDLR9'])
#split1 = split1.drop(columns=['SDLR921'])
split1 = split1.drop(columns=['SDLR2155'])
split1 = split1.drop(columns=['SDLR310'])
#split1 = split1.drop(columns=['SDBB9'])
split1 = split1.drop(columns=['SDBB20'])
split1 = split1.drop(columns=['SDKC9'])
#split1 = split1.drop(columns=['ADX'])
#split1 = split1.drop(columns=['ROC'])
#split1 = split1.drop(columns=['ATR3'])
split1 = split1.drop(columns=['ATR2'])
#split1 = split1.drop(columns=['RSI'])
split1 = split1.drop(columns=['TSI'])
#split1 = split1.drop(columns=['STOD7217'])
split1 = split1.drop(columns=['STOK7217'])
#split1 = split1.drop(columns=['STOD15657'])
#split1 = split1.drop(columns=['STOK15657'])
split1 = split1.drop(columns=['REMA'])
split1 = split1.drop(columns=['output'])

split2 = pd.read_csv(training_data_path)
split2 = split2.drop(columns=['outputC'])



#split2 = split2.drop(columns=['SD79'])
#split2 = split2.drop(columns=['SD813'])
#split2 = split2.drop(columns=['SD921'])
#split2 = split2.drop(columns=['SDLR9'])
split2 = split2.drop(columns=['SDLR921'])
split2 = split2.drop(columns=['SDLR2155'])
#split2 = split2.drop(columns=['SDLR310'])
split2 = split2.drop(columns=['SDBB9'])
split2 = split2.drop(columns=['SDBB20'])
#split2 = split2.drop(columns=['SDKC9'])
split2 = split2.drop(columns=['ADX'])
#split2 = split2.drop(columns=['ROC'])
split2 = split2.drop(columns=['ATR3'])
#split2 = split2.drop(columns=['ATR2'])
split2 = split2.drop(columns=['RSI'])
#split2 = split2.drop(columns=['TSI'])
split2 = split2.drop(columns=['STOD7217'])
#split2 = split2.drop(columns=['STOK7217'])
split2 = split2.drop(columns=['STOD15657'])
#split2 = split2.drop(columns=['STOK15657'])
split2 = split2.drop(columns=['REMA'])
split2 = split2.drop(columns=['output'])

df1 = split1
df2 = split2

output_df = pd.read_csv(training_data_path)
output = output_df['output'].values.reshape(-1, 1)

print(" ")
print(f"df1 number of features: {df1.shape[0]}   {df1.shape[1]}")
print(f"df2 number of features: {df2.shape[0]}   {df2.shape[1]}")
print(f"output number of features: {output.shape[0]}   {output.shape[1]}")

print(" ")

if df1.shape[1] != df2.shape[1]:
    print("ERROR: df1 and df2 must have the same number of features")
    exit()

# Sequence length for LSTM
seq_length = 13

# Preprocess data
X1, X2, y, scaler_output = preprocess_data(df1, df2, output, seq_length)

# Build and compile GAN
latent_dim_in = 20
n_features1 = X1.shape[2]
n_features2 = X2.shape[2]

layer1 = 11
layer2 = 11
layer3 = 0

generator = build_generator(latent_dim_in, seq_length, n_features1, n_features2, layer1, layer2, layer3)
discriminator = build_discriminator(seq_length, n_features1, n_features2, layer1, layer2, layer3)
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gan = compile_gan(generator, discriminator, latent_dim_in, seq_length, n_features1, n_features2, lr_g=0.00001, lr_d=0.00001)

# Train the GAN with early stopping
history_out = train_gan(generator, discriminator, gan, X1, X2, y, latent_dim_in, seq_length, epochs=100, batch_size=32, patience=25, min_delta=0.00001)

# Evaluate and plot results
evaluate_and_plot(generator, X1, X2, y, scaler_output, latent_dim_in, seq_length, history_out)

exit()
