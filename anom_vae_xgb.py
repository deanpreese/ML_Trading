import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    features = data.drop(['output', 'outputC'], axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    x_train, x_val = train_test_split(scaled_features, test_size=0.2, random_state=42)
    return x_train, x_val, scaler, data

# Function to build the VAE model
def build_vae(input_dim, latent_dim):
    # Encoder
    inputs = Input(shape=(input_dim,), name='encoder_input')
    h = Dense(256, activation='relu')(inputs)
    h = Dense(128, activation='relu')(h)
    z_mean = Dense(latent_dim, name='z_mean')(h)
    z_log_var = Dense(latent_dim, name='z_log_var')(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Decoder
    decoder_h1 = Dense(128, activation='relu')
    decoder_h2 = Dense(256, activation='relu')
    decoder_out = Dense(input_dim, activation='sigmoid')

    h_decoded = decoder_h1(z)
    h_decoded = decoder_h2(h_decoded)
    outputs = decoder_out(h_decoded)

    # Define custom VAE class
    class VAE(Model):
        def __init__(self, encoder, decoder, **kwargs):
            super(VAE, self).__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder

        def call(self, inputs):
            z_mean, z_log_var, z = self.encoder(inputs)
            reconstructed = self.decoder(z)
            reconstruction_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(inputs, reconstructed), axis=-1)
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            total_loss = K.mean(reconstruction_loss + kl_loss)
            self.add_loss(total_loss)
            return reconstructed

    # Instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # Instantiate decoder model
    decoder_input = Input(shape=(latent_dim,), name='z_sampling')
    h_decoded = decoder_h1(decoder_input)
    h_decoded = decoder_h2(h_decoded)
    decoder_output = decoder_out(h_decoded)
    decoder = Model(decoder_input, decoder_output, name='decoder')

    # Instantiate VAE model
    vae = VAE(encoder, decoder)
    return vae

# Function to train the VAE model
def train_vae(vae, x_train, x_val, epochs=50, batch_size=128):
    vae.compile(optimizer='adam')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = vae.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, None), callbacks=[early_stopping])
    return history

# Function to detect anomalies
def detect_anomalies(vae, x_val, threshold_percentile=95):
    reconstructed = vae.predict(x_val)
    reconstruction_error = np.mean(np.square(x_val - reconstructed), axis=1)
    threshold = np.percentile(reconstruction_error, threshold_percentile)
    anomalies = reconstruction_error > threshold
    return anomalies, reconstructed, reconstruction_error

# Function to plot results
def plot_results(x_val, reconstructed, anomalies):
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(x_val)), x_val[:, 0], label='Actual', alpha=0.5)
    plt.scatter(range(len(x_val)), reconstructed[:, 0], label='Reconstructed', alpha=0.5)
    plt.scatter(np.where(anomalies)[0], x_val[anomalies, 0], color='red', label='Anomalies', marker='x', s=100)
    plt.title('Actual vs. Reconstructed Values with Anomalies Highlighted')
    plt.xlabel('Sample Index')
    plt.ylabel('Feature Value')
    plt.legend()
    plt.show()

# Main function
def main():
    file_path = 'data/Lucky13_3070.csv'
    x_train, x_val, scaler, data = load_and_preprocess_data(file_path)
    input_dim = x_train.shape[1]
    latent_dim = 2
    vae = build_vae(input_dim, latent_dim)
    train_vae(vae, x_train, x_val)
    anomalies, reconstructed, reconstruction_error = detect_anomalies(vae, x_val)
    plot_results(x_val, reconstructed, anomalies)

    # Filtering anomalies for XGBoost
    anomalies_train, _, _ = detect_anomalies(vae, x_train)
    x_train_filtered = x_train[~anomalies_train]
    data_filtered = data.iloc[~anomalies_train]

    # Prepare XGBoost model
    X = data_filtered.drop(['output', 'outputC'], axis=1)
    y = data_filtered['output']
    xgb_model = XGBRegressor()
    xgb_model.fit(X, y)

    # Predictions before and after filtering anomalies
    X_all = scaler.transform(data.drop(['output', 'outputC'], axis=1))
    predictions_before = xgb_model.predict(X_all)

    X_filtered = scaler.transform(data_filtered.drop(['output', 'outputC'], axis=1))
    predictions_after = xgb_model.predict(X_filtered)

    print("Predictions before filtering anomalies:", predictions_before)
    print("Predictions after filtering anomalies:", predictions_after)

if __name__ == '__main__':
    main()
