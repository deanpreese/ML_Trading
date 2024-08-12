import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Sampling function for VAE
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# VAE Model Class
class VAE(Model):
    def __init__(self, input_dim, latent_dim, hidden_dims, name='vae', **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.encoder = self.build_encoder(input_dim, latent_dim, hidden_dims)
        self.decoder = self.build_decoder(input_dim, latent_dim, hidden_dims)
    
    def build_encoder(self, input_dim, latent_dim, hidden_dims):
        inputs = Input(shape=(input_dim,), name='encoder_input')
        h = inputs
        for dim in hidden_dims:
            h = Dense(dim, activation='relu')(h)
        z_mean = Dense(latent_dim, name='z_mean')(h)
        z_log_var = Dense(latent_dim, name='z_log_var')(h)
        
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        return Model(inputs, [z_mean, z_log_var, z], name='encoder')
    
    def build_decoder(self, input_dim, latent_dim, hidden_dims):
        decoder_input = Input(shape=(latent_dim,), name='z_sampling')
        h = decoder_input
        for dim in reversed(hidden_dims):
            h = Dense(dim, activation='relu')(h)
        decoder_output = Dense(input_dim, activation='sigmoid')(h)
        return Model(decoder_input, decoder_output, name='decoder')
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        reconstruction_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(inputs, reconstructed), axis=-1)
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1) * -0.5
        total_loss = K.mean(reconstruction_loss + kl_loss)
        self.add_loss(total_loss)
        return reconstructed

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    features = data.drop(['output', 'outputC'], axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, data, scaler

def train_vae(input_dim, latent_dim, hidden_dims, x_train, x_val, epochs=50, batch_size=128):
    vae = VAE(input_dim, latent_dim, hidden_dims)
    vae.compile(optimizer='adam')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    vae.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, None), callbacks=[early_stopping])
    return vae

def train_isolation_forest(data, contamination=0.05):
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(data)
    scores = iso_forest.decision_function(data)
    return iso_forest, scores

def detect_anomalies_ensemble_with_iforest(vae_ensemble, iso_forest_scores, data, threshold_percentile=95):
    # VAE ensemble reconstruction
    ensemble_reconstructions = np.array([vae.predict(data) for vae in vae_ensemble])
    avg_reconstruction = np.mean(ensemble_reconstructions, axis=0)
    reconstruction_error = np.mean(np.square(data - avg_reconstruction), axis=1)
    
    # Combine VAE reconstruction error with Isolation Forest scores
    combined_scores = (reconstruction_error + iso_forest_scores) / 2
    
    # Determine anomalies based on combined scores
    threshold = np.percentile(combined_scores, threshold_percentile)
    anomalies = combined_scores > threshold
    return anomalies, avg_reconstruction, combined_scores

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

def calculate_statistics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    wins = np.sum((y_pred > 0) & (y_true > 0)) + np.sum((y_pred <= 0) & (y_true <= 0))
    losses = np.sum((y_pred > 0) & (y_true <= 0)) + np.sum((y_pred <= 0) & (y_true > 0))
    return mse, r2, wins, losses

def main():
    file_path = 'data/Lucky13_3070.csv'
    scaled_features, data, scaler = load_and_preprocess_data(file_path)
    x_train, x_val = train_test_split(scaled_features, test_size=0.2, random_state=42)
    input_dim = x_train.shape[1]
    latent_dim = 2

    # Define different architectures for the VAE ensemble
    hidden_dims_list = [
        [256, 128],
        [128, 64],
        [512, 256],
        [256, 64],
        [128, 32]
    ]

    # Train an ensemble of VAE models
    vae_ensemble = [
        train_vae(input_dim, latent_dim, hidden_dims, x_train, x_val)
        for hidden_dims in hidden_dims_list
    ]

    # Train the Isolation Forest
    iso_forest, iso_forest_scores = train_isolation_forest(scaled_features)

    # Detect anomalies using the VAE ensemble and Isolation Forest
    anomalies, reconstructed, combined_scores = detect_anomalies_ensemble_with_iforest(
        vae_ensemble, iso_forest_scores, scaled_features
    )
    
    val_indices = np.arange(len(scaled_features))[-len(x_val):]
    val_anomalies = anomalies[val_indices]
    plot_results(x_val, reconstructed[val_indices], val_anomalies)

    # Filtering anomalies for XGBoost
    data_filtered = data.iloc[~anomalies]

    # Prepare XGBoost model
    X = data_filtered.drop(['output', 'outputC'], axis=1)
    y = data_filtered['output']
    xgb_model = XGBRegressor()
    xgb_model.fit(X, y)

    # Predictions before and after filtering anomalies
    X_all = scaler.transform(data.drop(['output', 'outputC'], axis=1))
    y_all = data['output']
    predictions_before = xgb_model.predict(X_all)

    X_filtered = scaler.transform(data_filtered.drop(['output', 'outputC'], axis=1))
    y_filtered = data_filtered['output']
    predictions_after = xgb_model.predict(X_filtered)

    print(" ")

    # Calculate and print statistics before filtering anomalies
    mse_before, r2_before, wins_before, losses_before = calculate_statistics(y_all, predictions_before)
    print(f"Statistics before filtering anomalies:")
    print(f"MSE: {mse_before} R2: {r2_before}  ")
    print(f"Wins: {wins_before} Losses: {losses_before} Total: {wins_before + losses_before}")
    print(f"Percent: {wins_before/(wins_before+losses_before):.2f}")

    print(" ")

    # Calculate and print statistics after filtering anomalies
    mse_after, r2_after, wins_after, losses_after = calculate_statistics(y_filtered, predictions_after)
    print(f"Statistics after filtering anomalies:")
    print(f"MSE: {mse_after} R2: {r2_after} ")
    print(f"Wins: {wins_after} Losses: {losses_after}  Total: {wins_after + losses_after} ")
    print(f"Percent: {wins_after/(wins_after+losses_after):.2f}")

if __name__ == '__main__':
    main()
