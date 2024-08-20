import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Define the GAN components
def build_generator(latent_dim, output_dim):
    model = tf.keras.Sequential([
        layers.Dense(32, activation='relu', input_dim=latent_dim),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(output_dim, activation='tanh')
    ])
    return model

def build_discriminator(input_dim):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compile here
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False  # Freeze the discriminator's weights
    model = models.Sequential([generator, discriminator])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# Data preparation
def prepare_data(datafile, sequence_length=30):
    data = pd.read_csv(datafile)
    X = data.drop(columns=['output', 'outputC']).values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# Early Stopping based on a custom metric
def should_stop_training(metric_history, patience):
    if len(metric_history) > patience:
        recent_metrics = metric_history[-patience:]
        if all(recent_metrics[i] >= recent_metrics[i + 1] for i in range(len(recent_metrics) - 1)):
            return True
    return False

# Training the GAN with early stopping and detailed monitoring
def train_gan(generator, discriminator, gan, data, latent_dim, epochs=10000, batch_size=64, patience=25):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    metric_history = []
    
    history = {'d_loss': [], 'g_loss': [], 'd_acc': []}
    best_g_loss = np.inf
    patience_counter = 0
    min_delta = 0.1
    
    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_data = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_data, valid)
        d_loss_fake = discriminator.train_on_batch(gen_data, fake)
        d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])  # Access the scalar loss value
        
        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, valid)
        
        metric_history.append(g_loss)
        
        # Extract accuracy for logging
        d_acc_real = d_loss_real[1] if len(d_loss_real) > 1 else None

        # Monitoring and logging
        if d_acc_real is not None:
            logging.info(f"Epoch {epoch + 1}/{epochs} | D Loss: {d_loss:.4f} | D Accuracy (real): {d_acc_real:.4f} | G Loss: {g_loss[0]:.4f}")
        else:
            logging.info(f"Epoch {epoch + 1}/{epochs} | D Loss: {d_loss:.4f} | G Loss: {g_loss[0]:.4f}")
        
    
        if g_loss[0] < best_g_loss :
            best_g_loss = g_loss[0]
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        history['d_loss'].append(d_loss)  # Appending d_loss[0] as d_loss contains [loss_value, accuracy]
        history['g_loss'].append(g_loss[0])
        history['d_acc'].append(d_loss_real)  # Update to d_loss_real[1] to track accuracy
        
        if (epoch + 1) % 1000 == 0:
            logging.info(f"Saving model at epoch {epoch + 1}")
            generator.save(f'generator_epoch_{epoch + 1}.h5')
            discriminator.save(f'discriminator_epoch_{epoch + 1}.h5')
            

# Anomaly detection with AnoGAN and detailed monitoring
def compute_anomaly_score(generator, discriminator, data_point, latent_dim):
    z = tf.Variable(np.random.normal(0, 1, (1, latent_dim)), dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    @tf.function
    def optimize_z():
        with tf.GradientTape() as tape:
            generated_data = generator(z)
            reconstruction_loss = tf.reduce_mean(tf.abs(data_point - generated_data))
            discriminator_loss = tf.reduce_mean(discriminator(generated_data))
            total_loss = reconstruction_loss + discriminator_loss
        gradients = tape.gradient(total_loss, [z])
        optimizer.apply_gradients(zip(gradients, [z]))
        return total_loss, reconstruction_loss, discriminator_loss
    
    for step in range(500):
        total_loss, reconstruction_loss, disc_loss = optimize_z()
        if step % 50 == 0:
            logging.info(f"Optimization step {step} | Total Loss: {total_loss:.4f} | Reconstruction Loss: {reconstruction_loss:.4f} | Discriminator Loss: {disc_loss:.4f}")
    
    final_generated_data = generator(z).numpy()
    anomaly_score = np.mean(np.abs(data_point - final_generated_data))
    logging.info(f"Anomaly Score: {anomaly_score:.4f}")
    return anomaly_score

# Main function
def main():
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    datafile = 'data/Lucky13_3070.csv'
    X, scaler = prepare_data(datafile)
    
    latent_dim = 16
    output_dim = X.shape[1]
    
    # Build and compile the generator and discriminator
    generator = build_generator(latent_dim, output_dim)
    discriminator = build_discriminator(output_dim)
    
    # Build and compile the GAN
    gan = build_gan(generator, discriminator)
    
    # Train the GAN with early stopping and monitoring
    train_gan(generator, discriminator, gan, X, latent_dim, patience=5)
    
    # Anomaly detection on a sample data point
    sample_index = np.random.randint(0, X.shape[0])
    data_point = np.expand_dims(X[sample_index], axis=0)
    anomaly_score = compute_anomaly_score(generator, discriminator, data_point, latent_dim)
    
    print(f"Anomaly score for the sample data point: {anomaly_score}")

if __name__ == "__main__":
    main()
