import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Disable GPU for this script
tf.config.set_visible_devices([], 'GPU')

# Data loading
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocessing
def preprocess_data(data, feature_columns, target_column):
    print("Preprocessing data...")
    scaler = MinMaxScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    data[target_column] = scaler.fit_transform(data[target_column].values.reshape(-1, 1))
    return data, scaler

# Create sequences
def create_sequences(data, feature_columns, target_column, seq_length=60):
    print("Creating sequences...")
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i+seq_length][feature_columns].values
        target = data.iloc[i+seq_length][target_column]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Transformer Block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dense(embed_dim, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Build Transformer model
def build_transformer_model(input_shape, embed_dim, num_heads, ff_dim, feature_dim):
    print("Building Transformer model...")
    
    inputs = Input(shape=input_shape)
    x = Dense(embed_dim)(inputs)  # Project inputs to the embedding dimension
    
    # Encoder
    encoder_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = encoder_block(x)
    x = Dropout(0.1)(x)
    x = Dense(embed_dim, activation="relu")(x)
    
    # Decoder
    decoder_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    decoder_inputs = Input(shape=(None, feature_dim))
    y = Dense(embed_dim)(decoder_inputs)
    y = decoder_block(y)
    y = Add()([x, y])
    y = Dropout(0.1)(y)
    y = TimeDistributed(Dense(1))(y)
    
    model = Model([inputs, decoder_inputs], y)
    return model

# Discriminator for adversarial training
class Discriminator(Model):
    def __init__(self, seq_length, feature_dim):
        super(Discriminator, self).__init__()
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.dense1 = Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dense2 = Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = tf.reshape(inputs, [-1, self.seq_length * self.feature_dim])
        x = self.dense1(x)
        return self.dense2(x)

# Adversarial training step
@tf.function
def train_step(generator, discriminator, real_data, fake_data, cross_entropy, generator_optimizer, discriminator_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(fake_data, training=True)

        # Calculate generator and discriminator losses
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss_real = cross_entropy(tf.ones_like(real_output), real_output)
        disc_loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = disc_loss_real + disc_loss_fake

    # Compute the gradients of the losses with respect to the model parameters
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Clip gradients to prevent exploding gradients
    gradients_of_generator = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in gradients_of_generator]
    gradients_of_discriminator = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in gradients_of_discriminator]

    # Apply the gradients to update the model parameters
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Adversarial training
def adversarial_train(generator, discriminator, real_data, seq_length, feature_dim, epochs, batch_size, cross_entropy, generator_optimizer, discriminator_optimizer):
    for epoch in range(epochs):
        for i in range(0, len(real_data), batch_size):
            real_batch = real_data[i:i+batch_size]
            noise = np.random.normal(0, 1, (batch_size, seq_length, feature_dim))
            fake_batch = generator.predict([real_batch, noise])
            gen_loss, disc_loss = train_step(generator, discriminator, real_batch, fake_batch, cross_entropy, generator_optimizer, discriminator_optimizer)

            print(f'ADV Train Epoch {epoch}, Generator Loss: {gen_loss.numpy()}, Discriminator Loss: {disc_loss.numpy()}')

# Plot results
def plot_results(actuals, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()

# Main function to run the entire script
def main():
    data = load_data("data/buildSeqInd_Lucky13_5M_ALL.csv")
    data = data.drop(columns=['outputC'])
    
    # Adjust these based on your dataset
    feature_columns = list(data.columns[:-1])
    target_column = data.columns[-1]
    
    data, scaler = preprocess_data(data, feature_columns, target_column)
    
    # Split data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    
    # Create sequences
    train_sequences, train_targets = create_sequences(train_data, feature_columns, target_column)
    test_sequences, test_targets = create_sequences(test_data, feature_columns, target_column)
    
    # Hyperparameters
    input_shape = (train_sequences.shape[1], train_sequences.shape[2])
    embed_dim = 256
    num_heads = 2
    ff_dim = 256
    seq_length = train_sequences.shape[1]
    feature_dim = train_sequences.shape[2]
    
    # Build and compile the Transformer model
    model = build_transformer_model(input_shape, embed_dim, num_heads, ff_dim, feature_dim)
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit([train_sequences, train_sequences], train_targets, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping])
    
    # Instantiate and compile the discriminator model
    discriminator = Discriminator(seq_length, feature_dim)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(1e-5)  # Slower learning rate
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)  # Slower learning rate
    
    # Perform adversarial training
    adversarial_train(model, discriminator, train_sequences, seq_length, feature_dim, epochs=100, batch_size=32, cross_entropy=cross_entropy, generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer)
    
    # Evaluate the model on test data
    test_predictions = model.predict([test_sequences, test_sequences])
    
    # Plot the results
    plot_results(test_targets, test_predictions)

if __name__ == "__main__":
    main()
