import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the WavKAN component
class WavKAN(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size):
        super(WavKAN, self).__init__()
        self.wavelet_transform = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )

    def call(self, x):
        return tf.sin(self.wavelet_transform(x))

# Define the JacobiKAN component
class JacobiKAN(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size):
        super(JacobiKAN, self).__init__()
        self.jacobi_transform = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )

    def call(self, x):
        return tf.tanh(self.jacobi_transform(x))

# Define the TaylorKAN component
class TaylorKAN(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size):
        super(TaylorKAN, self).__init__()
        self.taylor_transform = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )

    def call(self, x):
        return tf.nn.relu(self.taylor_transform(x))

# Define the RMoK model
class RMoK(tf.keras.Model):
    def __init__(self, input_size, horizon, hidden_size, num_experts=4):
        super(RMoK, self).__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size

        self.wav_kan = WavKAN(input_size, hidden_size)
        self.jacobi_kan = JacobiKAN(input_size, hidden_size)
        self.taylor_kan = TaylorKAN(input_size, hidden_size)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(
                hidden_size,
                activation='relu',
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            ),
            tf.keras.layers.Dense(
                hidden_size,
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            )
        ])
        self.gate = tf.keras.layers.Dense(
            num_experts,
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )
        self.output_layer = tf.keras.layers.Dense(
            horizon,
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )

    def call(self, x):
        wav_out = self.wav_kan(x)
        jacobi_out = self.jacobi_kan(x)
        taylor_out = self.taylor_kan(x)
        mlp_out = self.mlp(x)

        combined_out = tf.concat([wav_out, jacobi_out, taylor_out, mlp_out], axis=1)

        gate_values = tf.nn.softmax(self.gate(x), axis=1)
        gate_values_expanded = tf.expand_dims(gate_values, axis=2)
        repeats = combined_out.shape[1] // self.num_experts
        gate_values_expanded = tf.repeat(gate_values_expanded, repeats=repeats, axis=2)

        weighted_out = tf.reshape(combined_out, [-1, self.num_experts, repeats])
        weighted_out *= gate_values_expanded
        weighted_out = tf.reshape(weighted_out, [-1, self.num_experts * repeats])

        output = self.output_layer(weighted_out)
        return output

# Complete example with data preparation, model instantiation, compilation, training, and visualization
if __name__ == '__main__':
    # Hyperparameters
    input_size = 10
    hidden_size = 64
    horizon = 1
    num_experts = 4

    # Generate synthetic data
    X = np.random.rand(1000, input_size).astype(np.float32)
    y = np.random.rand(1000, horizon).astype(np.float32)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Instantiate the model
    model = RMoK(input_size, horizon, hidden_size, num_experts)

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Define EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Train the model with EarlyStopping
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping]
    )

    # Visualize training and validation loss over epochs
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Make predictions on validation set
    y_pred = model.predict(X_val)

    # Visualize predictions vs. ground truth for a subset
    plt.figure(figsize=(8, 6))
    plt.plot(y_val[:100], label='True Values')
    plt.plot(y_pred[:100], label='Predictions')
    plt.title('Predictions vs. True Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
