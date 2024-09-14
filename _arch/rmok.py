import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import pywt

from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from ml_model.model_stats import gen_reg_stats_x, gen_class_stats

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)

class WavLayer(tf.keras.layers.Layer):
    def __init__(self, wavelet_name='db1', trainable=False):
        super(WavLayer, self).__init__()
        self.wavelet_name = wavelet_name
        self.trainable = trainable

    def build(self, input_shape):
        wavelet = pywt.Wavelet(self.wavelet_name)
        dec_lo = np.array(wavelet.dec_lo)
        dec_hi = np.array(wavelet.dec_hi)

        filters = np.stack([dec_lo, dec_hi], axis=0)
        filter_length = filters.shape[1]
        num_filters = filters.shape[0]  # Number of filters

        filters = filters.T
        filters = np.expand_dims(filters, axis=1)  # Shape (filter_length, 1, num_filters)

        filters = tf.convert_to_tensor(filters, dtype=tf.float32)

        # Define Conv1D layer and build it with the correct input shape (1 channel)
        self.conv = tf.keras.layers.Conv1D(
            filters=num_filters,
            kernel_size=filter_length,
            strides=1,
            padding='same',
            use_bias=False,
            trainable=self.trainable,
        )
        # Build with the correct input shape (batch_size, sequence_length, 1)
        self.conv.build(input_shape=(None, input_shape[-1], 1))
        self.conv.set_weights([filters])

    def call(self, inputs):
        # Add a channel dimension to inputs if it is 2D (batch_size, sequence_length)
        if inputs.shape.ndims == 2:
            inputs = tf.expand_dims(inputs, axis=-1)  # Shape becomes (batch_size, sequence_length, 1)

        output = self.conv(inputs)
        output = tf.reshape(output, [tf.shape(inputs)[0], -1])  # Flatten the output
        return output


class LegendrePoly(tf.keras.layers.Layer):
    def __init__(self, degree):
        super(LegendrePoly, self).__init__()
        self.degree = degree

    def call(self, x):
        x = tf.clip_by_value(x, -1.0, 1.0)
        polys = [tf.ones_like(x), x]
        for n in range(1, self.degree):
            n_float = tf.cast(n, x.dtype)
            Pn = ((2.0 * n_float + 1.0) * x * polys[n] - n_float * polys[n - 1]) / (n_float + 1.0)
            polys.append(Pn)
        polys_stack = tf.stack(polys[:self.degree + 1], axis=-1)

        # Flatten the output to ensure it's compatible with the other features
        output_shape = [tf.shape(x)[0], -1]
        return tf.reshape(polys_stack, output_shape)


# Define the Taylor Polynomial Layer
class TaylorPolynomial(tf.keras.layers.Layer):
    def __init__(self, degree, **kwargs):
        super(TaylorPolynomial, self).__init__(**kwargs)
        self.degree = degree

    def call(self, x):
        outputs = [x]
        x_dtype = x.dtype
        for k in range(2, self.degree + 1):
            k_float = tf.cast(k, x_dtype)
            term = tf.math.pow(x, k_float) / tf.math.exp(tf.math.lgamma(k_float + 1.0))
            outputs.append(term)
        return tf.concat(outputs, axis=-1)
    

class RMoK(Model):
    def __init__(
        self,
        input_size,
        horizon,
        hidden_size,
        num_experts=4,
        polynomial_degree=5,
        use_wavlayer=True,
        use_legendrepoly=True,
        use_taylorpoly=True
    ):
        super(RMoK, self).__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.polynomial_degree = polynomial_degree
        self.use_wavlayer = use_wavlayer
        self.use_legendrepoly = use_legendrepoly
        self.use_taylorpoly = use_taylorpoly
        self.l2_reg = l2(0.01)
        self.initializer = GlorotUniform(seed=42)

    def build(self, input_shape):
        # Correct the feature sizes
        self.wav_size = 2 * self.input_size if self.use_wavlayer else 0
        self.legendre_size = self.input_size * (self.polynomial_degree + 1) if self.use_legendrepoly else 0
        self.taylor_size = self.input_size * self.polynomial_degree if self.use_taylorpoly else 0
        self.mlp_size = self.hidden_size
        self.total_feature_size = self.wav_size + self.legendre_size + self.taylor_size + self.mlp_size

        if self.use_wavlayer:
            self.wav_kan = WavLayer(wavelet_name='db1', trainable=False)

        if self.use_legendrepoly:
            self.jacobi_kan = LegendrePoly(degree=self.polynomial_degree)

        if self.use_taylorpoly:
            self.taylor_kan = TaylorPolynomial(degree=self.polynomial_degree)

        self.mlp = tf.keras.Sequential([
            layers.InputLayer(input_shape=(self.input_size,)),
            layers.Dense(self.hidden_size*2, activation='relu', kernel_initializer=self.initializer),
            #layers.Dense(self.hidden_size, activation='relu', kernel_initializer=self.initializer),
            #layers.Dense(self.hidden_size//2, activation='relu', kernel_initializer=self.initializer),
            layers.Dense(self.hidden_size, activation='relu', kernel_initializer=self.initializer),
        ])

        # Gating and output layer
        self.gate = layers.Dense(self.num_experts, kernel_initializer=self.initializer)
        self.output_layer = layers.Dense(units=self.horizon, kernel_initializer=self.initializer)
        super(RMoK, self).build(input_shape)


    def call(self, inputs):
        features = []

        # Apply WavLayer if enabled
        if self.use_wavlayer:
            wavelet_features = self.wav_kan(inputs)
            #print(f"WavLayer output shape: {wavelet_features.shape}")  # Debugging shape
            features.append(wavelet_features)

        # Apply Legendre Polynomial Layer if enabled
        if self.use_legendrepoly:
            legendre_features = self.jacobi_kan(inputs)
            legendre_features = tf.reshape(legendre_features, [tf.shape(inputs)[0], -1])  # Flatten the output
            #print(f"LegendrePoly output shape: {legendre_features.shape}")  # Debugging shape
            features.append(legendre_features)

        # Apply Taylor Polynomial Layer if enabled
        if self.use_taylorpoly:
            taylor_features = self.taylor_kan(inputs)
            #print(f"TaylorPolynomial output shape: {taylor_features.shape}")  # Debugging shape
            features.append(taylor_features)

        # Process through MLP
        mlp_features = self.mlp(inputs)
        #print(f"MLP output shape: {mlp_features.shape}")  # Debugging shape
        features.append(mlp_features)

        # Concatenate all features
        combined_features = tf.concat(features, axis=-1)
        #print(f"Combined features shape: {combined_features.shape}")  # Debugging shape

        # Ensure combined_features has a fully defined shape
        combined_features.set_shape([None, self.total_feature_size])

        # Compute gating mechanism for expert outputs
        gate_output = self.gate(combined_features)
        gate_output = tf.nn.softmax(gate_output)  # Convert to probabilities

        # Compute final prediction
        output = self.output_layer(combined_features)

        return output




    def evaluate_model(self, y_test, y_pred):
        correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(y_test, y_pred)
        print(f"Val MSE: {mse}, Val MAE: {mae}, R2: {r2}")
        print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.3f}")
        print(f"Number of Samples: {total}")

        
        

# Main script
if __name__ == '__main__':
    datafile = [
        'data/Lucky13_3070_oos.csv',
        'data/Lucky13_3070.csv',  # 1
        'data/ndata_diff_lucky13_3070_oos.csv',
        'data/ndata_diff_lucky13_3070.csv',  # 3
        'data/ndata_lucky_13_lag_3070_oos.csv',
        'data/ndata_lucky13_lag_3070.csv',  # 5
        'new_model_Z_lucky13_3070_oos.csv',
        'new_model_Z_lucky13_3070.csv',  # 7
        'data/Lucky13_3070_oos_3.csv',
        'data/Lucky13_3070_3.csv',  # 9
        'data/Lucky13_3070_oos_5.csv',
        'data/Lucky13_3070_5.csv',  # 11
        'data/new_model_HLC_lucky13.csv',  # 12
    ]

    file_path = datafile[1]
    df = pd.read_csv(file_path)
    df = df.drop(columns=['outputC'])
    X = df.drop(columns=['output']).values
    y = df['output'].values

    # Hyperparameters
    input_size = X.shape[1]  # Number of features
    hidden_size = 32
    horizon = 1
    num_experts = 4
    polynomial_degree = 2  # Set the degree for polynomial layers

    use_wavlayer = True
    use_legendrepoly = False
    use_taylorpoly = False

    model = RMoK(
        input_size,
        horizon,
        hidden_size,
        num_experts,
        polynomial_degree=polynomial_degree,
        use_wavlayer=use_wavlayer,
        use_legendrepoly=use_legendrepoly,
        use_taylorpoly=use_taylorpoly
    )

    checkpoint_dir = 'checkpoints/'
    trained_dir = 'trained_models/'

    checkpoint_model = os.path.join(checkpoint_dir, 'mcnn_model.keras')
    trained_model = os.path.join(trained_dir, 'mcnn_model.keras')

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', tf.keras.metrics.R2Score()])
    model.summary(expand_nested=True, show_trainable=True)

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2,
        patience=5, verbose=1,
        mode="auto", min_delta=0.000001,
        cooldown=0, min_lr=0,
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_model,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False, mode='min'
    )

    # Split into train and test sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        initial_epoch=0, epochs=150,
        batch_size=32,
        callbacks=[
            early_stopping,
            reduce_lr,
            model_checkpoint
        ]
    )

    # Make predictions on validation set
    y_pred = model.predict(X_val)
    model.evaluate_model(y_val, y_pred)

    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    axs[0].plot(history.history['loss'], label='Training Loss')
    axs[0].plot(history.history['val_loss'], label='Validation Loss')
    axs[0].set_title('Training and Validation Loss over Epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(y_val[:100], label='True Values')
    axs[1].plot(y_pred[:100], label='Predictions')
    axs[1].set_title('Predictions vs. True Values')
    axs[1].set_xlabel('Sample Index')
    axs[1].set_ylabel('Value')
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    """