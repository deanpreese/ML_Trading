import os
import numpy as np
import tensorflow as tf
import pywt
from tensorflow.keras.optimizers import Adam
from ml_model.model_stats import gen_reg_stats_x, gen_class_stats
from ml_model.data_func import sequence_and_normalize


from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Average, GlobalAveragePooling1D, Reshape, Concatenate, ConvLSTM1D, Flatten, SeparableConv1D, LayerNormalization, Bidirectional, Add, Dense,  Dropout, MaxPooling1D, LSTM, MultiHeadAttention, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2

from tensorflow.keras.regularizers import l2

from ml_model.model_stats import gen_reg_stats_x 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau



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

        filters = np.stack([dec_lo, dec_hi], axis=0)  # Shape (2, filter_length)
        filter_length = filters.shape[1]
        num_filters = filters.shape[0]  # Number of filters

        filters = filters.T
        filters = np.expand_dims(filters, axis=1)  # Shape (filter_length, 1, num_filters)
        filters = np.repeat(filters, input_shape[-1], axis=1)  # Repeat for each feature dimension

        filters = tf.convert_to_tensor(filters, dtype=tf.float32)

        # Define Conv1D layer and build it with the correct input shape (num_features)
        self.conv = tf.keras.layers.Conv1D(
            filters=num_filters,
            kernel_size=filter_length,
            strides=1,
            padding='same',
            use_bias=False,
            trainable=self.trainable,
        )
        # Build with the correct input shape (batch_size, sequence_length, num_features)
        self.conv.build(input_shape=(None, input_shape[-2], input_shape[-1]))
        self.conv.set_weights([filters])

    def call(self, inputs):
        # Add a channel dimension to inputs if it is 2D (batch_size, sequence_length)
        if inputs.shape.ndims == 2:
            inputs = tf.expand_dims(inputs, axis=-1)  # Shape becomes (batch_size, sequence_length, 1)

        output = self.conv(inputs)  # Apply wavelet convolution over the sequence
        return output  # Keep the sequence structure intact



# Define the model combining LSTM with WavLayer
def build_model(input_shape, lstm_units=32, wavelet_name='db1'):

    l2_reg = l2(0.01)
    drop_out = 0.2
    initializer = GlorotUniform(seed=42)

    inputs = tf.keras.Input(shape=input_shape)

    # WavLayer processing (wavelet transformation)
    wav_output = WavLayer(wavelet_name=wavelet_name)(inputs)
    wav_output = WavLayer(wavelet_name=wavelet_name)(wav_output)
    
    x = LSTM(64, return_sequences=True, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg)(wav_output)
    #inx = Dropout(self.drop_out)(inx)
    x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=initializer)(x)
    # x = Dropout(self.drop_out)(x)
    x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=initializer)(x)
    #x = Dropout(self.drop_out)(x)
    #x = MaxPooling1D(pool_size=1, strides=1)(x)
    x = LSTM(16, return_sequences=False, activation='relu')(x)
    #x = Dropout(self.drop_out)(x)
    
    outputs = Dense(1, kernel_regularizer=l2_reg)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

# Example training with early stopping and learning rate reduction
def train_model(model, x_train, y_train, x_val, y_val, epochs=100, batch_size=32):
    # Compile the model    
    
    checkpoint_dir = 'checkpoints/'
    trained_dir = 'trained_models/'

    checkpoint_model = os.path.join(checkpoint_dir, 'wav_lstm_model.keras')
    trained_model = os.path.join(trained_dir, 'wav_lstm_model.keras')

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

    return history


def evaluate_model(y_test, y_pred):
    correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(y_test, y_pred)
    print(f"Val MSE: {mse}, Val MAE: {mae}, R2: {r2}")
    print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.4f}")
    print(f"Number of Samples: {total}")



datafile = [ 
    'data/Lucky13_3070_oos.csv',   
    'data/Lucky13_3070.csv',  #1
    'data/ndata_diff_lucky13_3070_oos.csv', 
    'data/ndata_diff_lucky13_3070.csv', #3
    'data/ndata_lucky_13_lag_3070_oos.csv', 
    'data/ndata_lucky13_lag_3070.csv', #5
    'new_model_Z_lucky13_3070_oos.csv',
    'new_model_Z_lucky13_3070.csv', #7,
    'data/Lucky13_3070_oos_3.csv',   
    'data/Lucky13_3070_3.csv',  #8
    'data/Lucky13_3070_oos_5.csv',   
    'data/Lucky13_3070_5.csv',  #10
]

file_path = datafile[1]
time_steps = 24
feature_dims, X_train, X_val, y_train, y_val, scalers = sequence_and_normalize(file_path, time_steps)

print(X_train.shape)
#  (26072, 7, 14)

print(y_train.shape)
#   (26072,)

# Create the model
input_shape = (X_train.shape[1], X_train.shape[2])  # (24, 14)
model = build_model(input_shape)

# Train the model
history = train_model(model, X_train, y_train, X_val, y_val)
y_pred = model.predict(X_val)
evaluate_model(y_val, y_pred)

# Display the model summary




