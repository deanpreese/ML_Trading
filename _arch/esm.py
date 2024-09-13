import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Input, Conv1D, Average, Reshape, Concatenate, ConvLSTM1D, Flatten, SeparableConv1D, LayerNormalization, Bidirectional, Add, Dense,  Dropout, MaxPooling1D, LSTM, MultiHeadAttention, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from ml_model.data_func import sequence_and_normalize, sequence_and_split
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform


tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)

class EchoStateLayer(tf.keras.layers.Layer):
    def __init__(self, units, reservoir_size, spectral_radius=0.95, sparsity=0.1, return_sequences=False, **kwargs):
        super(EchoStateLayer, self).__init__(**kwargs)
        self.units = units
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.W_in = self.add_weight(shape=(input_shape[-1], self.reservoir_size), 
                                    initializer='random_normal', 
                                    trainable=False, 
                                    name="W_in",
                                    dtype=tf.float32)
        
        W_res = np.random.rand(self.reservoir_size, self.reservoir_size).astype(np.float32) - 0.5
        W_res[np.random.rand(*W_res.shape) > self.sparsity] = 0
        eigenvalues, _ = np.linalg.eig(W_res)
        W_res /= np.abs(eigenvalues).max() / self.spectral_radius
        self.W_res = tf.Variable(W_res, trainable=False, name="W_res", dtype=tf.float32)
        
        self.W_out = self.add_weight(shape=(self.reservoir_size, self.units), 
                                     initializer='random_normal', 
                                     trainable=True, 
                                     name="W_out",
                                     dtype=tf.float32)
        
        self.b = self.add_weight(shape=(self.units,), 
                                 initializer='zeros', 
                                 trainable=True, 
                                 name="bias",
                                 dtype=tf.float32)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        batch_size = tf.shape(inputs)[0]
        h = tf.zeros((batch_size, self.reservoir_size), dtype=tf.float32)
        
        def step(h, u_t):
            input_projection = tf.matmul(u_t, self.W_in)
            reservoir_projection = tf.matmul(h, self.W_res)
            h_new = tf.nn.tanh(input_projection + reservoir_projection)
            return h_new
        
        inputs_time_major = tf.transpose(inputs, perm=[1, 0, 2])
        h_states = tf.scan(step, inputs_time_major, initializer=h)
        
        if self.return_sequences:
            outputs = tf.transpose(h_states, perm=[1, 0, 2])  # Convert back to batch major
        else:
            outputs = h_states[-1]
        
        output = tf.matmul(outputs, self.W_out) + self.b
        return output

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.units)  # (batch_size, timesteps, units)
        else:
            return (input_shape[0], self.units)  # (batch_size, units)

# Define the ESN model with stacked ESN layers
def build_esn_model(timesteps, input_dim, reservoir_size=200, output_units=1):
    
    input_layer = Input(shape=(timesteps, input_dim))
    
    l2_reg = l2(0.02)
    initializer = GlorotUniform(seed=42)
    
    reservoir_size = 64

    # First ESN Layer (return full sequence)
    esn_layer_1 = EchoStateLayer(units=64, reservoir_size=reservoir_size, return_sequences=True)(input_layer)
    esn_layer_2 = EchoStateLayer(units=32, reservoir_size=reservoir_size//2, return_sequences=True)(esn_layer_1)
    esn_layer_3 = EchoStateLayer(units=64, reservoir_size=reservoir_size, return_sequences=True)(esn_layer_2)
    x = Conv1D(filters=64, kernel_size=2, activation='relu', kernel_initializer=initializer)(esn_layer_3)
    x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=initializer)(x)
    x = LSTM(32, return_sequences=True, activation='relu')(x)
    x = MaxPooling1D(pool_size=1, strides=1)(x)
    x = LSTM(32, return_sequences=False, activation='relu')(x)
    dense_layer = Dense(16, activation='relu')(x)
    
    # Output layer
    output_layer = Dense(output_units)(dense_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(optimizer=Adam(learning_rate=0.0005), 
                loss='mse', metrics=['mae', tf.keras.metrics.R2Score()])

    model.summary()
    
    return model



# Main function to load data, train model, and evaluate
def main():
   
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
        'data/Lucky13_3070_3.csv',  #9
        'data/Lucky13_3070_oos_5.csv',   
        'data/Lucky13_3070_5.csv',  #11
        'data/new_model_HLC_lucky13.csv', #12
        'data/R_HLC_lucky13.csv', #13
    ]

    file_path = datafile[1]
    timesteps = 14    
 
    feature_dims, X_train, X_val, y_train, y_val = sequence_and_split(file_path, timesteps)


    print(X_train.shape)
    print(y_train.shape)
    
    # Build the ESN model
    model = build_esn_model(timesteps, feature_dims)
    
    # Early Stopping Callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2,
        patience=5, verbose=1,
        mode="auto", min_delta=0.000001,
        cooldown=0, min_lr=0,
    )

    # Train the model with early stopping
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=100,
                        batch_size=32,
                        callbacks=[early_stopping, reduce_lr])
    
    # Evaluate the model
    val_loss = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {val_loss}")
    
    return model, history

# Call the main function
if __name__ == "__main__":
    model, history = main()