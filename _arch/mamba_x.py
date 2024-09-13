import tensorflow as tf
import os
import numpy as np
import pandas as pd

from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ml_model.model_stats import gen_reg_stats_x

# Ensure proper precision for stability
tf.keras.mixed_precision.set_global_policy('float32')

def create_sequences(data, seq_length):
    xs, ys = [], []
    data_len = len(data) - seq_length
    xs = np.zeros((data_len, seq_length, data.shape[1] - 1))
    ys = np.zeros(data_len)
    for i in range(data_len):
        xs[i] = data.iloc[i:(i + seq_length), :-1].values
        ys[i] = data.iloc[i + seq_length, -1]
    return xs, ys

# Define a custom SSM layer with dropout and residual connections
class SSM(layers.Layer):
    def __init__(self, hidden_dim, seq_len):
        super(SSM, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.kernel = self.add_weight(shape=(hidden_dim, hidden_dim),
                                      initializer="he_normal",
                                      trainable=True,
                                      dtype=tf.float32)
        self.bias = self.add_weight(shape=(hidden_dim,),
                                    initializer="zeros",
                                    trainable=True,
                                    dtype=tf.float32)
        self.input_projection = layers.Dense(hidden_dim, dtype=tf.float32, kernel_initializer="he_normal")
        self.dropout = layers.Dropout(0.3)  # Increase dropout to prevent overfitting

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        state = tf.zeros((batch_size, self.hidden_dim), dtype=tf.float32)
        states = []
        for t in range(self.seq_len):
            projected_input = self.input_projection(inputs[:, t, :])
            state = tf.nn.tanh(tf.matmul(state, self.kernel) + self.bias + projected_input)
            state = self.dropout(state)  # Apply dropout
            states.append(state)
        outputs = tf.stack(states, axis=1)
        residual = tf.expand_dims(self.input_projection(inputs[:, -1, :]), axis=1)
        residual = tf.tile(residual, [1, self.seq_len, 1])
        return outputs + residual

# Define a custom Attention Layer
class AttentionLayer(layers.Layer):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_weights = layers.Dense(1, activation='tanh')

    def call(self, inputs):
        # Calculate the attention scores
        scores = self.attention_weights(inputs)
        scores = tf.nn.softmax(scores, axis=1)
        # Apply the attention scores to the inputs
        context_vector = scores * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

# Define the Mamba Model using multiple SSM layers with Attention
class MambaModel(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, num_ssm_layers=2):
        super(MambaModel, self).__init__()
        self.ssm_layers = [SSM(hidden_dim, seq_len) for _ in range(num_ssm_layers)]
        self.attention = AttentionLayer(hidden_dim)
        self.bidirectional_lstm = layers.Bidirectional(
            layers.LSTM(hidden_dim, return_sequences=True, dtype=tf.float32, recurrent_dropout=0.2)
        )
        self.dense = layers.Dense(output_dim, dtype=tf.float32, kernel_regularizer=regularizers.l2(0.01))

    def call(self, inputs):
        x = inputs
        for ssm_layer in self.ssm_layers:
            x = ssm_layer(x)
        x = self.attention(x)  # Apply attention mechanism
        x = tf.expand_dims(x, axis=1)  # Expand dims for LSTM compatibility
        x = self.bidirectional_lstm(x)
        outputs = self.dense(x)
        return outputs[:, -1, :]

    def evaluate_model(self, y_test, y_pred):
        correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(y_test, y_pred)
        print(f"Test MSE: {mse}, Test MAE: {mae}, R2: {r2}")
        print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.3f}")
        print(f"Number of Samples: {total}")

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
        'data/Lucky13_3070_3.csv',  #8
        'data/Lucky13_3070_oos_5.csv',   
        'data/Lucky13_3070_5.csv',  #10
    ]

    file_path = datafile[1]    
    seq_len = 3
    hidden_dims = 32
    
    df = pd.read_csv(file_path)
    data = df.drop(columns=['outputC'])
    
    # Normalize the input data
    scaler = StandardScaler()
    data[data.columns[:-1]] = scaler.fit_transform(data[data.columns[:-1]])
    
    X, y = create_sequences(data, seq_len)
    num_features = X.shape[2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize the Mamba model with multiple SSM layers and attention
    model = MambaModel(input_dim=num_features, hidden_dim=hidden_dims, output_dim=1, seq_len=seq_len, num_ssm_layers=3)
    
    # Compile the model
    optimizer = optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Define early stopping and model checkpointing callbacks
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = callbacks.ModelCheckpoint('best_mamba_model.keras', save_best_only=True, monitor='val_loss')
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    
    # Train the model with early stopping and learning rate scheduling
    model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2, 
              callbacks=[early_stopping, model_checkpoint, lr_scheduler])
    
    # Load the best model
    model.load_weights('best_mamba_model.keras')
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    model.evaluate_model(y_test, y_pred)

if __name__ == "__main__":
    main()
