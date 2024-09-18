import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

from ml_model.model_stats import gen_reg_stats_x, gen_class_stats
from ml_model.data_func import sequence_and_normalize

tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)

# Define the NTMCell class (same as before, adapted if necessary)
class NTMCell(tf.keras.layers.Layer):
    def __init__(self, memory_size, memory_vector_dim, controller_units, **kwargs):
        super(NTMCell, self).__init__(**kwargs)
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.controller_units = controller_units

        # Store state and output sizes
        self._state_size = [
            tf.TensorShape([controller_units]),                   # Controller hidden state
            tf.TensorShape([controller_units]),                   # Controller cell state
            tf.TensorShape([memory_size, memory_vector_dim]),     # Memory matrix
            tf.TensorShape([memory_size]),                        # Read weights
            tf.TensorShape([memory_size]),                        # Write weights
            tf.TensorShape([memory_vector_dim])                   # Read vector
        ]
        self._output_size = controller_units + memory_vector_dim

        # Controller network (LSTM)
        self.controller = tf.keras.layers.LSTMCell(controller_units)

        # Parameters for read and write heads
        self.read_weighting_layer = tf.keras.layers.Dense(memory_size, activation='softmax')
        self.write_weighting_layer = tf.keras.layers.Dense(memory_size, activation='softmax')
        self.erase_vector_layer = tf.keras.layers.Dense(memory_vector_dim, activation='sigmoid')
        self.add_vector_layer = tf.keras.layers.Dense(memory_vector_dim, activation='tanh')

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, states):
        (
            prev_controller_hidden,
            prev_controller_cell,
            prev_memory,
            prev_read_weights,
            prev_write_weights,
            prev_read_vector
        ) = states

        # Concatenate the inputs with the previous read vector
        controller_input = tf.concat([inputs, prev_read_vector], axis=1)

        # Pass through the controller (LSTM)
        controller_output, [new_hidden_state, new_cell_state] = self.controller(
            controller_input, [prev_controller_hidden, prev_controller_cell]
        )

        # Read from memory
        read_weights = self.read_weighting_layer(controller_output)  # Shape: [batch_size, memory_size]
        read_vector = tf.matmul(tf.expand_dims(read_weights, 1), prev_memory)  # Shape: [batch_size, 1, memory_vector_dim]
        read_vector = tf.squeeze(read_vector, axis=1)  # Shape: [batch_size, memory_vector_dim]

        # Write to memory
        write_weights = self.write_weighting_layer(controller_output)  # Shape: [batch_size, memory_size]
        erase_vector = self.erase_vector_layer(controller_output)      # Shape: [batch_size, memory_vector_dim]
        add_vector = self.add_vector_layer(controller_output)          # Shape: [batch_size, memory_vector_dim]

        # Calculate the erase and add matrices
        erase_matrix = tf.matmul(tf.expand_dims(write_weights, 2), tf.expand_dims(erase_vector, 1))
        add_matrix = tf.matmul(tf.expand_dims(write_weights, 2), tf.expand_dims(add_vector, 1))

        # Update memory
        memory = prev_memory * (1 - erase_matrix) + add_matrix

        # Combine controller output and read vector for the final output
        output = tf.concat([controller_output, read_vector], axis=1)

        # New states
        new_states = [
            new_hidden_state,
            new_cell_state,
            memory,
            read_weights,
            write_weights,
            read_vector
        ]
        return output, new_states

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if dtype is None:
            dtype = tf.float32  # Default data type
        if batch_size is None:
            if inputs is not None:
                batch_size = tf.shape(inputs)[0]
            else:
                raise ValueError("batch_size must be specified if inputs is None")
        # Initialize states with zeros
        return [
            tf.zeros([batch_size, self.controller_units], dtype=dtype),                   # Controller hidden state
            tf.zeros([batch_size, self.controller_units], dtype=dtype),                   # Controller cell state
            tf.zeros([batch_size, self.memory_size, self.memory_vector_dim], dtype=dtype),# Memory matrix
            tf.zeros([batch_size, self.memory_size], dtype=dtype),                        # Read weights
            tf.zeros([batch_size, self.memory_size], dtype=dtype),                        # Write weights
            tf.zeros([batch_size, self.memory_vector_dim], dtype=dtype),                  # Read vector
        ]



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
feature_dims, X_train, X_test, y_train, y_test, scalers = sequence_and_normalize(file_path, time_steps)

print(X_train.shape)
#(26072, 7, 14)
print(y_train.shape)
#(26072,)
print(X_test.shape)
#(11169, 24, 14)
print(y_test.shape)
#(11169,)

# Parameters
batch_size = 32
input_dim = X_train.shape[2]   
memory_size = 256
memory_vector_dim = 256
controller_units = 100

# Initialize the NTM cell
ntm_cell = NTMCell(memory_size, memory_vector_dim, controller_units)

# Wrap the NTM cell in an RNN layer
ntm_layer = tf.keras.layers.RNN(ntm_cell, return_sequences=True)

# Build the model for regression
inputs_placeholder = tf.keras.Input(shape=(time_steps, input_dim))
ntm_outputs = ntm_layer(inputs_placeholder)
# Output layer for regression

print(f"NTM OUT {ntm_outputs.shape}")

final_output = tf.keras.layers.Dense(1)(ntm_outputs)  # No activation (linear activation by default)

model = tf.keras.Model(inputs=inputs_placeholder, outputs=final_output)

# Compile the model using mean squared error loss and include mean absolute error as a metric
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

model.summary(expand_nested=True,show_trainable=True)

# Add early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=100,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=1)


# Step 1: Make predictions on the validation set
predictions = model.predict(X_test)

print(f"Predicitons Shape  {predictions.shape}")

# Step 3: Compute evaluation metrics on the validation set
# Reshape predictions and targets to 1D arrays
#predictions_flat = predictions.reshape(predictions.shape[0],-1)
predictions_flat = predictions[:, 0, 0]
targets_val_flat = y_test.reshape(-1)

#print(f" Pred Flat {predictions_flat.shape} " )
#print(f" Target Flat {targets_val_flat.shape}")

# Calculate MSE and MAE
mse = mean_squared_error(targets_val_flat, predictions_flat)
mae = mean_absolute_error(targets_val_flat, predictions_flat)

#print(f"Validation MSE: {mse}")
#print(f"Validation MAE: {mae}")

evaluate_model(targets_val_flat, predictions_flat )

"""
# Step 4: Visualize predictions vs. actual targets
plt.figure(figsize=(10, 6))

# Select a batch to visualize
batch_to_plot = 0  # You can change this to visualize different batches

plt.plot(predictions[batch_to_plot], label='Predictions', marker='o')
plt.plot(y_test[batch_to_plot], label='Actual Targets', marker='x')

plt.title('Predictions vs. Actual Targets for One Sample in Validation Set')
plt.xlabel('Time Step')
plt.ylabel('Output Value')
plt.legend()
plt.show()
"""

