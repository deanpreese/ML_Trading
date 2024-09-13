import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
time_steps = 100
input_size = 1
hidden_size = 20
output_size = 1
memory_size = 10
memory_dim = 20

x_train = np.random.rand(100, time_steps, input_size)
y_train = np.random.rand(100, output_size)
x_test = np.random.rand(20, time_steps, input_size)
y_test = np.random.rand(20, output_size)

# Define LSTM Model
class LSTMModel(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=False)
        self.fc = tf.keras.layers.Dense(output_size)

    def call(self, x):
        out = self.lstm(x)
        out = self.fc(out)
        return out

# Define Memory-Augmented Neural Network (MANN) Model
class MANNModel(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size, memory_size, memory_dim):
        super(MANNModel, self).__init__()
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=False)
        self.memory = tf.Variable(tf.zeros((memory_size, memory_dim)), trainable=True)
        self.fc_memory = tf.keras.layers.Dense(output_size, input_shape=(hidden_size + memory_dim,))
        self.fc_write = tf.keras.layers.Dense(memory_dim, input_shape=(hidden_size,))

    def call(self, x):
        out, _ = self.lstm(x)
        memory_weights = tf.nn.softmax(tf.matmul(out, self.memory, transpose_b=True))
        memory_read = tf.matmul(memory_weights, self.memory)
        combined = tf.concat((out, memory_read), axis=1)
        output = self.fc_memory(combined)

        # Write to Memory (simplified for demonstration)
        memory_write = tf.nn.tanh(self.fc_write(out))
        write_weights = tf.nn.softmax(tf.matmul(out, self.memory, transpose_b=True))
        write_weights = tf.reduce_mean(write_weights, axis=0, keepdims=True)
        self.memory.assign((1 - write_weights) * self.memory + write_weights * tf.reduce_mean(memory_write, axis=0, keepdims=True))

        return output

# Initialize models, loss, and optimizer
lstm_model = LSTMModel(input_size, hidden_size, output_size)
mann_model = MANNModel(input_size, hidden_size, output_size, memory_size, memory_dim)

loss_fn = tf.keras.losses.MeanSquaredError()
optimizer_lstm = tf.keras.optimizers.Adam()
optimizer_mann = tf.keras.optimizers.Adam()

# Training Loop
def train_model(model, optimizer, x_train, y_train, num_epochs):
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            predictions = model(x_train)
            loss = loss_fn(y_train, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.numpy():.4f}')

# Train models
print("Training LSTM Model...")
train_model(lstm_model, optimizer_lstm, x_train, y_train, num_epochs=100)
print("Training MANN Model...")
train_model(mann_model, optimizer_mann, x_test, y_test, num_epochs=100)

# Prediction and Plotting
def predict(model, x):
    return model(x).numpy()

y_pred_lstm = predict(lstm_model, x_test)
y_pred_mann = predict(mann_model, x_test)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Values', color='black')
plt.plot(y_pred_lstm, label='LSTM Predictions', color='red', linestyle='--')
plt.plot(y_pred_mann, label='MANN Predictions', color='blue', linestyle='--')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.title('Prediction Comparison')
plt.legend()
plt.show()