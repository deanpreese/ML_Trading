import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ml_model.model_stats import gen_reg_stats_x, gen_class_stats
from ml_model.data_func import sequence_and_normalize

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

def set_seeds():
    """Set random seeds for reproducibility."""
    tf.config.set_visible_devices([], 'GPU')
    np.random.seed(42)
    tf.random.set_seed(42)


# Define Memory-Augmented Neural Network (MANN) Model
class MANNModel(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size, memory_size, memory_dim):
        super(MANNModel, self).__init__()
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=False)
        self.memory = tf.Variable(tf.zeros((memory_size, memory_dim)), trainable=True)
        self.fc_memory = tf.keras.layers.Dense(output_size, input_shape=(hidden_size + memory_dim,))
        self.fc_write = tf.keras.layers.Dense(memory_dim, input_shape=(hidden_size,))

    def call(self, x):
        out = self.lstm(x)
        memory_weights = tf.nn.softmax(tf.matmul(out, self.memory, transpose_b=True))
        memory_read = tf.matmul(memory_weights, self.memory)
        combined = tf.concat((out, memory_read), axis=1)
        output = self.fc_memory(combined)

        # Write to Memory
        memory_write = tf.nn.tanh(self.fc_write(out))
        write_weights = tf.nn.softmax(tf.matmul(out, self.memory, transpose_b=True))
        write_weights = tf.reduce_mean(write_weights, axis=0, keepdims=True)
        write_weights = tf.reshape(write_weights, (self.memory.shape[0], 1))

        # Perform memory update
        self.memory.assign((1 - write_weights) * self.memory + write_weights * tf.reduce_mean(memory_write, axis=0, keepdims=True))

        return output


def evaluate_model(y_test, y_pred):
    correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(y_test, y_pred)
    print(f"Val MSE: {mse}, Val MAE: {mae}, R2: {r2}")
    print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.4f}")
    print(f"Number of Samples: {total}")


# Training Loop
def train_model(model, optimizer, x_train, y_train, num_epochs):
    loss_fn = tf.keras.losses.MeanSquaredError()
    r2_fn = tf.keras.metrics.R2Score()
    mae_fn = tf.keras.metrics.MeanAbsoluteError()
    
    history = {'loss': [], 'r2': [], 'mae':[]}
    best_loss = np.inf
    patience = 10
    patience_counter = 0
    min_delta=0.001 
    lr_factor = 0.75
    
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            predictions = model(x_train)
            loss = loss_fn(y_train, predictions)
            r2 = r2_fn(y_train, predictions)
            mae = mae_fn(y_train,)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        history['loss'].append(loss)  
        history['r2'].append(r2)
        history["mae"].append(mae)

        if (epoch+1) % 2 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}  Patience: {patience_counter}  MSE {loss:4f}  MAE {mae:4f}   R2 {r2:4f}   lr {optimizer.learning_rate.numpy()}")

        if loss < best_loss - min_delta:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 3:
            new_lr = optimizer.learning_rate.numpy() * lr_factor
            optimizer.learning_rate.assign(new_lr)

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
    return history





def predict(model, x):
    return model(x).numpy()


def plot_results(y_test, y_pred_lstm, y_pred_mann):
    """Plot the predictions against the actual values."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Values', color='black')
    plt.plot(y_pred_lstm, label='LSTM Predictions', color='red', linestyle='--')
    plt.plot(y_pred_mann, label='MANN Predictions', color='blue', linestyle='--')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title('Prediction Comparison')
    plt.legend()
    plt.show()


def main():
    set_seeds()


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
    
    input_size = X_train.shape[1]
    hidden_size = 256
    output_size = 1
    memory_size = 256
    memory_dim = 256

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2,
        patience=5, verbose=1,
        mode="auto", min_delta=0.000001,
        cooldown=0, min_lr=0,
    )

    initial_lr = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=50,  # Adjust as needed
        decay_rate=0.95,
        staircase=True  # If False, it will apply continuous decay
    )

    mann_model = MANNModel(input_size, hidden_size, output_size, memory_size, memory_dim)
    optimizer_mann = tf.keras.optimizers.Adam(learning_rate=initial_lr)


    print("Training MANN Model...")
    train_model(mann_model, optimizer_mann, X_test, y_test, num_epochs=1000)

    y_pred_mann = predict(mann_model, X_test)
    print("\nMANN")
    evaluate_model(y_test, y_pred_mann)

    # Plot results
    #plot_results(y_test, y_pred_lstm, y_pred_mann)


if __name__ == "__main__":
    main()
