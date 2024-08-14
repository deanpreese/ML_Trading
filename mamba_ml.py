import tensorflow as tf
from tensorflow.keras import layers, callbacks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


tf.config.set_visible_devices([], 'GPU')

class SSMLayer(layers.Layer):
    def __init__(self, hidden_dim, input_dim, name="SSMLayer", **kwargs):
        super(SSMLayer, self).__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self._initialize_weights()

    def _initialize_weights(self):
        self.A = self.add_weight(shape=(self.hidden_dim, self.hidden_dim),
                                 initializer='glorot_uniform',
                                 trainable=True, name='A')
        self.B = self.add_weight(shape=(self.input_dim, self.hidden_dim),
                                 initializer='glorot_uniform',
                                 trainable=True, name='B')
        self.C = self.add_weight(shape=(self.hidden_dim, 1),
                                 initializer='glorot_uniform',
                                 trainable=True, name='C')
        self.D = self.add_weight(shape=(self.input_dim, 1),
                                 initializer='glorot_uniform',
                                 trainable=True, name='D')

    def _update_state(self, x_t, u_t):
        return tf.nn.tanh(tf.matmul(x_t, self.A) + tf.matmul(u_t, self.B))

    def _compute_output(self, x_t, u_t):
        return tf.matmul(x_t, self.C) + tf.matmul(u_t, self.D)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        initial_state = tf.zeros((batch_size, self.hidden_dim))
        
        def step_fn(previous_state, u_t):
            new_state = self._update_state(previous_state, u_t)
            output = self._compute_output(new_state, u_t)
            return new_state, output
        
        states, outputs = tf.scan(
            fn=lambda prev_state, u_t: step_fn(prev_state[0], u_t),
            elems=tf.transpose(inputs, [1, 0, 2]),
            initializer=(initial_state, tf.zeros([batch_size, 1]))
        )
        
        outputs = tf.transpose(outputs, [1, 0, 2])
        
        return outputs

class MambaModel(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim, name="MambaModel", **kwargs):
        super(MambaModel, self).__init__(name=name, **kwargs)
        self.ssm_layer = SSMLayer(hidden_dim, input_dim)
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        x = self.ssm_layer(inputs)
        output = self.output_layer(x)
        return output

def create_model(input_dim, hidden_dim, output_dim):
    model = MambaModel(input_dim, hidden_dim, output_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse', 
                  metrics=['mae', 'mse'])
    return model

def load_data(file_path):
    df = pd.read_csv(file_path)
    features = df.drop(columns=['output', 'outputC']).values
    target = df['output'].values

    sequence_length = 10
    X, y = [], []

    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(target[i + sequence_length - 1])

    X = np.array(X)
    y = np.array(y)
    y = y.reshape(-1, 1)

    return X, y

def plot_training_history(history):
    plt.figure(figsize=(14, 5))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history.get('val_loss', []), label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history.get('val_mae', []), label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend(loc='upper right')

    plt.show()


def calculate_wins_losses(model, X, y):
    
    predictions = model.predict(X)
    avg_predictions = np.mean(predictions, axis=1)  
    wins = np.sum((avg_predictions > 0) & (y > 0)) + np.sum((avg_predictions < 0) & (y < 0))
    losses = np.sum((avg_predictions < 0) & (y > 0))  + np.sum((avg_predictions > 0) & (y < 0))  
    total = wins+losses
    
    print(f"Wins: {wins}, Losses: {losses}, Win Ratio: {wins / total:.2%}")


class CustomCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1} - Loss: {logs['loss']:.4f}, MAE: {logs['mae']:.4f}")
        if 'val_loss' in logs:
            print(f"Validation Loss: {logs['val_loss']:.4f}, Validation MAE: {logs['val_mae']:.4f}")

def main():
    input_dim = 14
    hidden_dim = 64
    output_dim = 1

    file_path = 'data/Lucky13_3070.csv'
    X, y = load_data(file_path)

    model = create_model(input_dim, hidden_dim, output_dim)

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X, y, epochs=100, batch_size=32, 
                        validation_split=0.2,
                        callbacks=[CustomCallback(), early_stopping])

    plot_training_history(history)
    calculate_wins_losses(model, X, y)

if __name__ == "__main__":
    main()
