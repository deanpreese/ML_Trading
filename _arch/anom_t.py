import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LayerNormalization, Dropout, MultiHeadAttention, Add
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Ensure that GPU is disabled for this script (if desired)
tf.config.set_visible_devices([], 'GPU')

class TransformerAnomalyDetector:
    def __init__(self, input_dim=14, seq_len=30, d_model=64, num_heads=2, ff_dim=128, dropout_rate=0.1):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.model = self.build_model()

    def build_model(self):
        """Build the Transformer-based anomaly detection model."""
        inputs = Input(shape=(self.seq_len, self.input_dim))
        
        # Transformer Encoder Layer
        attention_output = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model)(inputs, inputs)
        attention_output = Dropout(self.dropout_rate)(attention_output)
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
        out1 = Add()([inputs, attention_output])
        
        # Feed Forward Network
        ffn_output = Dense(self.ff_dim, activation='relu')(out1)
        ffn_output = Dense(self.input_dim)(ffn_output)
        ffn_output = Dropout(self.dropout_rate)(ffn_output)
        ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output)
        outputs = Add()([out1, ffn_output])
        
        # Model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        
        return model

    def train(self, X_train, epochs=100, batch_size=32, validation_split=0.1, checkpoint_dir='checkpoints/'):
        """Train the Transformer model with early stopping."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'transformer_model.keras')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        
        self.model.fit(X_train, X_train, 
                       epochs=epochs, 
                       batch_size=batch_size, 
                       validation_split=validation_split, 
                       callbacks=[checkpoint, early_stopping], 
                       verbose=1)
        
        # Save the trained model
        self.model.save(checkpoint_path)

    def load_model(self, checkpoint_dir='checkpoints/'):
        """Load the trained Transformer model."""
        checkpoint_path = os.path.join(checkpoint_dir, 'transformer_model.keras')
        self.model = tf.keras.models.load_model(checkpoint_path)

    def predict_anomalies(self, X):
        """Predict anomalies for a given sequence."""
        reconstructions = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=2)  # Adjust axis for correct MSE calculation
        return mse

    def is_anomaly(self, X, threshold):
        """Determine if the given sequence is an anomaly based on the threshold."""
        mse = self.predict_anomalies(X)
        return mse > threshold

def load_and_prepare_data(filepath, seq_len):
    """Load data from a CSV file and prepare it for modeling."""
    df = pd.read_csv(filepath)
    df = df.drop(columns=['outputC'])
    X = df.drop(columns=['output']).values
    y = df['output'].values
    
    # Creating sequences
    X_sequences = []
    y_sequences = []
    for i in range(len(X) - seq_len):
        X_sequences.append(X[i:i+seq_len])
        y_sequences.append(y[i+seq_len-1])  # predict the output at the end of the sequence
    
    return np.array(X_sequences), np.array(y_sequences)

def main():
    # Set parameters
    filepath = 'data/Lucky13_3070.csv'
    seq_len = 7  # Length of the sequences
    input_dim = 14  # Number of features
    d_model = 32  # Dimension of model
    num_heads = 4  # Number of attention heads
    ff_dim = 64  # Feed-forward network dimension
    dropout_rate = 0.1  # Dropout rate
    
    # Load and prepare data
    X, y = load_and_prepare_data(filepath, seq_len)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize and train the model
    anomaly_detector = TransformerAnomalyDetector(input_dim=input_dim, seq_len=seq_len, d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate)
    anomaly_detector.train(X_train)
    
    # Load the trained model
    anomaly_detector.load_model()
    
    # Determine the threshold based on training data
    mse_train = anomaly_detector.predict_anomalies(X_train)
    threshold = np.mean(mse_train) + 3 * np.std(mse_train)
    
    # Check for anomalies in the test set
    is_anomaly = anomaly_detector.is_anomaly(X_test, threshold)
    
    num_anomalies = np.sum(is_anomaly)
    print(f"Total anomalies detected in test set: {num_anomalies}")
    
    # Evaluate model performance
    y_test_pred = anomaly_detector.model.predict(X_test)
    
    # Use the first feature (or another key feature) for MSE calculation
    y_test_last_step = y_test_pred[:, -1, 0]  # Use the first feature
    mse_test = mean_squared_error(y_test, y_test_last_step)
    print(f"Test set MSE: {mse_test}")

if __name__ == '__main__':
    main()
