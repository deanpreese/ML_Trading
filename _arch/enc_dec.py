import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Attention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt


data_loaded = pd.read_csv('data/sm13_3070.csv')
# Drop the outputC column
df = data_loaded.drop(columns=['outputC'])

# Simplified function to create sequences
def create_sequences(data, seq_length):
    xs = [data.iloc[i:i + seq_length, :-1].values for i in range(len(data) - seq_length)]
    ys = data.iloc[seq_length:, -1].values
    return np.array(xs), np.array(ys)

# Parameters
seq_length = 3  # Adjust as necessary
feature_dim = 13  # Assuming 13 features as mentioned

# Create sequences
X, y = create_sequences(df, seq_length)

# Normalize each sequence independently
def normalize_sequences(sequences):
    scalers = {}
    for i in range(sequences.shape[0]):
        scalers[i] = MinMaxScaler()
        sequences[i] = scalers[i].fit_transform(sequences[i])
    return sequences, scalers

X, scalers = normalize_sequences(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define custom metrics
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def r_squared(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res/(SS_tot + tf.keras.backend.epsilon()))

# Define the encoder-decoder model with attention
def create_model(units=50, dropout_rate=0.3, learning_rate=0.001):
    # Encoder
    encoder_inputs = Input(shape=(seq_length, feature_dim))
    encoder_lstm = LSTM(units, return_sequences=True, return_state=True, dropout=dropout_rate, recurrent_dropout=dropout_rate)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Attention
    attention = Attention()([encoder_outputs, encoder_outputs])
    attention_output = Concatenate()([encoder_outputs, attention])

    # Decoder
    decoder_lstm = LSTM(units, return_sequences=True, return_state=True, dropout=dropout_rate, recurrent_dropout=dropout_rate)
    decoder_outputs, _, _ = decoder_lstm(attention_output, initial_state=encoder_states)
    decoder_dense = Dense(1)
    decoder_outputs = decoder_dense(decoder_outputs)

    # Model
    model = Model(encoder_inputs, decoder_outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=[rmse, r_squared, tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError()])
    return model

model = create_model()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping, reduce_lr])

# Save the model
model.save('encoder_decoder_with_attention_model.keras')

# Evaluate the model
val_loss, val_rmse, val_r2, val_mae, val_mape = model.evaluate(X_val)
print(f'Validation Loss: {val_loss:.4f}')
print(f'Validation RMSE: {val_rmse:.4f}')
print(f'Validation R^2: {val_r2:.4f}')
print(f'Validation MAE: {val_mae:.4f}')
print(f'Validation MAPE: {val_mape:.4f}')

# Generate predictions
predictions = model.predict(X_val)

# Reverse scaling for predictions
def reverse_scaling(preds, scalers, seq_length, feature_dim):
    reversed_preds = []
    for i in range(len(preds)):
        temp_input = np.zeros((seq_length, feature_dim))
        temp_input[:, -1] = preds[i][:, 0]
        reversed_pred = scalers[i].inverse_transform(temp_input)
        reversed_preds.append(reversed_pred[:, -1])
    return np.array(reversed_preds)

# Reverse the scaling of predictions
predictions_reversed = reverse_scaling(predictions, scalers, seq_length, feature_dim)

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_val)), y_val, color='blue', label='Actual Values')
plt.plot(range(len(predictions_reversed)), predictions_reversed.flatten(), color='red', linestyle='--', label='Predicted Values')
plt.xlabel("Index")
plt.ylabel("Output")
plt.title("Actual vs. Predicted Output")
plt.legend()
plt.grid(True)
plt.show()
