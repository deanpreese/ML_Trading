import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# TSMixer Model in TensorFlow/Keras
def create_tsmixer_model(num_features, seq_length, hidden_dim=64):
    input_layer = tf.keras.layers.Input(shape=(seq_length, num_features))

    # Feature mixing
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(input_layer)
    x = tf.keras.layers.Dense(num_features, activation='relu')(x)

    # Time mixing
    x = tf.keras.layers.Permute((2, 1))(x)
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)
    x = tf.keras.layers.Dense(seq_length, activation='relu')(x)
    x = tf.keras.layers.Permute((2, 1))(x)
    

    # Flatten and output
    x = tf.keras.layers.Flatten()(x)
    output_layer = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    
    return model

# Data Preprocessing
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Assuming 'output' is the target and other columns are features
    X = data.drop(columns=['output', 'outputC'])  # Dropping outputC as per your context
    y = data['output']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
            
    
    # Reshape data for sequence modeling
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # [batch_size, seq_length, num_features]

    return X_scaled, y.values

# Metrics Calculation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test).flatten()
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    wins = 0
    losses = 0
    
    for i in range(len(y_test)):
        if (y_pred[i] > 0 and y_test[i] > 0) or (y_pred[i] < 0 and y_test[i] < 0):
            wins += 1
        elif (y_pred[i] > 0 and y_test[i] < 0) or (y_pred[i] < 0 and y_test[i] > 0):
            losses += 1
        elif (y_pred[i] == 0 and y_test[i] == 0):
            wins += 1
        elif (y_pred[i] == 0 and y_test[i] != 0):
            losses += 1
    
    total_samples = wins + losses
    win_percentage = (wins / total_samples) * 100    
    
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")
    print(f"Wins: {wins}, Losses: {losses}, Win Percentage: {win_percentage:.2f}%")
    
    return y_test, y_pred

# Plotting Function
def plot_results(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="True Values", color="blue")
    plt.plot(y_pred, label="Predictions", color="red")
    plt.title("True Values vs Predictions")
    plt.xlabel("Samples")
    plt.ylabel("Output")
    plt.legend()
    plt.show()

# Main Function
def main():
    # Load and preprocess data
    data_file = 'data/Lucky13_3070.csv'
    X, y = load_and_preprocess_data(data_file)
    
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Initialize model
    num_features = X_train.shape[2]
    seq_length = X_train.shape[1]
    model = create_tsmixer_model(num_features=num_features, seq_length=seq_length)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
    model.summary()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint("tsmixer_model.keras", save_best_only=True, monitor='val_loss', mode='min')
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=32, callbacks=[early_stopping, model_checkpoint])
    
    # Load the best model
    #model = tf.keras.models.load_model("tsmixer_model.keras")
    #print("Best model loaded for evaluation.")
    
    # Evaluate the model on the test set
    y_true, y_pred = evaluate_model(model, X_test, y_test)
    
    # Plot the results
    plot_results(y_true, y_pred)

if __name__ == "__main__":
    main()
