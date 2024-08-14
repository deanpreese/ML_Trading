import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import ml_model.model_params as mpp

def load_and_prepare_data(filepath):
    """Load data from a CSV file and prepare it for modeling."""
    df = pd.read_csv(filepath)
    df = df.drop(columns=['outputC'])
    X = df.drop(columns=['output'])
    y = df['output']
    return train_test_split(X, y, test_size=0.3, random_state=42)

def create_vae_model(input_dim, layer_sizes):
    """Create and return a VAE model with specified layer sizes."""
    input_layer = Input(shape=(input_dim,))
    x = input_layer
    
    # Build the encoder part of the VAE
    for size in layer_sizes['encoder']:
        x = Dense(size, activation="relu")(x)
    
    # Build the decoder part of the VAE
    for size in layer_sizes['decoder']:
        x = Dense(size, activation='relu')(x)
    
    output_layer = Dense(input_dim, activation='sigmoid')(x)
    vae = Model(inputs=input_layer, outputs=output_layer)
    vae.compile(optimizer='adam', loss='mse')
    
    # Print model summary
    vae.summary()
    
    return vae

def create_isolation_forest_model(contamination):
    """Create and return an Isolation Forest model."""
    return IsolationForest(contamination=contamination, random_state=42)

def train_anomaly_models(X_train, vae_models, iso_forest_models, checkpoint_dir):
    """Train VAE and Isolation Forest models."""
    for idx, vae in enumerate(vae_models):
        # Set up callbacks for early stopping, model checkpoint, and TensorBoard
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        model_checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, f'best_vae_model_{idx}.keras'), save_best_only=True, monitor='val_loss', verbose=1)
        tensorboard_log_dir = os.path.join("logs", f"vae_{idx}")
        tensorboard = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1, write_graph=True)

        # Train VAE with early stopping, checkpointing, and TensorBoard logging
        vae.fit(X_train, X_train, 
                epochs=100, 
                batch_size=32, 
                validation_split=0.1, 
                callbacks=[early_stopping, model_checkpoint, tensorboard],
                verbose=1)  # Use verbose=1 to show the progress bar
    
    for iso_forest in iso_forest_models:
        iso_forest.fit(X_train)

def train_xgboost_model(X_train, y_train):
    """Train and return the XGBoost regressor."""
    xgb_model = XGBRegressor(**mpp.xgbr_set)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def detect_anomalies(X_test, vae_models, iso_forest_models):
    """Detect anomalies in the test set using the trained models."""
    anomaly_flags = []
    
    for vae in vae_models:
        reconstructions = vae.predict(X_test)
        mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
        threshold_value = np.mean(mse) + 3 * np.std(mse)  # Example threshold
        anomaly_flags.append(mse > threshold_value)
    
    for iso_forest in iso_forest_models:
        predictions = iso_forest.predict(X_test)
        anomaly_flags.append(predictions == -1)
    
    return np.any(anomaly_flags, axis=0)

def calculate_baseline_metrics(y_test, y_pred):
    """Calculate and return baseline MSE, R2, and win percentage."""
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    win_percentage = calculate_win_percentage(y_pred, y_test)
    return mse, r2, win_percentage

def calculate_filtered_metrics(X_test, y_test, anomalies, xgb_model):
    """Filter out anomalies and calculate filtered metrics."""
    X_test_filtered = X_test[~anomalies]
    y_test_filtered = y_test[~anomalies]
    y_pred_filtered = xgb_model.predict(X_test_filtered)
    mse_filtered = mean_squared_error(y_test_filtered, y_pred_filtered)
    r2_filtered = r2_score(y_test_filtered, y_pred_filtered)
    return mse_filtered, r2_filtered, y_pred_filtered, y_test_filtered

def calculate_win_percentage(y_pred, y_true):
    """Calculate and return the win percentage."""

    wins = np.sum((y_pred > 0) & (y_true > 0)) + np.sum((y_pred < 0) & (y_true < 0))
    losses = np.sum((y_pred < 0) & (y_true > 0))  + np.sum((y_pred > 0) & (y_true < 0))  
    total = wins+losses
    win_percentage = (wins / total) 
    return win_percentage

def main():
    
    filepath = 'data/Lucky13_3070.csv'
    checkpoint_dir = 'checkpoints/'  # Directory to save model checkpoints
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Step 1: Data Preparation
    X_train, X_test, y_train, y_test = load_and_prepare_data(filepath)
    
    # Step 2: Define the layer sizes for each VAE model
    layer_sizes_list = [
        {'encoder': [8, 64], 'decoder': [32,8]},
        #{'encoder': [13, 16], 'decoder': [8,13]},
        #{'encoder': [32, 128], 'decoder': [128, 32]}
    ]
       
    iso_vars = [
        0.5, 
        #0.25, 
        #0.15, 
        0.2
        ]
    
    
    # Step 3: Create the Anomaly Ensemble
    vae_models = [create_vae_model(X_train.shape[1], layer_sizes) for layer_sizes in layer_sizes_list]
    iso_forest_models = [create_isolation_forest_model(iso_v) for iso_v in iso_vars]  
    
    # Step 4: Train the Anomaly Ensemble
    train_anomaly_models(X_train, vae_models, iso_forest_models, checkpoint_dir)
    
    # Step 5: Train the XGBoost Regressor
    xgb_model = train_xgboost_model(X_train, y_train)
    
    # Step 6: Anomaly Detection on Test Set
    anomalies = detect_anomalies(X_test, vae_models, iso_forest_models)
    
    # Step 7: Calculate Statistics on Anomalies
    num_anomalies = np.sum(anomalies)
    print(f"Total anomalies detected: {num_anomalies}")
    
    # Step 8: Evaluate XGBoost Model (Baseline)
    y_pred = xgb_model.predict(X_test)
    mse, r2, win_percentage_before = calculate_baseline_metrics(y_test, y_pred)
    mse_filtered, r2_filtered, y_pred_filtered, y_test_filtered = calculate_filtered_metrics(X_test, y_test, anomalies, xgb_model)
    win_percentage_after = calculate_win_percentage(y_pred_filtered, y_test_filtered)
    
    print(f" ")
    print(f"Baseline MSE: {mse}, R2: {r2}")
    print(f"Win Percentage: {win_percentage_before}%")
    print(f" ")   
    print(f"Filtered MSE: {mse_filtered}, R2: {r2_filtered}")
    print(f"Win Percentage After Filtering: {win_percentage_after}%")
    print(f" ")
   
# Run the main function with the path to your data
main()
