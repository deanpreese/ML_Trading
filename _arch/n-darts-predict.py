import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel, NHiTSModel
from darts.metrics import mae, mape, rmse, coefficient_of_variation, dtw_metric
import joblib

# Data loading
def load_data(file_path):
    data = pd.read_csv(file_path)
    print(data.head())
    return data

# Preprocessing
def preprocess_data(data, target_column):
    series = TimeSeries.from_dataframe(data, value_cols=target_column).astype(np.float32)
    scaler = Scaler()
    scaled_series = scaler.fit_transform(series)
    
    print(scaled_series)
    return scaled_series, scaler

# Plot results
def plot_results(actuals, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()

# Load model and run predictions
def run_predictions(model_path, data_path, target_column):
    
    model = NHiTSModel.load(model_path)
    
    # Load and preprocess out-of-sample data
    data = load_data(data_path)
    series, scaler = preprocess_data(data, target_column)
    
    # Make predictions
    predictions = model.predict(len(series))
    
    # Inverse transform the predictions and actual values
    actual_values = scaler.inverse_transform(series).values()
    predicted_values = scaler.inverse_transform(predictions).values()
    
    # Print descriptive statistics for model performance
    mae_score = mae(series, predictions)
    mape_score = mape(series, predictions)
    rmse_score = rmse(series, predictions)
    cov = coefficient_of_variation(series, predictions)
    dtw = dtw_metric(series, predictions)
    
    print(f'Dynamic Time Warping (DTW): {dtw:.4f}')
    print(f'Coefficient of Variation (CoV): {cov:.4f}')
    print(f'Mean Absolute Error (MAE): {mae_score:.4f}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape_score:.4f}%')
    print(f'Root Mean Squared Error (RMSE): {rmse_score:.4f}')
    
    plot_results(actual_values, predicted_values)

if __name__ == "__main__":
    MODEL_PATH = "nbeats_model.pk"  # Path to your saved model
    DATA_PATH = "data/lucky13_oos.csv"  # Path to your out-of-sample data
    TARGET_COLUMN = 'output'  # Replace with your actual target column name
    
    run_predictions(MODEL_PATH, DATA_PATH, TARGET_COLUMN)
