import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NHiTSModel
from torchmetrics import MetricCollection
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from darts.utils.likelihood_models import QuantileRegression
from darts.metrics import mae, mape, rmse, coefficient_of_variation, dtw_metric
from torchmetrics.regression import SpearmanCorrCoef, PearsonCorrCoef, R2Score, MeanAbsoluteError 
from torchmetrics.regression import MeanSquaredError, PearsonCorrCoef, MeanAbsolutePercentageError, CosineSimilarity

import joblib

def process_data(data, feature_columns, target_column, split):
    series = TimeSeries.from_dataframe(data).astype(np.float32)   
    train, test = series.split_after(split)
    X_train = train.drop_columns(target_column)
    X_test = test.drop_columns(target_column)
    y_train = train.drop_columns(feature_columns)
    y_test = test.drop_columns(feature_columns)
    
    scaler = Scaler()
    X_train = scaler.fit_transform(X_train) 
    X_test = scaler.transform(X_test.astype(np.float32))   
    
    return X_train, X_test, y_train, y_test, scaler


def plot_multi_results(actuals, predictions, predictions2, predictions3, last_x_rows):

    predictions = predictions[-last_x_rows:]
    predictions2 = predictions2[-last_x_rows:]
    predictions3 = predictions3[-last_x_rows:]
    actuals = actuals[-last_x_rows:]   
    
    plt.figure(figsize=(12, 6))
    actuals.plot(label='Actual')    
    predictions.plot(label='Predicted', color='blue' )
    predictions2.plot(label='Predicted2', color='green')
    predictions3.plot(label='Predicted3', color='red')

    #plt.xlabel('Time')
    #plt.ylabel('Stock Price')
    #plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()




# Plot results
def plot_results(actuals, predictions, last_x_rows=100):
    
    predictions = predictions[-last_x_rows:]
    actuals = actuals[-last_x_rows:]   
    
    plt.figure(figsize=(12, 6))
    #plt.plot(actuals, label='Actual')
    #plt.plot(predictions, label='Predicted')
    actuals.plot(label='Actual')    
    predictions.plot(label='Predicted')

    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()

def generate_statistics(test_series, predictions):
    # Print descriptive statistics for model performance
    mae_score = mae(test_series, predictions)
    mape_score = mape(test_series, predictions)
    rmse_score = rmse(test_series, predictions)
    cov = coefficient_of_variation(test_series, predictions)
    dtw = dtw_metric(test_series, predictions)
    
    print(f'Dynamic Time Warping (DTW): {dtw:.4f}')
    print(f'Coefficient of Variation (CoV): {cov:.4f}')
    print(f'Mean Absolute Error (MAE): {mae_score:.4f}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape_score:.4f}%')
    print(f'Root Mean Squared Error (RMSE): {rmse_score:.4f}')

def train_and_save_model(train_target, val_target, train_covariates, val_covariates, 
                         input_chunk_length, output_chunk_length, 
                         n_epochs, num_stacks, num_blocks, num_layers, layer_widths, 
                         patience_val, min_delta_val     
                          ):
        
    # TensorBoard logger
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Early stopping callback
    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=patience_val,
        min_delta=min_delta_val,
        verbose=True,
        mode='min'
    )

    pl_trainer_kwargs = {
        "callbacks": [early_stopper, lr_monitor],
        "accelerator": "gpu",
        "devices": [0]
    }
    
    
    metric_collection = MetricCollection([
        MeanAbsoluteError(),
        MeanSquaredError(), 
        MeanAbsolutePercentageError(), 
    ])
    
    
    # Build and train the NHiTS model
    model = NHiTSModel(
        input_chunk_length=input_chunk_length, 
        output_chunk_length=output_chunk_length, 
        n_epochs=n_epochs,
        batch_size=32, 
        random_state=42, 
        num_stacks=num_stacks, 
        num_blocks=num_blocks, 
        num_layers=num_layers, 
        layer_widths=layer_widths,
        pl_trainer_kwargs=pl_trainer_kwargs,
        likelihood=QuantileRegression(),
        torch_metrics=metric_collection,
        log_tensorboard=True
    )
    
    
    now = datetime.now()
    ts = now.strftime("%Y-%m%d-%H-%M-%S")
    model_base_name = f"dart_NHiTSModel_{input_chunk_length}-{output_chunk_length}-{ts}"
    model_save_path = f"darts_saved_models/{model_base_name}.pk"
    
    model.fit(series=train_target, val_series=val_target, 
              past_covariates=train_covariates, val_past_covariates=val_covariates)
    
    model.save(model_save_path)

    return model

# Main function to run the entire script
def main():
    file_path = "data/buildSeqInd_Lucky13_5M_ALL.csv"
    #file_path = "data/Fractal_ALL_5M_X.csv"

    data = pd.read_csv(file_path)
    
    drop_cols = [
        #'STOK1',
        #'RSI',
        #'ATR2',
        'ATR21',
        #'ATR3',
        'ATR31', 
        'ATR32',
        'ATR34',   
        #'ROC',     
        #'SDKC9',   
        #'SDKC91',  
        #'SDBB91',  
        #'SDLR310'
    ]


    data = data.drop(columns=['outputC'])
    #data = data.drop(columns=drop_cols)
    feature_columns = list(data.columns[:-1])
    
    target_column = 'output'  # Replace with your actual target column name
    input_chunk_length = 60
    output_chunk_length = 3
    n_epochs = 1000
    num_stacks = 4
    num_blocks = 3
    num_layers = 5
    layer_widths = 512
    test_split = 0.85


    #(data, feature_columns, target_column, split):
    X_train, X_test, y_train, y_test, scaler = process_data(data, feature_columns, target_column, test_split)

    print("Training model...")
    # (train_target, val_target, train_covariates, val_covariates, 
    model = train_and_save_model(y_train, y_test, X_train, X_test,
        input_chunk_length, output_chunk_length, 
            n_epochs, num_stacks, num_blocks, num_layers, layer_widths, 
            patience_val=5, min_delta_val=0.005)
    
    model2 = train_and_save_model(y_train, y_test, X_train, X_test,
        input_chunk_length, output_chunk_length+1, 
            n_epochs, num_stacks+1, num_blocks+1, num_layers+1, layer_widths, 
            patience_val=5, min_delta_val=0.005)
    

    model3 = train_and_save_model(y_train, y_test, X_train, X_test,
    input_chunk_length, output_chunk_length+2, 
        n_epochs, num_stacks+2, num_blocks+2, num_layers+2, layer_widths,
        patience_val=5, min_delta_val=0.005)


    # Make predictions
    print("Making predictions...")
    predictions = model.predict(output_chunk_length-1, series=y_test, past_covariates=X_test)
    
    print("Making predictions...")
    predictions2 = model2.predict(output_chunk_length-1, series=y_test, past_covariates=X_test)
    
    print("Making predictions...")
    predictions3 = model3.predict(output_chunk_length-1, series=y_test, past_covariates=X_test)
    
    
    #print("Generating statistics...")
    #generate_statistics(y_test, predictions)
    
    # Inverse transform the predictions and actual values
    #actual_values = scaler.inverse_transform(y_test).values()
    #predicted_values = scaler.inverse_transform(predictions).values()
    
    print("Plotting results...")
    #plot_results(y_test, predictions, 250)
    plot_multi_results(y_test, predictions, predictions2, predictions3, 100)

if __name__ == "__main__":
    main()
