
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NHiTSModel, NBEATSModel
from torchmetrics import MetricCollection
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from darts.utils.likelihood_models import QuantileRegression, LaplaceLikelihood, DirichletLikelihood, ContinuousBernoulliLikelihood
from darts.metrics import mae, mape, rmse, coefficient_of_variation, dtw_metric
from torchmetrics.regression import SpearmanCorrCoef, PearsonCorrCoef, R2Score, MeanAbsoluteError 
from torchmetrics.regression import MeanSquaredError, PearsonCorrCoef, MeanAbsolutePercentageError, CosineSimilarity
import joblib


def process_train_test_data(data, feature_columns, target_column, split):
    print("Processing data...")
    series = TimeSeries.from_dataframe(data).astype(np.float32)   
    train, test = series.split_after(split)
    X_train = train.drop_columns(target_column)
    X_test = test.drop_columns(target_column)
    y_train = train.drop_columns(feature_columns)
    y_test = test.drop_columns(feature_columns)
    
    scaler = Scaler()
    #X_train = scaler.fit_transform(X_train) 
    #X_test = scaler.transform(X_test.astype(np.float32))   
    
    print("X_train shape: ", X_train.all_values().shape)
    print("X_test shape: ", X_test.all_values().shape)
    print("y_train shape: ", y_train.all_values().shape)
    print("y_test shape: ", y_test.all_values().shape)
    
    return X_train, X_test, y_train, y_test, scaler


def build_NBeats(input_chunk_length, output_chunk_length, 
        n_epochs, num_stacks, num_blocks, num_layers, layer_widths, 
            patience_val, min_delta_val ):
        
    # TensorBoard logger
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Early stopping callback
    early_stopper = EarlyStopping(
        monitor="train_loss",
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
    model = NBEATSModel(
        input_chunk_length=input_chunk_length, 
        output_chunk_length=output_chunk_length, 
        n_epochs=n_epochs,
        batch_size=64, 
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
    return model

def build_NHits(input_chunk_length, output_chunk_length, 
        n_epochs, num_stacks, num_blocks, num_layers, layer_widths, 
            patience_val, min_delta_val ):
        
    # TensorBoard logger
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Early stopping callback
    early_stopper = EarlyStopping(
        monitor="train_loss",
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
        batch_size=64, 
        random_state=42, 
        num_stacks=num_stacks, 
        num_blocks=num_blocks, 
        num_layers=num_layers, 
        layer_widths=layer_widths,
        pl_trainer_kwargs=pl_trainer_kwargs,
        likelihood=QuantileRegression([0.2, 0.5, 0.8]),
        #likelihood=DirichletLikelihood(),
        #likelihood=LaplaceLikelihood(prior_b=0.1),
        #likelihood=ContinuousBernoulliLikelihood(),
        torch_metrics=metric_collection,
        log_tensorboard=True
    )
    return model


def gen_forecast(test_series, output_chunk, model, past_covariates=None, future_covariates=None):
    
    forecast_results = model.historical_forecasts(series=test_series, 
                                          past_covariates=past_covariates,
                                          future_covariates=future_covariates,
                                          start=0.8, 
                                          retrain=False,
                                          verbose=True, 
                                          predict_likelihood_parameters=False,
                                          forecast_horizon=output_chunk)
    return forecast_results    


def calc_c(predictions, actuals):
    correct = 0 
    total = 0
    
    forecast_results = predictions
    test_series = actuals
    e_rmse = rmse(test_series, forecast_results)
    
    for i in range(len(forecast_results)):
        
        predicted_output = forecast_results[i].values()[0][0]
        target_output = test_series[i].values()[0][0]

        #print(f"Predicted: {predicted_output}  Target: {target_output}")

        if ( target_output > 0 and predicted_output > 0):
            correct += 1 

        if ( target_output < 0 and predicted_output < 0):
            correct += 1 
        
        if ( target_output == 0 and predicted_output == 0):
            correct += 1     

        total +=  1    

    perf = round((correct)/total,4)
    print(f"RMSE: {e_rmse}")
    print(f"Total {total}  Correct {correct}  Percent {perf}")
    print(" ")
    
    return total, correct, perf, e_rmse


def calc_r(predictions, actuals):
    correct = 0 
    total = 0
   
    forecast_results = predictions
    test_series = actuals
    
    e_rmse = rmse(test_series, forecast_results)
    
    for i in range(len(forecast_results)):
        
        predicted_output = forecast_results[i].values()[0][0]
        target_output = test_series[i].values()[0][0]

        #print(f"Predicted: {predicted_output}  Target: {target_output}")

        if ( target_output > 0 and predicted_output > 0.5):
            correct += 1 

        if ( target_output == 0 and predicted_output < 0.5):
            correct += 1 
        
        #if ( target_output == 0 and predicted_output == 0):
        #    correct += 1     

        total +=  1    

    perf = round((correct)/total,4)
    
    print(f"RMSE: {e_rmse}")
    print(f"Total {total}  Correct {correct}  Percent {perf}")
    print(" ")
    
    return total, correct, perf, e_rmse
    

def plot_model(test_series, output_chunk, model, past_covariates=None, future_covariates=None):
    
    last_x_rows = 100
    like_results = model.historical_forecasts(series=test_series, 
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        start=0.8, 
        retrain=False,
        verbose=True, 
        predict_likelihood_parameters=True,
        forecast_horizon=output_chunk)        

    forecast_results = forecast_results[-last_x_rows:]
    test_series = test_series[-last_x_rows:]
    like_results = like_results[-last_x_rows:]   
    
    plt.figure(figsize=(12, 6))
    test_series.plot(label='actual', color='black')
    like_results.plot(low_quantile=0.2, high_quantile=0.8, label="20-80th percentiles", color='green')
    forecast_results.plot(label='backtest (n=10)', color='red')
    plt.show()


def main():
    
    datafile = [ 
            'data/NewModel_3070_oos.csv',   
            'data/NewModel_3070.csv',  #1
            'data/NewModel_ALL_oos.csv',   
            'data/NewModel_ALL.csv',  #3
            'data/NewModel_span3_3070_oos.csv',   
            'data/NewModel_span3_3070.csv',  #5        
    ]   

    data = pd.read_csv(datafile[1])
    feature_columns = list(data.columns[:-1])
    
    target_column = 'outputC'  # Replace with your actual target column name
    input_chunk_length = 7
    output_chunk_length = 1
    n_epochs = 100
    num_stacks = 3
    num_blocks = 2
    num_layers = 4
    layer_widths = 512
    test_split = 0.80


    #(data, feature_columns, target_column, split):
    X_train, X_test, y_train, y_test, scaler = process_train_test_data(data, feature_columns, target_column, test_split)
    
    print("Training model...")
    #model = build_NBeats(input_chunk_length, output_chunk_length, 
    #        n_epochs, num_stacks, num_blocks, num_layers, layer_widths, 
    #        patience_val=10, min_delta_val=0.005)
    
    model = build_NHits(input_chunk_length, output_chunk_length, 
            n_epochs, num_stacks, num_blocks, num_layers, layer_widths, 
            patience_val=5, min_delta_val=0.005)

    model.fit(series=y_train, val_series=y_test, 
              past_covariates=X_train, val_past_covariates=X_test)
    
    forecast_results = gen_forecast(y_test, output_chunk_length, model, past_covariates=X_test, future_covariates=None)
    
    #plot_model(y_test, output_chunk_length, model_hits, past_covariates=None, future_covariates=None)
    calc_r(forecast_results, y_test)
    calc_c(forecast_results, y_test)

    
if __name__ == "__main__":
    main()
