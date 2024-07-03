import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NHiTSModel, NBEATSModel
from torchmetrics import MetricCollection
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from darts.utils.likelihood_models import QuantileRegression, LaplaceLikelihood, DirichletLikelihood, ContinuousBernoulliLikelihood
from darts.metrics import mae, mape, rmse, coefficient_of_variation, dtw_metric
from torchmetrics.regression import SpearmanCorrCoef, PearsonCorrCoef, R2Score, MeanAbsoluteError 
from torchmetrics.regression import MeanSquaredError, PearsonCorrCoef, MeanAbsolutePercentageError, CosineSimilarity
import joblib

def process_oos_data(data, feature_columns, target_column):
    print("Processing data...")
    
    oos = TimeSeries.from_dataframe(data).astype(np.float32)   
    X_oos = oos.drop_columns(target_column)
    y_oos = oos.drop_columns(feature_columns)
    
    scaler = Scaler()
    X_oos = scaler.fit_transform(X_oos) 
    
    return X_oos, y_oos, scaler


def process_train_test_data(data, feature_columns, target_column, split):
    print("Processing data...")
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


def eval_model(plot_tf, test_series, output_chunk, model, past_covariates=None, future_covariates=None):
    
    forecast_results = model.historical_forecasts(series=test_series, 
                                          past_covariates=past_covariates,
                                          future_covariates=future_covariates,
                                          start=0.8, 
                                          retrain=False,
                                          verbose=True, 
                                          predict_likelihood_parameters=False,
                                          forecast_horizon=output_chunk)
    
    e_rmse = rmse(test_series, forecast_results)
    correct = 0 
    total = 0
    
    for i in range(len(forecast_results)):
        
        predicted_output = forecast_results[i].values()[0][0]
        target_output = test_series[i].values()[0][0]

        if ( target_output > 0 and predicted_output > 0):
            correct += 1 

        if ( target_output < 0 and predicted_output < 0):
            correct += 1 
        
        if ( target_output == 0 and predicted_output == 0):
            correct += 1     

        total +=  1    

    perf = round((correct)/total,4)
    
    if plot_tf:
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

    print(f"RMSE: {e_rmse}")
    print(f"Total {total}  Correct {correct}  Percent {perf}")
    print(" ")

    return e_rmse, total, correct, perf


def main():
    
    #file_path = 'data/buildSeqInd_Lucky13_F.csv'
    #oos_data = pd.read_csv(oos_file)
    #oos_file = 'data/oos_Lucky13_F.csv'
    
    file_path = 'data/Ind_F.csv'

    data = pd.read_csv(file_path)
    
    list80 = ['SDKC9', 'ATR3', 'STOK1', 'SDKC91', 'ATR21']
    drop_cols = [
        #'STOK1',
        'RSI',
        #'ATR2',
        'ATR21',
        'ATR3',
        'ATR31', 
        'ATR32',
        'ATR33',   
        'ROC',     
        'SDKC9',   
        'SDKC91',  
        'SDBB91',  
        'SDLR310'
    ]

    #oos_data = data.drop(columns=['outputC'])
    data = data.drop(columns=['outputC'])
    
    oos_data = data.drop(columns=drop_cols)
    data = data.drop(columns=drop_cols)
    
    feature_columns = list(data.columns[:-1])
    
    target_column = 'output'  # Replace with your actual target column name
    input_chunk_length = 3
    output_chunk_length = 1
    n_epochs = 100
    num_stacks = 3
    num_blocks = 2
    num_layers = 4
    layer_widths = 512
    test_split = 0.80


    #(data, feature_columns, target_column, split):
    X_train, X_test, y_train, y_test, scaler = process_train_test_data(data, feature_columns, target_column, test_split)
    X_oos, y_oos, ooos_scaler =  process_oos_data(oos_data, feature_columns, target_column)
    
    
    #print("X_train shape: ", X_train.all_values().shape)
    #print("X_test shape: ", X_test.all_values().shape)
    #print("y_train shape: ", y_train.all_values().shape)
    #print("y_test shape: ", y_test.all_values().shape)
    
    
    print("Training model...")
    #model_beats = build_NBeats(input_chunk_length, output_chunk_length, 
    #        n_epochs, num_stacks, num_blocks, num_layers, layer_widths, 
    #        patience_val=10, min_delta_val=0.005)
    
    model_hits = build_NHits(input_chunk_length, output_chunk_length, 
            n_epochs, num_stacks, num_blocks, num_layers, layer_widths, 
            patience_val=10, min_delta_val=0.005)

    model_hits.fit(series=y_train)
    eval_model(False, y_test, output_chunk_length, model_hits)
    
    #y_train, y_test, X_train, X_test,
    #model_hits.fit(series=y_train, val_series=y_test, 
    #          past_covariates=X_train, val_past_covariates=X_test)
    
    #eval_model(y_test, model_hits, X_test)
    
    
    #print("Making predictions...")
    #predictions_hits = model_hits.predict(output_chunk_length-1, series=y_test, past_covariates=X_test, predict_likelihood_parameters=True)
    
    #actual_values = scaler.inverse_transform(y_test).values()
    #predicted_values = scaler.inverse_transform(predictions).values()
    
    #print("Plotting results...")
    #plot_results(y_test, predictions_hits, 250)
    #plot_results_m(y_test, predictions_hits, predictions_hits2, 100)

if __name__ == "__main__":
    main()
