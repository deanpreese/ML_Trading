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

    #print("X_oos shape: ", X_oos.all_values().shape)
    #print("y_oos shape: ", y_oos.all_values().shape)        
    
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
    
    #print("X_train shape: ", X_train.all_values().shape)
    #print("X_test shape: ", X_test.all_values().shape)
    #print("y_train shape: ", y_train.all_values().shape)
    #print("y_test shape: ", y_test.all_values().shape)
    
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


def eval_model(use_display, test_series, output_chunk, model, past_covariates=None, future_covariates=None):
    
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
    
    if use_display:
        
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

    print(" ")
    print(f"RMSE: {e_rmse}")
    print(f"Total {total}  Correct {correct}  Percent {perf}")
    print(" ")

    return e_rmse, total, correct, perf


def main():
    
    datafile = [ 
                'data/Lucky13_3070_oos.csv',   
                'data/Lucky13_3070.csv',  #1
                'data/ndata_diff_lucky13_3070_oos.csv', 
                'data/ndata_diff_lucky13_3070.csv', #3
                'data/ndata_lucky_13_lag_3070_oos.csv', 
                'data/ndata_lucky13_lag_3070.csv', #5
                'new_model_Z_lucky13_3070_oos.csv',
                'new_model_Z_lucky13_3070.csv' #7,
                'data/Lucky13_3070_oos_3.csv',   
                'data/Lucky13_3070_3.csv',  #8
                'data/Lucky13_3070_oos_5.csv',   
                'data/Lucky13_3070_5.csv',  #10
                
        ]

    data = pd.read_csv(datafile[1])
        
    X = data
    X = X.drop(columns=['output', 'outputC'])
    feature_columns = list(X.columns)
    target_column = 'outputC'  # Replace with your actual target column name
    base_input_chunk_length = 5
    base_output_chunk_length = 1
    n_epochs = 100
    num_stacks = 3
    num_blocks = 2
    num_layers = 4
    layer_widths = 512
    test_split = 0.80


    #(data, feature_columns, target_column, split):
    X_train, X_test, y_train, y_test, scaler = process_train_test_data(data, feature_columns, target_column, test_split)
    
    print("Training models...")
    
    model_results = []
    
    out_chunk_list = [1]
    in_chunk_list = [5,7,9,11,13,15,21, 23,25]
    
    for o in range(len(out_chunk_list)):
        for i in range(len(in_chunk_list)):
            
            input_chunk_length = in_chunk_list[i]
            output_chunk_length = out_chunk_list[o]

            print(" ")
            print(f"Runnng Models with Input Chunk: {input_chunk_length}  Output Chunk: {output_chunk_length}")
            print(" ")

            #model_beats = build_NBeats(input_chunk_length, output_chunk_length, 
            #        n_epochs, num_stacks, num_blocks, num_layers, layer_widths, 
            #        patience_val=10, min_delta_val=0.005)
            
            model_hits = build_NHits(input_chunk_length, output_chunk_length, 
                    n_epochs, num_stacks, num_blocks, num_layers, layer_widths, 
                    patience_val=10, min_delta_val=0.005)    
                            
            
            #model_beats.fit(series=y_train)
            #e_rmse_b, total_b, correct_b, perf_b = eval_model(False, y_test, output_chunk_length, model_beats)


            model_hits.fit(series=y_train)
            e_rmse_h, total_h, correct_h, perf_h = eval_model(False, y_test, output_chunk_length, model_hits)
        
            output = [input_chunk_length, output_chunk_length, 0,0,0,0, e_rmse_h, total_h, correct_h, perf_h]
            #output = [input_chunk_length, output_chunk_length, e_rmse_b, total_b, correct_b, perf_b, 0,0,0,0]

            model_results.append(output)
            
    e_perf = pd.DataFrame(model_results)        
    e_perf.columns = ["In Chunk", "Out Chunk", "RMSE B", "Total B", "Correct B", "Perf B", "RMSE H", "Total H", "Correct H", "Perf H"]        
    e_perf.to_csv('NBeats_model_results.csv', index=False)

    print(e_perf)
    print(" ")

if __name__ == "__main__":
    main()
