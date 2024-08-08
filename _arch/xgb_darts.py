import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import XGBModel
from darts.dataprocessing.transformers import Scaler
from darts.models import NHiTSModel, NBEATSModel
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError
from darts.metrics import rmse
from darts.utils.likelihood_models import QuantileRegression
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
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test.astype(np.float32))
    
    print("X_train shape: ", X_train.all_values().shape)
    print("X_test shape: ", X_test.all_values().shape)
    print("y_train shape: ", y_train.all_values().shape)
    print("y_test shape: ", y_test.all_values().shape)
    
    return X_train, X_test, y_train, y_test, scaler


def calc_c(predictions, actuals):
    correct = 0 
    total = 0
    e_rmse = rmse(actuals, predictions)
    
    for i in range(len(predictions)):
        predicted_output = predictions[i].values()[0][0]
        target_output = actuals[i].values()[0][0]
        
        if (target_output > 0 and predicted_output > 0) or (target_output < 0 and predicted_output < 0) or (target_output == 0 and predicted_output == 0):
            correct += 1     
        total += 1    

    perf = round(correct / total, 4)
    print(f"RMSE: {e_rmse}")
    print(f"Total {total}  Correct {correct}  Percent {perf}")
    print(" ")
    
    return total, correct, perf, e_rmse


def calc_r(predictions, actuals):
    correct = 0 
    total = 0
    e_rmse = rmse(actuals, predictions)
    
    for i in range(len(predictions)):
        predicted_output = predictions[i].values()[0][0]
        target_output = actuals[i].values()[0][0]

        if (target_output > 0 and predicted_output > 0.5) or (target_output == 0 and predicted_output < 0.5):
            correct += 1     
        total += 1    

    perf = round(correct / total, 4)
    print(f"RMSE: {e_rmse}")
    print(f"Total {total}  Correct {correct}  Percent {perf}")
    print(" ")
    
    return total, correct, perf, e_rmse


def plot_model(test_series, output_chunk, model, past_covariates=None, future_covariates=None):
    last_x_rows = 100
    forecast_results = model.historical_forecasts(
        series=test_series, 
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        start=0.8,
        retrain=False,
        verbose=True,
        predict_likelihood_parameters=True,
        forecast_horizon=output_chunk
    )
    
    forecast_results = forecast_results[-last_x_rows:]
    test_series = test_series[-last_x_rows:]
    
    plt.figure(figsize=(12, 6))
    test_series.plot(label='actual', color='black')
    forecast_results.plot(low_quantile=0.2, high_quantile=0.8, label="20-80th percentiles", color='green')
    forecast_results.plot(label='backtest (n=10)', color='red')
    plt.show()




def gen_forecast(test_series, output_chunk, model, past_covariates=None, future_covariates=None):
    
    forecast_results = model.historical_forecasts(series=test_series, 
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        #start=0.8, 
        retrain=False,
        verbose=True,
        #last_points_only=True, 
        predict_likelihood_parameters=False,
        forecast_horizon=output_chunk)
    return forecast_results    





def main():
    datafile = [ 
        'data/Lucky13_3070_oos.csv',   
        'data/Lucky13_3070.csv',  #1
        'data/ndata_diff_lucky13_3070_oos.csv', 
        'data/ndata_diff_lucky13_3070.csv', #3
        'data/ndata_lucky_13_lag_3070_oos.csv', 
        'data/ndata_lucky13_lag_3070.csv', #5
        'new_model_Z_lucky13_3070_oos.csv',
        'new_model_Z_lucky13_3070.csv', #7
        'data/Lucky13_3070_oos_3.csv',   
        'data/Lucky13_3070_3.csv',  #9
        'data/Lucky13_3070_oos_5.csv',   
        'data/Lucky13_3070_5.csv',  #11
        
    ]

    data = pd.read_csv(datafile[1])
    data = data.drop(columns=['output'])
    
    feature_columns = list(data.columns[:-1])
    target_column = 'outputC'  # Replace with your actual target column name
    input_chunk_length = 24
    output_chunk_length = 3
    n_epochs = 100
    num_stacks = 4
    num_blocks = 2
    num_layers = 4
    layer_widths = 128
    test_split = 0.80

    X_train, X_test, y_train, y_test, scaler = process_train_test_data(data, feature_columns, target_column, test_split)
    
    model = XGBModel(
    lags=12,
    lags_past_covariates=12,
    #lags_future_covariates=[0,1,2,3,4,5],
    output_chunk_length=6,
)
    
    model.fit(target, past_covariates=past_cov, future_covariates=future_cov)
    pred = model.predict(6)

    model.fit(
        series=y_train, val_series=y_test, 
        past_covariates=X_train, val_past_covariates=X_test
    )
    
    forecast_results = gen_forecast(y_test, output_chunk_length, model, past_covariates=X_test)
    calc_r(forecast_results, y_test)
    calc_c(forecast_results, y_test)


if __name__ == "__main__":
    main()
