
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NHiTSModel, NBEATSModel
from darts.models import LightGBMModel, XGBModel, CatBoostModel
from torchmetrics import MetricCollection
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from darts.utils.likelihood_models import QuantileRegression, LaplaceLikelihood, DirichletLikelihood, ContinuousBernoulliLikelihood
from darts.metrics import mae, mape, rmse, coefficient_of_variation, dtw_metric
from torchmetrics.regression import SpearmanCorrCoef, PearsonCorrCoef, R2Score, MeanAbsoluteError 
from torchmetrics.regression import MeanSquaredError, PearsonCorrCoef, MeanAbsolutePercentageError, CosineSimilarity

from scipy.stats import mode

import joblib

from sklearn.metrics import confusion_matrix, precision_score, recall_score



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
    
    #print("X_train shape: ", X_train.all_values().shape)
    #print("X_test shape: ", X_test.all_values().shape)
    #print("y_train shape: ", y_train.all_values().shape)
    #print("y_test shape: ", y_test.all_values().shape)
    
    return X_train, X_test, y_train, y_test, scaler



def calc_c(predictions, actuals, model_name):
    correct = 0 
    total = 0
   
    forecast_results = predictions
    test_series = actuals
    
    for i in range(len(forecast_results)):
        predicted_output = forecast_results[i].values()[0][0]
        target_output = test_series[i].values()[0][0]

        if ( target_output > 0 and predicted_output > 0.5):
            correct += 1 

        if ( target_output == 0 and predicted_output < 0.5):
            correct += 1      

        total +=  1    

    perf = round((correct)/total,4)
    print(f"Classification Results:  {model_name} Total {total}  Correct {correct}  Percent {perf}")
    
    return total, correct, perf
    

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

def build_cat(input_sequence_len, output_chunk_length, cov_lags=None):
    model = CatBoostModel(
        lags=input_sequence_len,
        lags_past_covariates=cov_lags,
        #likelihood='poisson',
        output_chunk_length=output_chunk_length
    )
    return model


def build_xgb(input_sequence_len, output_chunk_length, cov_lags=None):
    model = XGBModel(
        lags=input_sequence_len,
        lags_past_covariates=cov_lags,
        #likelihood='poisson',
        output_chunk_length=output_chunk_length
    )
    return model


def build_lightGBM(input_sequence_len, output_chunk_length, cov_lags=None):
    model = LightGBMModel(
        lags=input_sequence_len,
        lags_past_covariates=cov_lags,
        #likelihood='poisson',
        output_chunk_length=output_chunk_length,
        verbose=-1
    )
    return model


def ensemble_model(all_predict_data, actuals, forecast_len):
    
    print("Ensemble model...")
    actuals = actuals[-forecast_len:]

    final_preds = []
    final_actuals = []

    correct_ave = 0
    correct_maj = 0
    correct_ccc = 0
    correct_xxx = 0
    total = 0

    for y in range(len(actuals)):
        
        target_output = actuals[y].values()[0][0]
        comb_pred = 0
        w_comb_pred =0
        
        final_actuals.append(target_output)
        
        total += 1
        wt_p = 0
                
        for v in range(len(all_predict_data)):
            
            predict = all_predict_data[v]            
            pre_val = predict[y].values()[0][0]
            comb_pred += pre_val
            w_comb_pred += pre_val*.52    

        ave_p = comb_pred/len(all_predict_data)
        wt_p = w_comb_pred/len(all_predict_data)

        algo_pred = 0

        if ( target_output > 0 and (ave_p > 0.5  and wt_p > 0.5)):
            correct_xxx += 1
        
        if ( target_output == 0 and (ave_p < 0.5 and wt_p < 0.5)):
            correct_xxx += 1

        if ( target_output > 0 and (ave_p > 0.5  or wt_p > 0.5)):
            correct_ccc += 1
        
        if ( target_output == 0 and (ave_p < 0.5 or wt_p < 0.5)):
            correct_ccc += 1

        if ( target_output > 0 and ave_p > 0.5):
            correct_ave += 1
             
        if ( target_output == 0 and ave_p < 0.5):
            correct_ave += 1
        
        if ( target_output > 0 and wt_p > 0.5):
            correct_maj += 1

        if ( target_output == 0 and wt_p < 0.5):
            correct_maj += 1
                  
            
    perf_ave = round((correct_ave)/total,4)
    perf_maj = round((correct_maj)/total,4)
    perf_cc = round((correct_ccc)/total,4)
    perf_xxx = round((correct_xxx)/total,4)
        
    print(f"Ensemble Perf Ave: {perf_ave}  Perf Maj: {perf_maj}  CCC {perf_cc} XXX {perf_xxx}  Total: {total}")
        
    return perf_ave, perf_maj, perf_cc, total
        
        
        
def main(input_chunk_length, output_chunk_length):
    
    #file_path = 'data/buildSeqInd_Lucky13_F.csv'
    #data = pd.read_csv(file_path)
    #list80 = ['SDKC9', 'ATR3', 'STOK1', 'SDKC91', 'ATR21', 'output']
    
    file_path = 'data/Ind_F.csv'
    data = pd.read_csv(file_path)
    
    list80 = ['CCI20', 'ATR2', 'CCI9', 'RSI3', 'RSI9', 'VOSC7', 'STOK15657', 'STOD15657', 'ADX20', 'STOD7217', 'STOK7217', 'SDKC9', 'RSI14', 'VOSC9', 'SDKC14', 'ADX14', 'outputC']
    list60 = ['CCI20', 'ATR2', 'CCI9', 'RSI3', 'RSI9', 'VOSC7', 'STOK15657', 'STOD15657', 'ADX20', 'STOD7217', 'STOK7217', 'SDKC9', 'outputC']
    
    listX = [
            'CCI20', 
             'ATR2', 
             'CCI9', 
             'RSI3', ##
             'RSI9',  ##
             'VOSC7',    #
             'STOK15657', #
             'STOD15657',  #
              # 'ADX20', 
              # 'STOD7217', 
              # 'STOK7217', 
              # 'SDKC9', 
              # 'RSI14',
              # 'VOSC9', 
              # 'SDKC14', 
              # 'ADX14', 
              'LR1033',
              'ROC9',
              'SDKC9',
              'BB14',
              #'LR813',
              #'BB9',
              #'ROC7',
              #'SDKC14',
              #'LR310',
              'outputC']
    
    data = data[listX]
    
    #data = data.drop(columns=['outputC'])
    #data = data.drop(columns=['output'])
        
    feature_columns = list(data.columns[:-1])
    
    target_column = 'outputC'  
    #input_chunk_length = 9
    #output_chunk_length = 1
    test_split = 0.80


    #(data, feature_columns, target_column, split):
    X_train, X_test, y_train, y_test, scaler = process_train_test_data(data, feature_columns, target_column, test_split)
    
    print("Training model...")
 
    cov_lags_in = None
    cov_lags_in = input_chunk_length
    past_train_covariates_in = None
    past_train_covariates_in = X_train 
    past_test_covariates_in = None
    past_test_covariates_in = X_test 

    model_cat = build_cat(input_chunk_length, output_chunk_length, cov_lags=cov_lags_in)
    model_cat.fit(y_train, past_covariates=past_train_covariates_in)
    model_xgb = build_xgb(input_chunk_length, output_chunk_length, cov_lags=cov_lags_in)
    model_xgb.fit(y_train, past_covariates=past_train_covariates_in)
    model_lgb = build_lightGBM(input_chunk_length, output_chunk_length, cov_lags=cov_lags_in)
    model_lgb.fit(y_train, past_covariates=past_train_covariates_in)
    
    forecast_results_cat = gen_forecast(y_test, output_chunk_length, model_cat, past_covariates=past_test_covariates_in, future_covariates=None)
    forecast_results_xgb = gen_forecast(y_test, output_chunk_length, model_xgb, past_covariates=past_test_covariates_in, future_covariates=None)
    forecast_results_lgb = gen_forecast(y_test, output_chunk_length, model_lgb, past_covariates=past_test_covariates_in, future_covariates=None)
    

    model_cat2 = build_cat(input_chunk_length, output_chunk_length, cov_lags=None)
    model_cat2.fit(y_train, past_covariates=None)
    model_xgb2 = build_xgb(input_chunk_length, output_chunk_length, cov_lags=None)
    model_xgb2.fit(y_train, past_covariates=None)
    model_lgb2 = build_lightGBM(input_chunk_length, output_chunk_length, cov_lags=None)
    model_lgb2.fit(y_train, past_covariates=None)
    
    forecast_results_cat2 = gen_forecast(y_test, output_chunk_length, model_cat2, past_covariates=None, future_covariates=None)
    forecast_results_xgb2 = gen_forecast(y_test, output_chunk_length, model_xgb2, past_covariates=None, future_covariates=None)
    forecast_results_lgb2 = gen_forecast(y_test, output_chunk_length, model_lgb2, past_covariates=None, future_covariates=None)
    
    forecast_results_len = len(forecast_results_xgb)
    predict_data1 = [forecast_results_lgb, forecast_results_xgb, forecast_results_cat]
    predict_data2 = [forecast_results_xgb2, forecast_results_xgb ] #***
    predict_data3 = [forecast_results_xgb, forecast_results_xgb2, forecast_results_cat ]
    predict_data4 = [forecast_results_xgb, forecast_results_xgb2, forecast_results_cat2 ] #***

    print(" ")
    perf_ave, perf_maj, perf_cc, total = ensemble_model(predict_data1, y_test, forecast_results_len)
    perf_ave, perf_maj, perf_cc, total = ensemble_model(predict_data2, y_test, forecast_results_len)
    perf_ave, perf_maj, perf_cc, total = ensemble_model(predict_data3, y_test, forecast_results_len)
    perf_ave, perf_maj, perf_cc, total = ensemble_model(predict_data4, y_test, forecast_results_len)
     
    #forecast_results_len = len(forecast_results_xgb)
    #predict_data = [forecast_results_lgb, forecast_results_xgb,forecast_results_cat]
    #perf_ave, perf_maj, perf_cc, total = ensemble_model(predict_data, y_test, forecast_results_len)
    
    #plot_model(y_test, output_chunk_length, model_hits, past_covariates=None, future_covariates=None)
    
    #e_rmse_xgb = rmse(y_test, forecast_results_xgb)
    #e_rmse_lgb = rmse(y_test, forecast_results_lgb)
    #e_rmse_cat = rmse(y_test, forecast_results_cat)
    
    print(" ")
    #print(f"RMSE LGB: {e_rmse_lgb}")
    #print(f"RMSE XGB: {e_rmse_xgb}")
    #print(f"RMSE CAT: {e_rmse_cat}")
    
    """
    total_rc, correct_rc, perf_rc = calc_c(forecast_results_xgb, y_test, 'XGB')
    total_lc, correct_lc, perf_lc = calc_c(forecast_results_lgb, y_test, 'LGB')
    total_ccat, correct_ccat, perf_ccat = calc_c(forecast_results_cat, y_test, 'CAT')
    
    total_rc, correct_rc, perf_rc = calc_c(forecast_results_xgb2, y_test, 'XGB2')
    total_lc, correct_lc, perf_lc = calc_c(forecast_results_lgb2, y_test, 'LGB2')
    total_ccat, correct_ccat, perf_ccat = calc_c(forecast_results_cat2, y_test, 'CAT2')
    """
    
    
    
    #return input_chunk_length, output_chunk_length, e_rmse_lgb, e_rmse_xgb,e_rmse_cat,0, perf_rc, 0, perf_lc, 0, perf_ccat
    
    
if __name__ == "__main__":
    
    output_list = []
    num_predicts = 1
    
    #for o in range(1,3,1):
    #    for i in range(5,9,1):
    #        rslts = main(i, num_predicts+o)
    #        output_list.append(rslts)

    rslts = main(21, 1)
    
    # 15  1 
    #   2  Ensemble Perf Ave: 0.5161  Perf Maj: 0.5221  CCC 0.6442   Total: 43620
    #   4  Ensemble Perf Ave: 0.5155  Perf Maj: 0.5221  CCC 0.6499   Total: 43620
    
    #   2  Ensemble Perf Ave: 0.5145  Perf Maj: 0.522  CCC 0.6483   Total: 43614
    #   5  Ensemble Perf Ave: 0.5102  Perf Maj: 0.522  CCC 0.6514   Total: 43614
    
    #output_list.append(rslts)
    #out_df = pd.DataFrame(output_list, columns=['input_chunk_length', 'output_chunk_length', 'e_rmse_lgb', 'e_rmse_xgb', 'e_rmse_cat','perf_rx', 'perf_rc', 'perf_lr', 'perf_lc', 'perf_rcat', 'perf_ccat'])
    #out_df.to_csv('chunk_data__out.csv')
    #print(out_df)