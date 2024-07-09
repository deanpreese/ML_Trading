#import csv
#import datetime
from flask import Flask, request
#import numpy as np
import pandas as pd
#from io import StringIO
#import cProfile
from io import BytesIO
#import time as mytime

from strategy.model_loader import ModelLoader

import logging
logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)
logging.getLogger('mlflow.pyfunc').setLevel(logging.ERROR)
logging.getLogger('lightgbm').setLevel(logging.ERROR)

import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore", category=FutureWarning)
#warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module='LightGBM')

model_loader = ModelLoader()
models_one = []
models_two = []
models_three = []
models_four = []


def LoadModels(group_id, experiment_id, num_models):
    return model_loader.load_composite_models( experiment_id, num_models, group_id)
    
    
def get_prediction(data_df, models):
    loaded_prediction = 0
    for m in range(len(models)):
        loaded_prediction = models[m].do_predict(data_df)
        print(f"Model  {loaded_prediction}")

    return loaded_prediction    

def get_v_prediction(data_df, models):
    loaded_prediction = 0
    loaded_percent = 0
    loaded_sum_predicts = 0
    
    for m in range(len(models)):
        percent, predict, sum_predicts = models[m].do_predict_v(data_df)
        print(f"Model V {percent}  {predict}  {sum_predicts}")
        
        loaded_prediction += predict
        loaded_percent += percent
        loaded_sum_predicts += sum_predicts
        
        
    final_ave_pct = loaded_percent / len(models)
    final_ave_predict = loaded_prediction / len(models)
    final_ave_sum_predicts = loaded_sum_predicts / len(models)

    print(f"Final  V    {final_ave_predict}  {final_ave_sum_predicts}  {final_ave_pct} ")
    
    return final_ave_predict


def get_v2_prediction(data_df, models):
    loaded_prediction = 0
    loaded_percent = 0
    loaded_sum_predicts = 0
    
    for m in range(len(models)):
        percent, predict, sum_predicts = models[m].do_predict_v(data_df)
        print(f"Model V {percent}  {predict}  {sum_predicts}")
        
        loaded_prediction += predict
        loaded_percent += percent
        loaded_sum_predicts += sum_predicts
        
    final_ave_pct = loaded_percent / len(models)
    final_ave_predict = loaded_prediction / len(models)
    final_ave_sum_predicts = loaded_sum_predicts / len(models)

    if final_ave_predict > 0 and final_ave_pct < 0.76:
        final_ave_predict = 0

    if final_ave_predict < 0 and final_ave_pct < 0.76:
        final_ave_predict = 0

    print(f"Final  V    {final_ave_predict}  {final_ave_sum_predicts}  {final_ave_pct} ")
    
    return final_ave_predict

    

def init_app():
    app = Flask(__name__)

    with app.app_context():
       
       #models_one = LoadModels(0, ["12"], 1)
       #models_two = LoadModels(0, ["12"], 10)
       #models_three = LoadModels(0, ["23"], 1)
       #models_four = LoadModels(0, ["23"], 6)
       
       models_one = LoadModels(0, ["25"], 1)
       models_two = LoadModels(0, ["25"], 7)
       models_three = LoadModels(0, ["25"], 9)
       models_four = LoadModels(0, ["25"], 11)
       
       
              
    @app.route('/predict-one', methods=['POST'])
    def predict_one():
        
        csv_data = BytesIO(request.data)
        #column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR34', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']
        column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR33', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']

        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)

        loaded_prediction = 0

        loaded_prediction = get_prediction(data_df, models_one)
        print(f"predict 1 Model  {loaded_prediction}")
        return str(loaded_prediction)

    @app.route('/predict-two', methods=['POST'])
    def predict_two():
        
        csv_data = BytesIO(request.data)
        #column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR34', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']
        column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR33', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']
        
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)
        
        final_predict = get_v2_prediction(data_df, models_two)
        
        loaded_prediction = final_predict
        print(f"Predict 2 Model  {loaded_prediction}")            
        return str(loaded_prediction)
        
        
    @app.route('/predict-three', methods=['POST'])
    def predict_three():
        
        csv_data = BytesIO(request.data)
        #column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR34', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']
        column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR33', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']        
        
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)

        final_predict = get_v2_prediction(data_df, models_three)
        
        loaded_prediction = final_predict
        print(f"Predict 3 Model  {loaded_prediction}")            
        return str(loaded_prediction)
    
    @app.route('/predict-four', methods=['POST'])
    def predict_four():
        
        csv_data = BytesIO(request.data)
        #column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR34', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']
        column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR33', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']        
        
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)

        loaded_prediction = 0
        loaded_prediction = get_v2_prediction(data_df, models_four)
        print(f"Predict 4 Model  {loaded_prediction}")            
        return str(loaded_prediction)    
            
        
        
    return app

   

app = init_app()
    
if __name__ == '__main__':
    print("Starting Flask application.")
      
    
    app.run(
        debug=True, 
        use_reloader=False,
        port=9999, 
        host='0.0.0.0'
        )
    