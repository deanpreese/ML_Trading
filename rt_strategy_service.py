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


def get_agg_prediction(data_df, models):
    loaded_prediction = 0
    for m in range(len(models)):
        p2 = models[m].do_predict_v(data_df)
        print(f"Model V {p2}")
        loaded_prediction += p2 
    
    final_predict = loaded_prediction/(len(models)+1)
    print(f"Final   {final_predict}")

    return final_predict


def get_v2_prediction(data_df, models):
    loaded_prediction = 0
    loaded_percent = 0
    rtn_predict = 0
    
    pre_up = 0
    pre_down = 0
    pct_up = 0
    pct_down = 0
    
    for m in range(len(models)):
        percent, predict = models[m].do_predict_v(data_df)
        print(f"Model V {percent}  {predict} ")
        loaded_prediction += predict
        loaded_percent += percent
        
        if ( predict > 0):
            pre_up += 1
    
        if ( predict < 0):
            pre_down += 1   
    
        if percent > 0.5:
            pct_up += 1
            
        if percent < 0.5:
            pct_down += 1            

    if pct_up > pct_down and pre_up > pre_down:
        rtn_predict = 1.0
        
    if pct_up < pct_down and pre_down > pre_up:
        rtn_predict = -1.0

    print(f"Final   {rtn_predict}")
    
    return rtn_predict

    

def init_app():
    app = Flask(__name__)

    with app.app_context():
        #models_one = LoadModels(0, ["42"], 1)
        #models_two = LoadModels(0, ["40"], 1)
       
       #models_one = LoadModels(0, ["12"], 1)
       #models_two = LoadModels(0, ["12"], 10)
       models_three = LoadModels(0, ["23"], 1)
       models_four = LoadModels(0, ["23"], 6)
              
    @app.route('/predict-one', methods=['POST'])
    def predict_one():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR34', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)

        loaded_prediction = 0

        loaded_prediction = get_prediction(data_df, models_one)
        print(f"predict 1 Model  {loaded_prediction}")
        return str(loaded_prediction)

    @app.route('/predict-two', methods=['POST'])
    def predict_two():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR34', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)

        loaded_prediction = 0
        loaded_prediction = get_v2_prediction(data_df, models_two)
        print(f"Predict 2 Model  {loaded_prediction}")            
        return str(loaded_prediction)
        
        
    @app.route('/predict-three', methods=['POST'])
    def predict_three():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR33', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)

        loaded_prediction = 0
        loaded_prediction = get_prediction(data_df, models_three)
        print(f"Predict 3 Model  {loaded_prediction}")            
        return str(loaded_prediction)
    
    @app.route('/predict-four', methods=['POST'])
    def predict_four():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR33', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)

        loaded_prediction = 0
        loaded_prediction = get_v2_prediction(data_df, models_four)
        print(f"Predict 4 Model  {loaded_prediction}")            
        return str(loaded_prediction)    
            
        
    @app.route('/predict-comp', methods=['POST'])
    def predict_comp():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR33', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)

        p2 = get_agg_prediction(data_df, models_two)            
        p3 = get_agg_prediction(data_df, models_three)            
        p4 = get_agg_prediction(data_df, models_four)            

        predict = (p2 + p3 + p4) / 3
 
        print(f" --->>  Composite predict {predict}")
                   
        return str(predict)        
        
        
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
    