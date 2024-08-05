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
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module='[LightGBM]')

model_loader = ModelLoader()
models_one = []
models_two = []
models_three = []
models_four = []


def LoadModels(group_id, experiment_id, num_models):
    return model_loader.load_composite_strategy( experiment_id, num_models, group_id)    


def get_base_predictions(data_df, models):
    
    all_agg_predicts = 0
    all_agg_weighted = 0
    all_predicts = []
    
    for m in range(len(models)):
        agg_predict, agg_weighted_predict, predicts = models[m].do_predict_base(data_df)
        
        all_agg_predicts += agg_predict
        all_agg_weighted += agg_weighted_predict
        all_predicts.append(predicts)
        
    all_agg_predicts = all_agg_predicts/len(all_agg_predicts)
    all_agg_weighted = all_agg_weighted/len(all_predicts)        
        
    return all_agg_predicts, all_agg_weighted, all_predicts




def init_app():
    app = Flask(__name__)

    with app.app_context():

       models_one = LoadModels(0, ["68"], 2)
       models_two = LoadModels(0, ["68"], 2)
       models_three = LoadModels(0, ["66"], 2)
       models_four = LoadModels(0, ["66"], 2)

    # ------------------------------------------
    # Baseline Aggregate Prediction          
    # ------------------------------------------
    @app.route('/predict-one', methods=['POST'])
    def predict_one():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR34', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)

        agg_prediction, agg_weighted_prediction, all_predicts = get_base_predictions(data_df, models_one)
        
        print(f"Aggregate Prediction Model  {agg_prediction}")
        return str(agg_prediction)


    # ------------------------------------------
    # Weighted Aggregate Prediction          
    # ------------------------------------------
    @app.route('/predict-two', methods=['POST'])
    def predict_two():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR34', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']
        
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)
        
        agg_prediction, agg_weighted_prediction, all_predicts = get_base_predictions(data_df, models_one)
        
        print(f"Aggregate Weighted Prediction Model  {agg_weighted_prediction}")
        return str(agg_weighted_prediction)


        
    # ------------------------------------------
    # Comp Weighted Prediction          
    # ------------------------------------------        
    @app.route('/predict-three', methods=['POST'])
    def predict_three():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR33', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']        

        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)
        agg_prediction, agg_weighted_prediction, all_predicts = get_base_predictions(data_df, models_one)
        comp_predict = ((0.46 * agg_prediction) + (0.54 * agg_weighted_prediction)  )
        
        print(f"Composite Weighted Prediction Model  {comp_predict}")
        return str(comp_predict)

    
    
    
    # ------------------------------------------
    # Threshold Filter Prediction          
    # ------------------------------------------    
    @app.route('/predict-four', methods=['POST'])
    def predict_four():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR34', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']        
        
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)

        agg_prediction, agg_weighted_prediction, all_predicts = get_base_predictions(data_df, models_one)
        
        final_predict = 0
        
        if agg_prediction > 0.5 or agg_weighted_prediction > 0.5:
            final_predict = agg_prediction
        
        if agg_prediction < -0.5 or agg_weighted_prediction < 0.5:
            final_predict = agg_prediction
            
        
        print(f"Aggregate Weighted Prediction Model  {agg_weighted_prediction}")
        return str(agg_weighted_prediction)
   
            
        
        
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
    