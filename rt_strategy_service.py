#import csv
#import datetime
from flask import Flask, request
#import numpy as np
import pandas as pd
#from io import StringIO
#import cProfile
from io import BytesIO
#import time as mytime

from common.model_loader import ModelLoader

import logging
logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)
logging.getLogger('mlflow.pyfunc').setLevel(logging.ERROR)


model_loader = ModelLoader()
models_one = []
models_two = []



def LoadModels(group_id, experiment_id, num_models):
   
    return model_loader.load_composite_models( experiment_id, num_models, group_id)
    

def init_app():
    app = Flask(__name__)

    with app.app_context():
        models_one = LoadModels(0, ["42"], 1)
        models_two = LoadModels(0, ["40"], 1)
       
       
    @app.route('/predict-one', methods=['POST'])
    def predict_one():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR34', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)

        loaded_prediction = 0
       
        for m in range(len(models_one)):
            loaded_prediction = models_one[m].do_predict(data_df)
            print(f"Model-One {m}    {loaded_prediction}")
            
        return str(loaded_prediction)

    @app.route('/predict-two', methods=['POST'])
    def predict_two():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR34', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)

        loaded_prediction = 0
       
        for m in range(len(models_two)):
            loaded_prediction = models_two[m].do_predict(data_df)
            print(f"Model-Two {m}    {loaded_prediction}")
            
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
    