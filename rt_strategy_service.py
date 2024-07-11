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
    return model_loader.load_composite_models( experiment_id, num_models, group_id)    

# single model prediction -- returns 1 to -1   
def get_prediction(data_df, models):
    loaded_prediction = 0
    for m in range(len(models)):
        loaded_prediction = models[m].do_predict(data_df)
        print(f"Model  {loaded_prediction}")

    return loaded_prediction    



def get_v_prediction(data_df, models):
    
    t_pct_cnt = 0
    t_rtn = 0
    t_agg = 0
    t_agg_w = 0
    t_ens = 0
    
    for m in range(len(models)):
        percentage_positive, return_predict, agg_predict, agg_weighted_predict = models[m].do_predict_v(data_df)
        
        if percentage_positive >= 0.5:
            t_pct_cnt += 1
        
        t_rtn += return_predict
        t_agg += agg_predict   
        t_agg_w += agg_weighted_predict             

    rtn = t_rtn 
    pct = t_pct_cnt/len(models)
    agg = t_agg/len(models) 
    agg_w = t_agg_w/len(models)

    if pct > 0.5 and (  rtn > 0 ) :
        t_ens = rtn    

    if pct < 0.5 and ( rtn < 0 ) :
        t_ens = rtn   

    if pct == 0.5 and ( rtn > 0) :
        t_ens = rtn   

    if pct == 0.5 and ( rtn < 0 ) :
        t_ens = rtn   

    comp_predict = ((0.49 * rtn) + (0.53 * pct) + (0.44 * agg) + (0.42 * agg_w) + (0.46 * t_ens))/5
    
    return rtn, pct, agg, agg_w, t_ens, comp_predict



def init_app():
    app = Flask(__name__)

    with app.app_context():

       models_one = LoadModels(0, ["59"], 1)
       models_two = LoadModels(0, ["59"], 3)
       models_three = LoadModels(0, ["59"], 3)
       models_four = LoadModels(0, ["59"], 5)

              
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
        
        rtn, pct, agg, agg_w, t_ens, comp_predict = get_v_prediction(data_df, models_two)
        
        
        if pct > 0.5: 
            pct = 1
        else:
            pct = -1    
        
        print(f"Predict pct Model  {pct}")            
        return str(pct)
        
        
    @app.route('/predict-three', methods=['POST'])
    def predict_three():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR34', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']        
        
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)

        rtn, pct, agg, agg_w, t_ens, comp_predict = get_v_prediction(data_df, models_two)
           
        print(f"Predict agg_w Model  {agg}")            
        return str(agg)
    
    @app.route('/predict-four', methods=['POST'])
    def predict_four():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR34', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']        
        
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)

        rtn, pct, agg, agg_w, t_ens, comp_predict = get_v_prediction(data_df, models_two)
        print(f"Predict comp_predict Model  {comp_predict}")            
        return str(comp_predict)    
            
        
        
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
    