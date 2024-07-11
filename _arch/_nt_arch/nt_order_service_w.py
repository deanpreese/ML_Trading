#import csv
#import datetime
import json
from flask import Flask, request
#import numpy as np
import pandas as pd
#from io import StringIO
#import cProfile
from io import BytesIO
import time as mytime
import logging

from common.model_loader_w import ModelLoader_W
from common.order_manager import OrderManager

cli_api = "http://10.0.0.240:8786/"

order_manager = OrderManager(cli_api)
order_manager.is_sim(False)

model_loader = ModelLoader_W()
model_loader.init_api(cli_api)

models = []


logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)
logging.getLogger('mlflow.pyfunc').setLevel(logging.ERROR)


def LoadModels():
    m = []

    group_id = 81
    experiment_id = ["81"]
    num_models = 2    

    # USed for base models
    #m = model_loader.load_random_models(experiment_id, 5)
    
    #used for comp models
    #m1 = model_loader.load_composite_models( experiment_id, num_models, group_id)
    m1 = model_loader.load_composite_models( experiment_id, num_models, group_id)

    # used for comp models by feature count
    #m2 = model_loader.load_composite_models_by_feature_count( experiment_id, num_models, group_id)
    
    m = m1 
        
    return m

def init_app():
    app = Flask(__name__)

    with app.app_context():
        models = LoadModels()
       
       
    @app.route('/predict', methods=['POST'])
    def predictx():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR34', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1', 'output', 'outputC', 'actual']
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        
        time_raw = data_df["time"]
        px_f = float(data_df["actual"][0])

        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)
        reshaped_df = data_df.stack().reset_index(level=0, drop=True)
                
        orders = []        
        order_manager.process_tick_rt(px_f,time_raw[0])
                
        for m in range(len(models)):
            
            loaded_prediction = models[m].do_predict(reshaped_df)            
            #loaded_prediction = models[m].do_predict_x(reshaped_df)    
                    
            orders = orders + order_manager.process_model_ninja_data(models[m], px_f, loaded_prediction)
            #mytime.sleep(0.025)
        
        return json.dumps(orders,default=str)
        
    @app.route('/close-all-open', methods=['POST'])
    def close_all_open():
        for m in range(len(models)):
            order_manager.close_all(models[m])
    
        return "ok"
    
    
    @app.route('/reload-models', methods=['POST'])
    def reload_models():
        models = LoadModels()
        for m in range(len(models)):
            m.position = 0
            
        return "ok"
    
        
    return app

   

app = init_app()
    
if __name__ == '__main__':
    print("Starting Flask application.")
      
    
    app.run(
        debug=True, 
        use_reloader=False,
        port=9898, 
        host='0.0.0.0'
        )
    