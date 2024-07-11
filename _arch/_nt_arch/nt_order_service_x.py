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

from common.model_loader import ModelLoader
from common.order_manager import OrderManager

cli_api = "http://10.0.0.147:8786/"

order_manager = OrderManager(cli_api)
order_manager.is_sim(False)

model_loader = ModelLoader()
model_loader.init_api(cli_api)

models = []


logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)
logging.getLogger('mlflow.pyfunc').setLevel(logging.ERROR)



def LoadModels():
    
    
    strat_runs85 =    [ 'f2f5d1c95c4a4cc4b66584ab67f47824', '5281f99646454a92932b1c058f211489', '28ed7e2ce7794fb6abd82204b7f9c7ea', 
                       '70efa8a43b3f4f28b026b54aed87cb8b', '68721281d99649a3a93123189f169a02']
   
    strat_runs83 = ['28ed7e2ce7794fb6abd82204b7f9c7ea', '70efa8a43b3f4f28b026b54aed87cb8b', '68721281d99649a3a93123189f169a02']
    
    #og_2
    strat_runs88 =    ['61fc96a504134fc1884477eb7eec73b7', '177e530098794f6aaa53e893b1c2ecc9', 'e0b22b6f127348e99e3b4455eccce320', 'f2f5d1c95c4a4cc4b66584ab67f47824', 
                    '5281f99646454a92932b1c058f211489', '28ed7e2ce7794fb6abd82204b7f9c7ea', '70efa8a43b3f4f28b026b54aed87cb8b', '68721281d99649a3a93123189f169a02']
   
    strat_runs89 = ['e0b22b6f127348e99e3b4455eccce320', 'f2f5d1c95c4a4cc4b66584ab67f47824', 
                    '5281f99646454a92932b1c058f211489']
   
   # -------
   
   
    strat_runs310 = ['e0b22b6f127348e99e3b4455eccce320', '9b32eacda2174ef0be6548229dc3d372', '0b97535148444f48923807597f6a5a43']
    strat_runs3 = ['13d5eeb0625a4115b7ca49c4fec0a730', '37da5caaa8a94dafb5b8cb6161b10670', '026a20f3cf1b43e4b1f38902273e2dde']
   
    strat_runs6 = ['61fc96a504134fc1884477eb7eec73b7', '177e530098794f6aaa53e893b1c2ecc9', 'e0b22b6f127348e99e3b4455eccce320', 
                    '9b32eacda2174ef0be6548229dc3d372', '0b97535148444f48923807597f6a5a43', '13d5eeb0625a4115b7ca49c4fec0a730' ]
   
    og_1 = ['61fc96a504134fc1884477eb7eec73b7', '177e530098794f6aaa53e893b1c2ecc9', 'e0b22b6f127348e99e3b4455eccce320', 
              '9b32eacda2174ef0be6548229dc3d372', '0b97535148444f48923807597f6a5a43', '13d5eeb0625a4115b7ca49c4fec0a730', 
              '37da5caaa8a94dafb5b8cb6161b10670', '026a20f3cf1b43e4b1f38902273e2dde']
      
    group_id = 310
    return model_loader.load_models_by_run_ids(strat_runs310, group_id)
    
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
    