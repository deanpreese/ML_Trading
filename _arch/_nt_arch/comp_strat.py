

from common.model_loader import ModelLoader
from common.order_manager import OrderManager

import pandas as pd
#import datetime as dt
#import random as rand
import time

import logging
logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)
logging.getLogger('mlflow.pyfunc').setLevel(logging.ERROR)

def load_strat_models( group_id, api_url, strat_runs):
    
    model_loader = ModelLoader()
    model_loader.init_api(api_url)
    
    return model_loader.build_composite_strategy(group_id, "comp_comp_strat", strat_runs)

def run_new_sim(file, trades, delay, group_id, api_url, s_runs):

    data = pd.read_csv(file)   
    num_columns = len(data.axes[1]) 
    input_features =  num_columns -2
    X = data.iloc[:, 0:input_features]  
    y = data["output"].values
    
    models = load_strat_models(group_id, api_url, s_runs)
    
    order_manager = OrderManager(api_url)
    order_manager.is_sim(True)
    
    total = 0
    order_total = 0
    start = time.time()

    for i in range(len(y)):
           

            for m in range(len(models)):
                
                predict = models[m].do_predict(X.iloc[i])
                
                #print(X.iloc[i])    
                
                order_manager.process_model(models[m], y[i], predict)            
                order_total += 1
                
                time.sleep(delay)

            total += 1        
            order_manager.process_tick(y[i])

            if total > trades:
                break    


    for m in range(len(models)):
        order_manager.close_all(models[m])

    end = time.time()

    t = round(end-start,2)
    print(f"Time {t} seconds to process {order_total} predictions  --  {round(order_total/t,2)}/sec ")
    print(" ") 
    
    
group_id = 1

trades = 500
#trades = 250
#trades = 100
#trades = 10


#sim_delay = 0.0500
#sim_delay = 0.025
sim_delay = 0.0000000000002


#file = "data/lucky13_short.csv"
    
file = "data/lucky13_oos.csv"    
api_u = "http://10.0.0.147:8786/"    

strat_runs = ['61fc96a504134fc1884477eb7eec73b7', '177e530098794f6aaa53e893b1c2ecc9', 'e0b22b6f127348e99e3b4455eccce320', 
              '9b32eacda2174ef0be6548229dc3d372', '0b97535148444f48923807597f6a5a43', '13d5eeb0625a4115b7ca49c4fec0a730', 
              '37da5caaa8a94dafb5b8cb6161b10670', '026a20f3cf1b43e4b1f38902273e2dde']

run_new_sim( file, trades, sim_delay, group_id, api_u, strat_runs)    

