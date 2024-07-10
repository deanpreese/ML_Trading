

from common.model_loader import ModelLoader
from common.order_manager import OrderManager

import pandas as pd
#import datetime as dt
#import random as rand
import time

import logging
logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)
logging.getLogger('mlflow.pyfunc').setLevel(logging.ERROR)

def load_models(exp_id, n_models, group_id):
    
    experiment_id = exp_id
    num_models = n_models    
    model_loader = ModelLoader()
    return model_loader.load_top_models( experiment_id, num_models, group_id)

def run_sim(exp_id, n_models, file, trades, delay, group_id):


    data = pd.read_csv(file)   
    num_columns = len(data.axes[1]) 
    input_features =  num_columns -2
    X = data.iloc[:, 0:input_features]  
    y = data["output"].values
    
    models = load_models(exp_id, n_models, group_id)

    order_manager = OrderManager()
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
    
    
exp_idx = ["41"]
group_id = 41


#num_models = 10
num_models = 25
#num_models = 2


#trades = 500
trades = 250
#trades = 100
#trades = 10


#sim_delay = 0.0500
#sim_delay = 0.025
sim_delay = 0.0000000000002


#file = "data/lucky13_short.csv"
    
file = "data/lucky13_oos.csv"    
    
run_sim(exp_idx, num_models, file, trades, sim_delay, group_id)    

