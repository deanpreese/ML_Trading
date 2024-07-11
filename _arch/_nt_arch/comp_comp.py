
from common.model_loader import ModelLoader
from common.order_manager import OrderManager

import pandas as pd
#import datetime as dt
#import random as rand
import time

import logging
logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)
logging.getLogger('mlflow.pyfunc').setLevel(logging.ERROR)

#from models import wrapped_models

#from  common.CommonCli import CommonCli as common_cli
#from CompositeStrategy import CompositeStrategy    

def load_models(exp_id, n_models, group, api_url):
    
    experiment_id = exp_id
    num_models = n_models    
    model_loader = ModelLoader()
    model_loader.init_api(api_url)
    return model_loader.load_composite_models( experiment_id, num_models, group)


def run_sim(models_in, file, trades, delay):


    data = pd.read_csv(file)   
    num_columns = len(data.axes[1]) 
    input_features =  num_columns -2
    X = data.iloc[:, 0:input_features]  
    y = data["output"].values
    
    models = models_in

    order_manager = OrderManager()
    order_manager.is_sim(True)
    
    
    total = 0
    order_total = 0
    start = time.time()

    for i in range(len(y)):
           

            for m in range(len(models)):
                
                predict = models[m].do_predict(X.iloc[i])
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
    

# ------------------
    
api_u = "http://10.0.0.147:8786/"    
    
models1 = load_models(["31"], 1, 31, api_u)
models2 = load_models(["36"], 1, 36, api_u)
models3 = load_models(["33"], 1, 33, api_u)



models_agg = models1 + models2 + models3 

#trades = 750
trades = 250
#trades = 10


#sim_delay = 0.05
sim_delay = 0.025
#sim_delay = 0.00000002
    
file = "data/lucky13_oos.csv"    
    
run_sim(models_agg, file, trades, sim_delay)    

