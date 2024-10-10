
import logging
logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)
logging.getLogger('mlflow.tracking._tracking_service.client').setLevel(logging.ERROR)
logging.getLogger('mlflow.utils.requirements_utils').setLevel(logging.ERROR)

logging.getLogger('mlflow.pyfunc').setLevel(logging.ERROR)
logging.getLogger('lightgbm').setLevel(logging.ERROR)
logging.getLogger('[LightGBM]').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, request
import numpy as np
import pandas as pd
from io import BytesIO
import time 
import ml_model.model_run_data as mrd
import itertools

from strategy.model_loader import ModelLoader

def run_sim(file, models, target):

    df = pd.read_csv(file)                   
    #df = df[((df['RSI'] > 20) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 80)]  
    df = df[((df['RSI'] > 25) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 75)] 
    
    X = df
    X = X.drop(columns=['output', 'outputC'])
    y = df[target].values
    fl_out = list(X.columns)
    start = time.time()

    percent_pos_rtn = 0
    combined_rtn = 0
    agg_rtn = 0
    agg_w_rtn = 0
    comp_rtn = 0
    agg_agree = 0

    y_count = 0

    print("Calculating Predictions")

    for i in range(len(y)):
        
            t_pct_cnt = 0
            t_pct_cnt_w = 0
            t_count_above_threshold = 0
            t_agg = 0
            t_agg_w = 0
            
            for m in range(len(models)):
                percentage_positive, percentage_positive_w, count_above_threshold_predict, agg_predict, agg_weighted_predict= models[m].do_predict_v(X.iloc[i])
                
                t_pct_cnt_w += percentage_positive_w
                t_pct_cnt += percentage_positive
                t_agg += agg_predict   
                t_agg_w += agg_weighted_predict
                t_count_above_threshold += count_above_threshold_predict             

            final_agg = t_agg/len(models) 
            final_agg_w = t_agg_w/len(models)
            
            comp_predict = ((0.46 * final_agg) + (0.54 * final_agg_w)  )
        
            if y[i] > 0.5:
                
                y_count += 1
            
                if final_agg > 0:
                    agg_rtn += 1
                
                if final_agg_w > 0:
                    agg_w_rtn += 1
                
                if comp_predict > 0:
                    comp_rtn += 1
                
                if agg_weighted_predict > 0  and agg_predict > 0:
                    agg_agree += 1
                    
                
            if y[i] < -0.5:
                
                y_count += 1
        
                if final_agg < 0:
                    agg_rtn += 1
                        
                if final_agg_w < 0:
                    agg_w_rtn += 1

                if comp_predict < 0:
                    comp_rtn += 1
                    
                if agg_weighted_predict < 0  and agg_predict < 0:
                    agg_agree += 1
                                                  
                    
                    
                                       
    print("")
    print("Predict Results")
    print(f"Combined {round(comp_rtn/y_count,4)}  ")
    print(f"Aggregate {round(agg_rtn/y_count,4)} ")
    print(f"Aggregate Weighted {round(agg_w_rtn/y_count,4)}  ")
    print(f"Aggregate Agree {round(agg_agree/y_count,4)}  ")
    print(" ")

    
    end = time.time()
    t = round(end-start,2)
    
    print(F"Predictions processed in range {y_count}")
    print(f"Time {t} seconds to process {len(y)} predictions  --  {round(len(y)/t,2)}/sec ")
    print(" ") 
    


def run_test():
    
    datafile = [ 
            'data/Lucky13_3070_oos.csv',   
            'data/Lucky13_3070.csv',  #1
            'data/ndata_diff_lucky13_3070_oos.csv', 
            'data/ndata_diff_lucky13_3070.csv', #3
            'data/ndata_lucky_13_lag_3070_oos.csv', 
            'data/ndata_lucky13_lag_3070.csv', #5
            'new_model_Z_lucky13_3070_oos.csv',
            'new_model_Z_lucky13_3070.csv' #7,

    ]

    target = 'output'
    model_loader = ModelLoader()
    models = model_loader.load_composite_strategy(["312"], 1, 0)   
    run_sim( datafile[0], models, target)


if __name__ == "__main__":
    run_test()
