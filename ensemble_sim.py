from flask import Flask, request
import numpy as np
import pandas as pd
from io import BytesIO
import time 
import data_s.model_run_data as mrd

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



def run_sim(file, models):

    data = pd.read_csv(file)                   
    X = data
    X = X.drop(columns=['output', 'outputC'])
    #X.columns = ['SDLR310', 'SDBB91', 'SDKC91', 'SDKC9', 'ROC', 'ATR33', 'ATR32', 'ATR31', 'ATR3', 'ATR21', 'ATR2', 'RSI', 'STOK1']

    y = data['output'].values
    fl_out = list(X.columns)
    start = time.time()

    p_pct = 0
    p_rtn = 0
    p_agg = 0
    p_agg_w = 0
    p_ens = 0
    p_cc = 0


    for i in range(len(y)):
        
            t_pct_cnt = 0
            t_rtn = 0
            t_agg = 0
            t_agg_w = 0
            t_ens = 0
            
    
            for m in range(len(models)):
                percentage_positive, return_predict, agg_predict, agg_weighted_predict = models[m].do_predict_v(X.iloc[i])
                
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

            #print(f"Raw      {rtn} {t_pct_cnt} {t_agg} {t_agg_w} {len(models)}")  
            #print(f"DIV      {rtn} {pct} {agg}    {agg_w}   {t_ens} {y[i]}")  

            comp_predict = ((0.49 * rtn) + (0.53 * pct) + (0.44 * agg) + (0.42 * agg_w) + (0.46 * t_ens))/5

            if y[i] > 0:
                if rtn > 0:
                    p_rtn += 1

                if  pct > 0.5 :
                    p_pct += 1        
        
                if agg > 0:
                    p_agg += 1
                
                if agg_w > 0:
                    p_agg_w += 1
                
                if t_ens > 0:
                    p_ens += 1
                    
                if comp_predict > 0:
                    p_cc += 1
                
            if y[i] < 0:
                if rtn < 0:
                    p_rtn += 1

                if  pct < 0.5 :
                    p_pct += 1        
        
                if agg < 0:
                    p_agg += 1
                        
                if agg_w < 0:
                    p_agg_w += 1

                if t_ens < 0:
                    p_ens += 1  

                if comp_predict < 0:
                    p_cc += 1
                                      
                                       
    
    print(" ")                
    print(f"Total Predict {p_rtn} {p_pct} {p_agg} {p_agg_w} {p_ens} {p_cc} {len(y)} ")
    print(f"Total Predict {round(p_rtn/len(y),4)} {round(p_pct/len(y),4)} {round(p_agg/len(y),4)}  {round(p_agg_w/len(y),4)}  {round(p_ens/len(y),4)}  {round(p_cc/len(y),4)}  {len(y)} ")
    print(" ")                

    end = time.time()
    t = round(end-start,2)
    print(f"Time {t} seconds to process {len(y)} predictions  --  {round(len(y)/t,2)}/sec ")
    print(" ") 
    

def run_test():

    model_loader = ModelLoader()
    
    #run_list = mrd.fetch_r2()
    #run_list = mrd.fetch_perf()
    #print(run_list)
        
    #models = model_loader.load_virtual_composite_model(run_list)            
        
    exp_idx = ["66"]
    num_models = 1
    models = model_loader.load_composite_models(exp_idx, num_models, 0)        
    
    file = "data/lucky13_oos.csv"    
    
    run_sim(file, models)    

if __name__ == "__main__":
    run_test()
