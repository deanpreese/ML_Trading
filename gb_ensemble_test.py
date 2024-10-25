
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
import datetime  as dte_time
import ml_model.model_run_data as mrd
import itertools

from strategy.model_loader import ModelLoader

def run_sim(df, models, target):
 
    X = df
    X = X.drop(columns=['output', 'outputC'])
    y = df[target].values
    fl_out = list(X.columns)

    percent_pos_rtn = 0
    combined_rtn = 0
    agg_rtn = 0
    agg_w_rtn = 0
    comp_rtn = 0
    agg_agree = 0

    y_count = 0

    print(" ")
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
        
            #if y[i] > 0.5:
            if y[i] > 0:
                
                y_count += 1
            
                if final_agg > 0:
                    agg_rtn += 1
                
                if final_agg_w > 0:
                    agg_w_rtn += 1
                
                if comp_predict > 0:
                    comp_rtn += 1
                
                if agg_weighted_predict > 0  and agg_predict > 0:
                    agg_agree += 1
                    
                
            #if y[i] < -0.5:
            if y[i] < 0:                
                
                y_count += 1
        
                if final_agg < 0:
                    agg_rtn += 1
                        
                if final_agg_w < 0:
                    agg_w_rtn += 1

                if comp_predict < 0:
                    comp_rtn += 1
                    
                if agg_weighted_predict < 0  and agg_predict < 0:
                    agg_agree += 1
                                                  
            if y[i] == 0:
                
                y_count += 1
        
                if final_agg == 0:
                    agg_rtn += 1
                        
                if final_agg_w == 0:
                    agg_w_rtn += 1

                if comp_predict == 0:
                    comp_rtn += 1
                    
                if agg_weighted_predict == 0  and agg_predict == 0:
                    agg_agree += 1
                                    
                    
                                       
    print("")
    print("Predict Results")
    print(f"Combined {round(comp_rtn/y_count,4)}  ")
    print(f"Aggregate {round(agg_rtn/y_count,4)} ")
    print(f"Aggregate Weighted {round(agg_w_rtn/y_count,4)}  ")
    print(f"Aggregate Agree {round(agg_agree/y_count,4)}  ")
    print(" ")

    


def run_virtuaL_test(file, model_list ):
    
    target = 'output'
    model_loader = ModelLoader()
    models = model_loader.load_virtual_composite_model(model_list)  
    run_sim( file, models, target)



if __name__ == "__main__":
    
    datafile = [ 
        'data/Lucky13_3070_oos.csv',   
        'data/Lucky13_3070.csv',  #1
        'data/Lucky13_PLUS_3070_oos.csv',   
        'data/Lucky13_PLUS_3070.csv',  #3
    ]

    df = pd.read_csv(datafile[0])                   
    #df = df[((df['RSI'] > 20) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 80)]  
    #df = df[((df['RSI'] > 25) & (df['RSI'] < 40))|(df['RSI'] > 60) & (df['RSI'] < 75)] 
    #df = df[(df['RSI'] > 60)]  
    #df = df[(df['RSI'] < 40)]  
    
    run_test = True
        
    if run_test:
        exp =  ["11"]
        target = 'output'
        model_loader = ModelLoader()
        models = model_loader.load_composite_strategy(exp, 1, 0)   
        run_sim( df, models, target)
    
    
    """
    x_mc = 1
    x_ad = "ASC"
    x_exp = "= 392"
    x_feat = 7
    list_x= mrd.fetch_data(x_mc, x_ad, x_exp, x_feat, "'R2'")

    x_mc =  1 
    x_ad = "DESC"
    x_exp = "= 370"
    x_feat = 7
    list_x_2= mrd.fetch_data(x_mc, x_ad, x_exp, x_feat, "'R2'")

    x_mc = 1 
    x_ad = "ASC"
    x_exp = "= 440"
    x_feat = 0
    list_x_3= mrd.fetch_data(x_mc, x_ad, x_exp, x_feat, "'R2'")
    
    list_x = list_x + list_x_2 + list_x_3
    #list_x = list_x_3
    #model_list = list(set(list_x))

    model_list = ["f2f83a4dfe11408b934290ec03defb47","2092c7f4c4e24cfa826cff0e217e8222","b3026c0e4a744fb6ad15e8eee4a8cacf","626ba76f501940f5aa9f0aa8d088b312"
            ,"2c6b620d3cd947dab63fddeaada653c0","e0a6df73c852461fbf0152d7dd4a5537","59564d88ee4b43249e035c0a03f3e0b9","015b013331a642ee9173c91c870e4991"]        
    run_virtuaL_test(datafile[0], model_list )
    """