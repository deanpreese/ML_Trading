from flask import Flask, request
import numpy as np
import pandas as pd
from io import BytesIO
import time 
import data_s.model_run_data as mrd
import itertools

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



def run_sim(file, models, target):

    data = pd.read_csv(file)                   
    X = data
    X = X.drop(columns=['output', 'outputC'])
    y = data[target].values
    fl_out = list(X.columns)
    start = time.time()

    p_pct = 0
    p_rtn = 0
    p_agg = 0
    p_agg_w = 0
    p_ens = 0
    p_cc = 0

    print("Calculating Predictions")

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
    
    return round(p_rtn/len(y),4), round(p_pct/len(y),4)


def run_combos(model_list, file):
    
    combo_perf = []
    
    model_loader = ModelLoader()
    
    all_combinations = []
    for r in range(2,len(model_list)):
        combinations = list(itertools.combinations(model_list, r))
        all_combinations.extend(combinations)
    
    for m in range(len(all_combinations)):
        models = model_loader.load_virtual_composite_model(all_combinations[m])  
        p_rtn, p_pct = run_sim(file, models)
        perf_t = [all_combinations[m], p_rtn , p_pct]
        combo_perf.append(perf_t)
    
    return combo_perf


def run_single(model_list, file, target):
    model_loader = ModelLoader()
    models = model_loader.load_virtual_composite_model(model_list)  
    run_sim(file, models, target)    


def run_test():
    file = "data/lucky13_oos.csv"    
    
    #7601
    list_a = mrd.fetch_data(3, "DESC", "=100", 0, "'R2'")
    list_a = list_a + mrd.fetch_data(1, "DESC", "=100", 5, "'R2'")
    list_a = list_a + mrd.fetch_data(1, "ASC", "=100", 1, "'R2'")
    
    
    x_mc = 4
    x_ad = "DESC"
    x_exp = "= 109"
    x_feat = 2
    list_x= mrd.fetch_data(x_mc, x_ad, x_exp, x_feat, "'R2'")

    x_mc =  2 #0
    x_ad = "DESC"
    x_exp = "= 109"
    x_feat = 5
    list_x_2= mrd.fetch_data(x_mc, x_ad, x_exp, x_feat, "'R2'")

    x_mc = 1 #0
    x_ad = "ASC"
    x_exp = "= 109"
    x_feat = 1
    list_x_3= mrd.fetch_data(x_mc, x_ad, x_exp, x_feat, "'R2'")
    list_x = list_x + list_x_2 + list_x_3
    
    c_mc =  2 #2
    c_ad = "DESC"
    c_exp = "= 111"
    c_feat = 6
    list_c = mrd.fetch_data(c_mc, c_ad, c_exp, c_feat, "'R2'")

    c_mc =  3 #1
    c_ad = "DESC"
    c_exp = "= 111"
    c_feat = 6
    list_c_2 = mrd.fetch_data(c_mc, c_ad, c_exp, c_feat, "'R2'")
    list_c = list_c + list_c_2

    lg_mc = 0
    lg_ad = "DESC"
    lg_exp = "= 113"
    lg_feat = 1
    list_lg = mrd.fetch_data(lg_mc, lg_ad, lg_exp, lg_feat, "'R2'")

    lg_mc = 0
    lg_ad = "ASC"
    lg_exp = "= 113"
    lg_feat = 5
    list_lg_2 = mrd.fetch_data(lg_mc, lg_ad, lg_exp, lg_feat, "'R2'")
    list_lg = list_lg + list_lg_2    

    rf_mc = 0
    rf_ad = "DESC"
    rf_exp = "= 115"
    rf_feat = 0
    list_rf = mrd.fetch_data(rf_mc, rf_ad, rf_exp, rf_feat, "'R2'")

    rf_mc = 0
    rf_ad = "ASC"
    rf_exp = "= 115"
    rf_feat = 4
    list_rf_2 = mrd.fetch_data(rf_mc, rf_ad, rf_exp, rf_feat, "'R2'")
    
    l13_mc = 1
    l13_ad = "DESC"
    l13_exp = "= 63"
    l13_feat = 0
    list_13= mrd.fetch_data(l13_mc, l13_ad, l13_exp, l13_feat, "'R2'")


    list_rf = list_rf + list_rf_2 
    model_list = list(set(list_x + list_c + list_lg + list_rf+ list_13))
    
    
    print(" ")
    print(f"Models {len(model_list)}")
    print(model_list)
    print(" ")
        
    #combo_p = run_combos(model_list, file)
    #df = pd.DataFrame(combo_p)
    #print(df)
    
    target = 'output'
    #run_single(model_list, file, target)
      
    model_loader = ModelLoader()
    models = model_loader.load_composite_models(["66"], 2, 0)        
    models = models + model_loader.load_composite_models(["108"], 2, 0)        
    models = models + model_loader.load_composite_models(["116"], 2, 0)        
    models = models + model_loader.load_composite_models(["68"], 2, 0)        
    run_sim( file, models, target)


if __name__ == "__main__":
    run_test()
