#import json
#import mlflow
#import pandas as pd
#import requests
#import datetime as dt
#import random as rand

#from models import wrapped_models

from strategy.common_strategy import  CommonStrategy


class CompositeStrategy (CommonStrategy):

    def __init__(self): 
        super().__init__()
           
        self.comp_run_name = ""
        self.comp_run_id = 0 
        self.strategy_models = []   
        
        self.run_id = 0
        self.run_name = ""
        
        
        self.long_threshold = 0.5
        self.short_threshold = -0.5
        self.long_big_threshold = 1.5
        self.short_big_threshold = -1.5        
        
        self.trader_id = 0
        self.trader_group = 0

    
    def do_predict_base(self,data):

        agg_predict = 0    
        agg_weighted_predict = 0    
        self.set_predict_data(data) 
        
        predicts = []
        
        for m in range(len(self.strategy_models)):
            
            perf = self.strategy_models[m].metrics["Perf"]
            predict = self.strategy_models[m].do_predict(data)
            
            if "Classifier" in self.strategy_models[m].model_name:
                predict = (predict - 0.5) *2 
            
            agg_weighted_predict += predict * perf
            agg_predict += predict
            predicts.append(float(round(predict[0],6)))
            
        return agg_predict, agg_weighted_predict, predicts
    

    def do_predict(self,data):

        agg_predict, agg_weighted_predict, predicts = self.do_predict_base(data)
                    
        if agg_predict > self.long_big_threshold or agg_weighted_predict > self.long_threshold:
            return_predict = 1
        elif agg_predict < self.short_big_threshold or agg_weighted_predict < self.short_threshold:
            return_predict = -1    
        else:
            return_predict = 0            
            
        return return_predict
    
    
    
    def do_predict_v(self,data):

        count_above_threshold_predict = 0
        agg_predict, agg_weighted_predict, predicts = self.do_predict_base(data)
            
        count_u = sum(1 for x in predicts if x > 0)             
        count_u_w = sum(1 for x in agg_weighted_predict if x > 0)             
        
        total_items = len(predicts)
        percentage_positive = (count_u / total_items) 
        percentage_positive_w = (count_u / total_items) 
        
        if agg_predict > self.long_threshold or agg_weighted_predict > self.long_threshold:
            count_above_threshold_predict = 1
        elif agg_predict < self.short_threshold or agg_weighted_predict < self.short_threshold:
            count_above_threshold_predict = -1         

        if agg_predict > self.long_big_threshold or agg_weighted_predict > self.long_big_threshold:
            count_above_threshold_predict += 3
        elif agg_predict < self.short_big_threshold or agg_weighted_predict < self.short_big_threshold:
            count_above_threshold_predict += -3         


        return percentage_positive, percentage_positive_w, count_above_threshold_predict, agg_predict, agg_weighted_predict
    
    