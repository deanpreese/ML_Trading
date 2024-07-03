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
        
        
        self.long_threshold = 0
        self.short_threshold = 0
        self.long_big_threshold = 0
        self.short_big_threshold = 0        
        
        self.trader_id = 0
        self.trader_group = 0


    def do_reg_predict(self, agg_predict, agg_weighted_predict): 
        
        return_predict = 0   
        
        if agg_predict > self.long_big_threshold or agg_weighted_predict > self.long_threshold:
            #return_predict = max(agg_predict, agg_weighted_predict)[0]
            return_predict = 1
        elif agg_predict < self.short_big_threshold or agg_weighted_predict < self.short_threshold:
            #return_predict = min(agg_predict, agg_weighted_predict)[0]
            return_predict = -1    
        else:
            return_predict = 0
        
        return return_predict
    


    def do_class_predict(self,data):
        pass

        

    def do_predict(self,data):

        agg_predict = 0    
        agg_weighted_predict = 0    
        return_predict = 0
        
        self.set_predict_data(data) 
        
        for m in range(len(self.strategy_models)):
            
            perf = self.strategy_models[m].metrics["Perf"]
            predict = self.strategy_models[m].do_predict(data)
            
            agg_weighted_predict += predict * perf
            agg_predict += predict
            
        """
        if agg_predict > self.long_big_threshold or agg_weighted_predict > self.long_threshold:
            #return_predict = max(agg_predict, agg_weighted_predict)[0]
            return_predict = 1
        elif agg_predict < self.short_big_threshold or agg_weighted_predict < self.short_threshold:
            #return_predict = min(agg_predict, agg_weighted_predict)[0]
            return_predict = -1    
        else:
            return_predict = 0
        """      
        
        return self.do_reg_predict(agg_predict, agg_weighted_predict) 
    
    

    
    
    
    
    def do_predict_v(self,data):

        return_predict = 0
        
        self.set_predict_data(data) 
        
        predicts = []
        
        for m in range(len(self.strategy_models)):
            
            predict = self.strategy_models[m].do_predict(data)
            predicts.append(predict)
            
        count_u = sum(1 for x in predicts if x > 0)             
        total_items = len(predicts)
        percentage_positive = (count_u / total_items) 
        
        return_predict = 0
        if percentage_positive > 0.69:
            return_predict = 1

        if percentage_positive < 0.31:
            return_predict = -1
                     
        return return_predict
    
    
    def do_predict_x(self,data):

        agg_predict = 0    
        agg_weighted_predict = 0    
        return_predict = 0
        
        self.set_predict_data(data) 
        
        for m in range(len(self.strategy_models)):
            
            perf = self.strategy_models[m].metrics["Perf"]
            predict = self.strategy_models[m].do_predict(data)
            
            agg_weighted_predict += predict * perf
            agg_predict += predict

            return_predict = agg_predict

        
        return return_predict