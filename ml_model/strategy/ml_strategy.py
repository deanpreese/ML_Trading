#from enum import Enum
#import json
#import mlflow
import pandas as pd
#import requests
#import datetime as dt
#import random as rand
#from models import wrapped_models

from strategy.common_strategy import CommonStrategy

class MLStrategy (CommonStrategy):
    def __init__(self, model, col_filter, run_id):
        super().__init__()
        
        self.model = model
        self.column_filter = col_filter
        self.run_id = run_id
        self.run_name = ""
        self.trader_id = 0
        self.trader_group = 25
        self.model_name = ""

    def do_predict(self,data):
        
        self.set_predict_data(data) 
        
        XD = data
        if len(self.column_filter) == 0:
            self.column_filter = list(data.keys())
        else:
            XD = data[self.column_filter]
            
        df = pd.DataFrame(XD.to_numpy().reshape(1, -1), columns=self.column_filter)          
            
        return self.model.predict(df)
        
  