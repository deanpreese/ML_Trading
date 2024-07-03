import json
import mlflow
import pandas as pd
import requests
import datetime as dt
import random as rand

from common.model_loader import ModelLoader


#import daal4py as d4p 

from common.ml_strategy import MLStrategy
from common.composite_strategy import CompositeStrategy


class ModelConfig:
    def __init__(self, config_dict):
        self.config = config_dict
    
    def get_config(self, model_type, key=None):
        """
        Retrieve specific configuration settings for a given model type.
        If key is specified, return a specific value; otherwise, return the whole config for the model.
        """
        model_config = self.config.get(model_type, {})
        if key:
            return model_config.get(key)
        return model_config

    def list_all_models(self):
        """ List all model types available in the configuration. """
        return list(self.config.keys())



class ModelLoader_W(ModelLoader):
    
    def add_model(self, rid, isReg):
        rinfo = mlflow.get_run(rid)
        run_txt = f"runs:/{rid}/model" 
        
        selected_model = mlflow.pyfunc.load_model(run_txt)
        model_config = ModelConfig(selected_model.metadata.flavors)

        all_m = model_config.list_all_models()
        #print(all_m)

        loaded_model = selected_model

        if 'lightgbm' in all_m:
            print(model_config.get_config('lightgbm')['data'])
            loaded_model = mlflow.lightgbm.load_model(run_txt)

        if 'xgboost' in all_m:
            print(model_config.get_config('xgboost')['data'])
            loaded_model = mlflow.xgboost.load_model(run_txt)

        cols =[]
        try:
            art = json.loads(rinfo.data.tags['mlflow.loggedArtifacts'])
            art_file = art[0].get('path', None)
            art_uri = rinfo.info.artifact_uri
            art_to_load = f"{art_uri}/{art_file}"
            print(art_to_load)
            arti_d = mlflow.artifacts.load_dict(art_to_load)
            cols = [x[0] for x in arti_d['data'] if x[0] != 'output']
            self.l_artifacts.append(cols)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            cols = []    
        
        #self.l_models.append(loaded_model)
        lm = MLStrategy(loaded_model, cols,rid)
        lm.trader_group = self.model_group
        lm.run_name = rinfo.info.run_name
        lm.metrics = rinfo.data.metrics
        lm.perf = rinfo.data.metrics["Perf"]
        
        if isReg:
            t_id = self.common_cli.initialize_trader(lm.run_name, lm.trader_group)
            lm.trader_id = t_id
            
            print(f"Trader {t_id} initialized for {lm.run_name}")
            
        else:
            lm.trader_id = 0    
        
        self.model_list.append(lm)        
                
        return loaded_model, cols