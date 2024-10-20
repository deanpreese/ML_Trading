import json
import mlflow
import pandas as pd
import requests
import datetime as dt
import random as rand
import yaml


from strategy.ml_strategy import MLStrategy
from strategy.composite_strategy import CompositeStrategy

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


import mlflow
mlflow.set_tracking_uri(uri="http://10.0.0.50:8888")



class ModelLoader:

    def __init__(self):
        self.experiment_id = 0
        #self.l_models = []
        self.l_artifacts = []
        self.model_list = []
        self.model_group = 0
        
        self.flavor_list = ['catboost', 'xgboost', 'lightgbm']

         

    # -------------------------
    # Main add_model function
    # -------------------------
    def add_model(self, rid, model_n):
        rinfo = mlflow.get_run(rid)
        run_txt = f"runs:/{rid}/model" 
        
        loaded_model = mlflow.pyfunc.load_model(run_txt)
        print(model_n)
        
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
        lm.model_name = model_n
        lm.trader_group = self.model_group
        lm.run_name = rinfo.info.run_name
        lm.metrics = rinfo.data.metrics
        
        #print(f"Metrics {lm.metrics}")
        
        lm.perf = rinfo.data.metrics["Perf"]
        lm.trader_id = 0    
        self.model_list.append(lm)        
        return loaded_model, cols, lm


    # -------------------------
    def load_selected_models(self, runs):        
        for index, run in runs.iterrows():    
            run_id = run.run_id
            self.add_model(run_id, True)
        return self.model_list

    def load_models(self, experiment_id):
        runs = mlflow.search_runs(experiment_id)
        self.load_selected_models(runs)
        return self.model_list    

    def load_models_by_run_ids(self, run_ids):
        ml = []
        for rid in run_ids:
            model, cols, lm = self.add_model(rid, True)
            ml.append(lm)
        return ml                 
            
            
    def load_virtual_composite_model(self, run_list):
        
        self.model_group = 0
        comp_strategies = []
        comp_strat = CompositeStrategy()
        comp_strat.run_id = 0
        comp_strat.run_name = "virtual_strategy"
        comp_strat.trader_group = 0     
        t_id = 1
        comp_strat.trader_id = t_id

        try:
            for i in range(len(run_list)):
                r_id = run_list[i]
                rinfo = mlflow.get_run(r_id)
                
                model_n = rinfo.data.params["ModelName"]
                print(f"Run Id     {r_id}    {model_n}")
                self.add_model(r_id, model_n)
                        
            comp_strat.strategy_models = self.model_list    
            comp_strategies.append(comp_strat)
        except Exception as e:
            print(f"An error occurred: {e}")            
            
        return comp_strategies    
        
            
    def load_composite_strategy(self, experiment_id, num_models, group_id): 
        
        print("Querying Runs ...")
        #runs = mlflow.search_runs(experiment_ids=experiment_id, filter_string="", order_by=["metrics.cxp DESC"], max_results=num_models)
        #runs = mlflow.search_runs(experiment_ids=experiment_id, filter_string="", order_by=["metrics.cpp DESC"], max_results=num_models)
        runs = mlflow.search_runs(experiment_ids=experiment_id, filter_string="", order_by=["metrics.R2 DESC"], max_results=num_models)
        self.model_group = group_id
        comp_strategies = []

        for i in range(len(runs)):
            self.model_list = []    
            r_id = runs.iloc[i].run_id 
            print(f"Run Id     {r_id}")
            
            rinfo = mlflow.get_run(r_id)
            comp_strat = CompositeStrategy()
            comp_strat.run_id = r_id
            comp_strat.run_name = rinfo.info.run_name   
            comp_strat.trader_group = group_id     

            t_id = 1
            comp_strat.trader_id = t_id

            try:
                art = json.loads(rinfo.data.tags['mlflow.loggedArtifacts'])
                for item in art:
                    if item.get('path') == "all_perf_data.json" :
                        art_file = item.get('path', None)
                        art_uri = rinfo.info.artifact_uri
                        art_to_load = f"{art_uri}/{art_file}"
                        print(art_to_load)
                        arti_d = mlflow.artifacts.load_dict(art_to_load)
                        for item_data in arti_d['data']:
                            
                            if "V2" in item_data[0] :
                                self.add_model(item_data[3], item_data[0])
                            if "V2" not in item_data[0] :
                                self.add_model(item_data[8], item_data[0])
                            
                comp_strat.strategy_models = self.model_list    
                        
            except Exception as e:
                print(f"An error occurred: {e}")

            comp_strategies.append(comp_strat)
        return comp_strategies    
