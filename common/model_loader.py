import json
import mlflow
import pandas as pd
import requests
import datetime as dt
import random as rand

from common.ml_strategy import MLStrategy
from common.composite_strategy import CompositeStrategy

from models import wrapped_models

from  common.common_cli import CommonCli 



class ModelLoader:

    def __init__(self):
        self.experiment_id = 0
        #self.l_models = []
        self.l_artifacts = []
        self.model_list = []
        self.model_group = 0

    def init_api(self, api_url):
        self.common_cli = CommonCli(api_url)

    # -------------------------
    # Main add_model function
    # -------------------------
    def add_model(self, rid, isReg):
        rinfo = mlflow.get_run(rid)
        run_txt = f"runs:/{rid}/model" 
        
        loaded_model = mlflow.pyfunc.load_model(run_txt)
        
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
            print(cols)
            
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
                
        return loaded_model, cols, lm


    def load_models_by_run_ids(self, run_ids, group_id):
        
        self.model_group = group_id
        
        ml = []
        
        for rid in run_ids:
            model, cols, lm = self.add_model(rid, True)
            ml.append(lm)
            
        return ml            

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
    
    def load_top_models(self, experiment_id, num_models, group_id): 
        
        print("Querying Runs ...")
        runs = mlflow.search_runs(experiment_ids=experiment_id, filter_string="", order_by=["metrics.MSE DESC"], max_results=num_models)
        
        for i in range(len(runs)):
            r_id = runs.iloc[i].run_id 
            self.model_group = group_id
            print(f"Run Id     {r_id}")
            
            if(self.model_group > 0):
                self.add_model(r_id, True)
            else:
                self.add_model(r_id, False)                
        
        return self.model_list     

    
    
    def load_random_models(self, experiment_id, num_models): 
        
        print("Querying Runs ...")
        #runs = mlflow.search_runs(experiment_ids=experiment_id, filter_string="", order_by=["metrics.MSE DESC"], max_results=num_models)
        runs = mlflow.search_runs(experiment_id)
        idxx = rand.sample(range(1, len(runs) -2 ), num_models)
        print("Random Runs Selected...")
        
        for i in idxx:
            r_id = runs.iloc[i].run_id 
            print(f"Run Id     {r_id}")
            self.add_model(r_id, True)
        
        return self.model_list     


    # -------------------------
    def load_composite_models_by_feature_count(self, experiment_id, num_models, group_id): 
        
        print("Querying Runs ...")
        runs = mlflow.search_runs(experiment_ids=experiment_id, filter_string="", order_by=["metrics.FeatureCount DESC"], max_results=num_models)
       
        self.model_group = group_id
       
        comp_strategies = []
        
        for i in range(len(runs)):
            self.model_list = []    
            
            r_id = runs.iloc[i].run_id 
            print(f"Run Id     {r_id}")
            loaded_strat = self.add_composite_strategies(r_id, group_id)
            comp_strategies.append(loaded_strat)
        
        return comp_strategies    

        
    def load_composite_models(self, experiment_id, num_models, group_id): 
        
        print("Querying Runs ...")
        runs = mlflow.search_runs(experiment_ids=experiment_id, filter_string="", order_by=["metrics.cpp DESC"], max_results=num_models)
       
        self.model_group = group_id
       
        comp_strategies = []
        
        for i in range(len(runs)):
            self.model_list = []    
            
            r_id = runs.iloc[i].run_id 
            print(f"Run Id     {r_id}")
            loaded_strat = self.add_composite_strategies(r_id, group_id)
            comp_strategies.append(loaded_strat)
        
        return comp_strategies    




    def add_composite_strategies(self, rid, group_id):
        
        rinfo = mlflow.get_run(rid)
        
        comp_strat = CompositeStrategy()
        comp_strat.run_id = rid
        comp_strat.run_name = rinfo.info.run_name   
        comp_strat.trader_group = group_id     

        t_id = 1
        if group_id > 0:
            t_id = self.common_cli.initialize_trader(comp_strat.run_name, comp_strat.trader_group)
            
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
                        print( item_data[8])
                        self.add_model(item_data[8], False)
                        
            comp_strat.strategy_models = self.model_list    
            return comp_strat
                    
        except Exception as e:
            print(f"An error occurred: {e}")

