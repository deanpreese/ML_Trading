from models.wrapped_models import *

experiment_id = ['405381979368050886']

runs = mlflow.search_runs(experiment_id)

#list_size = 1
#rrs = random.sample(range(1, len(runs)-5 ), list_size )
#random_rows = runs.sample(n=1)
#for index, run in random_rows.iterrows():    

for index, run in runs.iterrows():    

    run_id = run.run_id
    rinfo = mlflow.get_run(run_id)
    run_txt = f"runs:/{run_id}/model"    
    
    #for x , a in rinfo.data.metrics.items() :
    #        print(f"{x}    {a} ")        
    
    #print("   ")
    
    #for x , a in rinfo.data.params.items() :
    #        print(f"{x}    {a} ")        

    
    #history = rinfo.data.tags['mlflow.log-model.history']
    #runs_data = json.loads(history)        
    #for run_info in runs_data:        
    #    print(run_info['artifact_path'])
    #    print(run_info)
    
    print("  ")
    print(rinfo.data.tags['mlflow.loggedArtifacts'] )
    print("   ")
   
    art = json.loads(rinfo.data.tags['mlflow.loggedArtifacts'])
    print(art[0].get('path', None))
    print("   ")
    
    #for x , a in rinfo.data.tags.items() :
    #        print(f"{x}    {a} ")        

    #print("   ")
    #print(" -------  ")
    #print("   ")
        
    print(rinfo.info.artifact_uri )        
    print("   ")
        
    #for x ,a in rinfo.info :
    #        print(f"{x}    {a} ")        

    #print("   ")
    #print(" -------  ")
    #print("   ")
    
