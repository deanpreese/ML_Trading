

from datetime import datetime
from time import time
import mlflow
import pandas as pd
from common.common_func import full_split_and_scale
from common.wrapped_models import TunableCatBoostClassifier, TunableLGBMClassifier, TunableXGBClassifier


def RunData(datafile, models):
        pass

datafile = [ 
        'data/ReFried_5M_ALL.csv',   #0
        'data/buildSeqInd_13X_5M_ALL.csv',  #1
        'data/buildSeqIndX_5M_ALL.csv',  #2
        'data/buildSeqInd_Lucky13_5M_3070.csv',  #3
        'data/markov.csv',  #4
        'data/ReFried_5M_3070.csv',  #5
        'data/buildSeqInd_Lucky13_5M_ALL.csv',  #6
        'data/buildSeqInd_13X_5M_3070.csv',  #7
        'data/buildSeqIndX_5M_3070.csv',  #8
        'data/Fractal_ALL_5M.csv', #9 Classifier Only  
        'data/Fractal_ALL_5M_30_70.csv', #10 Classifier Only
    ]


data = pd.read_csv(datafile[9])
X_train, X_test, y_train, y_test, input_features  = full_split_and_scale(data, 2, 0.2, 0, 'outputC')  

feature_names = data.columns


models = {}
models['lgbc'] = TunableLGBMClassifier()
models["xgbc"] = TunableXGBClassifier()
models['cac'] = TunableCatBoostClassifier()

all_predict_data = pd.DataFrame()
perf_data = [] 
 
time_stamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
exp_name = f"comp_cls_{time_stamp}"

try:
        experiment_id = mlflow.create_experiment(exp_name)
except Exception as e:
        print(f"{e}")
        experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id        
        
 
for key in models.keys():

        start = time.time()
        print(f"fitting and predicting model {key} ")
        
        models[key].SetFeatureList( list(X_train.columns) )       
        
        accuracy, precision, recall, TNPerc, FPPerc, FNPerc, TPPerc, tot, y_pred, pred_proba = models[key].TrackInMLFlow(experiment_id, True, models[key], X_train, y_train, X_test, y_test)        

        all_predict_data[key] = y_pred
        k = f"{key}-proba"
        all_predict_data[k] = pred_proba.tolist()

        t = [key, accuracy, precision, recall, TNPerc, FPPerc, FNPerc, TPPerc, tot]
        perf_data.append(t)    
        end = time.time()
        print(f" {key} execution - {end-start}")


        all_predict_data["target"] = y_test

        #correctX, correctY, correctP, totalX, cxp, cyp, cpp = CalcClassResults(all_predict_data, models)

        p_df = pd.DataFrame(perf_data)    
        p_df.columns = ["Model", "Accuracy", "Precision", "Recall", "True Neg", "False Pos", "False Neg", "True Pos", "Total"]
        p_df.sort_values(by=['Accuracy'], ascending=False, inplace=True)

                   
        #print(" ")
        #print(f"Correct X% : {correctX}     {cxp}" )
        #print(f"Correct Y% : {correctY}     {cyp}" )
        #print(f"Correct P% : {correctP}     {cpp}" )
        #print(f"Total Predicted {totalX} ")  
        print(" ")
        print(p_df)
        print(" ")