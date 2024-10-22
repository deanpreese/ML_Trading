import os
from flask import Flask, request, jsonify
import pandas as pd
from io import BytesIO
import json
import tensorflow as tf
import numpy as np
from datetime import datetime, timedelta

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


#tf.config.set_visible_devices([], 'GPU')


model_loader = ModelLoader()
models_one = []
models_two = []
models_three = []
models_four = []
models_five = []

keras_upper_models = []
keras_lower_models = []


# ----------------------------------------

def load_keras_models(model_dir):
    
    model_list = []
    files_ = [ f for f in os.listdir(model_dir) if f.endswith('.keras') ]
    for f in files_:
        fm = os.path.join(model_dir, f)
        print(f"Loading ... {fm}")
        xm = tf.keras.models.load_model(fm)
        model_list.append(xm)
        
    return model_list


def load_models(group_id, experiment_id, num_models):
    return model_loader.load_composite_strategy( experiment_id, num_models, group_id)    

def format_json(output_data):
    data_serializable = {
        "agg_prediction": float(output_data["agg_prediction"]),
        "agg_weighted_prediction": float(output_data["agg_weighted_prediction"]),
        "all_predicts": [float(pred) for pred in output_data["all_predicts"]]
    }
    print(f"OUTPUT DATA   {data_serializable}")
    return data_serializable

def get_model_predictions_base(data_df, models):
    all_agg_predicts = 0
    all_agg_weighted = 0
    all_predicts = []
    for m in range(len(models)):
        agg_predict, agg_weighted_predict, predicts = models[m].do_predict_base(data_df)
        
        all_agg_predicts += agg_predict
        all_agg_weighted += agg_weighted_predict
        all_predicts += predicts
    return all_agg_predicts, all_agg_weighted, all_predicts
        
def get_model_predictions(data_df, models):
    all_agg_predicts, all_agg_weighted, all_predicts = get_model_predictions_base(data_df, models)
    output_data = {
            "agg_prediction" : round(all_agg_predicts[0],6),
            "agg_weighted_prediction" : round(all_agg_weighted[0],6),
            "all_predicts": all_predicts
        }
    out_data = format_json(output_data)
    return out_data

def gen_zero_predictions():
    output_data = {
            "agg_prediction" : float(0.0),
            "agg_weighted_prediction" : float(0.0),
            "all_predicts": [float(0.0)]
        }
    #out_data = format_json(output_data)
    return output_data

def run_predicts(x_val, model_list):
    
    x_val = x_val.reshape((1, 14, 1)) 
    predicts = 0
    for m in range(len(model_list)):
        predicts += model_list[m].predict(x_val)[0]
        
        print(m)

    count_up = len([x for x in predicts if x > 0])
    cnt_pct = count_up/len(predicts)

    if cnt_pct > .5:
        y_val = 1
    else:
        y_val = -1

    #y_val = predicts/len(model_list)
    return y_val



# ----------------------------------------
def init_app():
    app = Flask(__name__)

    with app.app_context():
        
        c_kan_model_dir_upper = "saved_models/c_kan/upper"
        c_kan_model_dir_lower = "saved_models/c_kan/lower"
        #keras_upper_models = load_keras_models(c_kan_model_dir_upper)
        #keras_lower_models = load_keras_models(c_kan_model_dir_lower)    
        
        dcnn_model_dir_upper = "saved_models/dcnn_ens/upper"
        dcnn_model_dir_lower = "saved_models/dcnn_ens/lower"
        #keras_upper_models = load_keras_models(dcnn_model_dir_upper)
        #keras_lower_models = load_keras_models(dcnn_model_dir_lower)    
        
        comp_model_dir_upper = "saved_models/comp/upper_d"
        comp_model_dir_lower = "saved_models/comp/lower_d"
        keras_upper_models = load_keras_models(comp_model_dir_upper)
        keras_lower_models = load_keras_models(comp_model_dir_lower)    
    
# ----------------------------------------
    @app.route('/predict-keras', methods=['POST'])
    def predict_keras():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310','SDBB91','SDKC91','SDKC9','ROC','ATR54','ATR53','ATR52','ATR51','ATR5','ATR21','ATR2','RSI','STOK1', 'output', 'outputC', 'actual']
        df = pd.read_csv(csv_data, header=None, names=column_names)
        
        j_out = None
        
            
        if (df['RSI'][0] < 40):  
            df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)
            
            x_val = df.values
            x_val = x_val.reshape((1, 14, 1)) 
            predicts = 0
            for m in range(len(keras_lower_models)):
                predicts += keras_lower_models[m].predict(x_val)[0]
                #print(f"{df['RSI'][0]}    {m}  ")


            
            y_val = (predicts/len(keras_lower_models))[0]
            
            #if y_val < 0:
            #    y_val = 0
            
            print(f"{df['RSI'][0]}   {y_val}  ")
            
            out_data = {
            "agg_prediction" : float(y_val),
            "agg_weighted_prediction" : float(y_val),
            "all_predicts": [float(y_val)]
            }
            j_out = json.dumps(out_data, indent=4)
            
            
        elif (df['RSI'][0] > 60) :  
            df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)
            
            x_val = df.values
            x_val = x_val.reshape((1, 14, 1)) 
            predicts = 0
            for m in range(len(keras_upper_models)):
                predicts += keras_upper_models[m].predict(x_val)[0]
                #print(f"{df['RSI'][0]}    {m}  ")
            
            y_val = (predicts/len(keras_upper_models))[0]
            
            #if y_val > 0:
            #    y_val = 0
            
            print(f"{df['RSI'][0]}   {y_val}  ")
            
            
            
            out_data = {
            "agg_prediction" : float(y_val),
            "agg_weighted_prediction" : float(y_val),
            "all_predicts": [float(y_val)]
            }
            j_out = json.dumps(out_data, indent=4)
            
            
        return j_out


        
    return app

   
app = init_app()
    
if __name__ == '__main__':
    print("Starting Flask application.")
      
    
    app.run(
        debug=True, 
        use_reloader=False,
        port=9999, 
        host='0.0.0.0'
        )
    