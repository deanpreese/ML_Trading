from flask import Flask, request, jsonify
import pandas as pd
from io import BytesIO
import json
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

model_loader = ModelLoader()
models_one = []
models_two = []
models_three = []
models_four = []
models_five = []

# ----------------------------------------
def LoadModels(group_id, experiment_id, num_models):
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




# ----------------------------------------
def init_app():
    app = Flask(__name__)

    with app.app_context():
        
        #models_one = LoadModels(0, ["385"], 1)
        #models_two = LoadModels(0, ["387"], 1)
        #models_three = LoadModels(0, ["389"], 1)
        #models_four = LoadModels(0, ["393"], 1)
        #models_five = LoadModels(0, ["453"], 1)
        
        models_one = LoadModels(0, ["469"], 1)
        models_two = LoadModels(0, ["470"], 1)
        models_three = LoadModels(0, ["473"], 1)
        models_four = LoadModels(0, ["474"], 1)
       
        
       
        
    # ----------------------------------------
    @app.route('/predict-one', methods=['POST'])
    def predict_one():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310','SDBB91','SDKC91','SDKC9','ROC','ATR54','ATR53','ATR52','ATR51','ATR5','ATR21','ATR2','RSI','STOK1', 'output', 'outputC', 'actual']
        df = pd.read_csv(csv_data, header=None, names=column_names)
        
        #j_out = None
        #if (df['RSI'][0] <  25)  |  (df['RSI'][0] > 75):
        #    out_data = gen_zero_predictions()
        #    j_out = json.dumps(out_data, indent=4)
        #else:
        df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)
        out_data = get_model_predictions(df, models_one)
        j_out = json.dumps(out_data, indent=4)
        #print(jd)
        return j_out

    # ----------------------------------------
    @app.route('/predict-two', methods=['POST'])
    def predict_two():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310','SDBB91','SDKC91','SDKC9','ROC','ATR54','ATR53','ATR52','ATR51','ATR5','ATR21','ATR2','RSI','STOK1', 'output', 'outputC', 'actual']
        df = pd.read_csv(csv_data, header=None, names=column_names)
        
        #j_out = None
        #if (df['RSI'][0] <  25)  |  (df['RSI'][0] > 75):
        #    out_data = gen_zero_predictions()
        #    j_out = json.dumps(out_data, indent=4)
        #else:
        df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)
        out_data = get_model_predictions(df, models_two)
        j_out = json.dumps(out_data, indent=4)
        #print(jd)
        return j_out
        
    # ----------------------------------------
    @app.route('/predict-three', methods=['POST'])
    def predict_three():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310','SDBB91','SDKC91','SDKC9','ROC','ATR54','ATR53','ATR52','ATR51','ATR5','ATR21','ATR2','RSI','STOK1', 'output', 'outputC', 'actual']
        df = pd.read_csv(csv_data, header=None, names=column_names)
        
        j_out = None
        
        
        #if (df['RSI'][0] < 25)  |  (df['RSI'][0] > 75):
        #    out_data = gen_zero_predictions()
        #    j_out = json.dumps(out_data, indent=4)
        #else:
        df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)
        out_data = get_model_predictions(df, models_three)
        j_out = json.dumps(out_data, indent=4)
        #print(jd)
        return j_out


 # ----------------------------------------
    @app.route('/predict-four', methods=['POST'])
    def predict_four():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310','SDBB91','SDKC91','SDKC9','ROC','ATR54','ATR53','ATR52','ATR51','ATR5','ATR21','ATR2','RSI','STOK1', 'output', 'outputC', 'actual']
        df = pd.read_csv(csv_data, header=None, names=column_names)
        
        j_out = None
        
        #if (df['RSI'][0] < 25)  |  (df['RSI'][0] > 75):
        #    out_data = gen_zero_predictions()
        #    j_out = json.dumps(out_data, indent=4)
        #else:
        df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)
        out_data = get_model_predictions(df, models_four)
        j_out = json.dumps(out_data, indent=4)
        #print(jd)
        return j_out



 # ----------------------------------------
    @app.route('/predict-five', methods=['POST'])
    def predict_five():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310','SDBB91','SDKC91','SDKC9','ROC','ATR54','ATR53','ATR52','ATR51','ATR5','ATR21','ATR2','RSI','STOK1', 'output', 'outputC', 'actual']
        df = pd.read_csv(csv_data, header=None, names=column_names)
        
        j_out = None
        
        #if (df['RSI'][0] < 25)  |  (df['RSI'][0] > 75):
        #    out_data = gen_zero_predictions()
        #    j_out = json.dumps(out_data, indent=4)
        #else:
        df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)
        out_data = get_model_predictions(df, models_five)
        j_out = json.dumps(out_data, indent=4)
        #print(jd)
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
    