from flask import Flask, request, jsonify
import pandas as pd
from io import BytesIO
import json
import numpy as np
from datetime import datetime, timedelta

from strategy.model_loader import ModelLoader
from models.ts_mixer_model import TSMixerModel
from models.cnn_lstm_model import CNN_LSTM
from models.kan_mixer_model import KANMixerModel
from models.anom_ensemble_model import Anomaly_Ensemble 

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

ts_mixer = TSMixerModel(epochs=100, batch_size=32)
cnn_model = CNN_LSTM()
kan_mixer = KANMixerModel(epochs=100, batch_size=32)
anom_ens = Anomaly_Ensemble(epochs=75, batch_size=32)

last_anom_timestamp = None
last_anom_value = False

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


def get_anomaly_score(X):
    return anom_ens.detect_anomalies_single(X.iloc[0])

def load_other_models():
    ts_mixer.load_saved_model()
    cnn_model.load_saved_model()
    kan_mixer.load_saved_model()
    anom_ens.load_saved_ensemble()

# ----------------------------------------
def init_app():
    app = Flask(__name__)

    with app.app_context():
        models_one = LoadModels(0, ["251"], 1)
        models_two = LoadModels(0, ["253"], 1)
        models_three = LoadModels(0, ["257"], 1)
        load_other_models()


    # ----------------------------------------
    @app.route('/predict-one', methods=['POST'])
    def predict_one():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310','SDBB91','SDKC91','SDKC9','ROC','ATR54','ATR53','ATR52','ATR51','ATR5','ATR21','ATR2','RSI','STOK1', 'output', 'outputC', 'actual']
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)
        out_data = get_model_predictions(data_df, models_one)
        jd = json.dumps(out_data, indent=4)
        #print(jd)
        return jd

    # ----------------------------------------
    @app.route('/predict-two', methods=['POST'])
    def predict_two():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310','SDBB91','SDKC91','SDKC9','ROC','ATR54','ATR53','ATR52','ATR51','ATR5','ATR21','ATR2','RSI','STOK1', 'output', 'outputC', 'actual']
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)
        out_data = get_model_predictions(data_df, models_two)
        jd = json.dumps(out_data, indent=4)
        #print(jd)
        return jd
        
    # ----------------------------------------
    @app.route('/predict-three', methods=['POST'])
    def predict_three():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310','SDBB91','SDKC91','SDKC9','ROC','ATR54','ATR53','ATR52','ATR51','ATR5','ATR21','ATR2','RSI','STOK1', 'output', 'outputC', 'actual']
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)
        out_data = get_model_predictions(data_df, models_three)
        jd = json.dumps(out_data, indent=4)
        #print(jd)
        return jd
    
    # ----------------------------------------
    @app.route('/predict-kan', methods=['POST'])
    def predict_kan():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310','SDBB91','SDKC91','SDKC9','ROC','ATR54','ATR53','ATR52','ATR51','ATR5','ATR21','ATR2','RSI','STOK1', 'output', 'outputC', 'actual']
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)
        
        X = data_df.values
        
        x_val = X.reshape((1, 14, 1)) 
        y_val = kan_mixer.model.predict(x_val)
        
        predicts = [y_val[0][0]]
        
        out_data = {
            "agg_prediction" : round(y_val[0][0],6),
            "agg_weighted_prediction" : round(y_val[0][0],6),
            "all_predicts": predicts
        }
        
        out_data = format_json(out_data)
        jd = json.dumps(out_data, indent=4)
        print(jd)
        return jd

   # ----------------------------------------
    @app.route('/predict-cnn', methods=['POST'])
    def predict_cnn():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310','SDBB91','SDKC91','SDKC9','ROC','ATR54','ATR53','ATR52','ATR51','ATR5','ATR21','ATR2','RSI','STOK1', 'output', 'outputC', 'actual']
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)
        
        X = data_df.values
        
        x_val = X.reshape((1, 14, 1)) 
        y_val = cnn_model.model.predict(x_val)
        predicts = [y_val[0][0]]
        
        out_data = {
            "agg_prediction" : round(y_val[0][0],6),
            "agg_weighted_prediction" : round(y_val[0][0],6),
            "all_predicts": predicts
        }
        
        out_data = format_json(out_data)
        jd = json.dumps(out_data, indent=4)
        print(jd)
        return jd


   # ----------------------------------------
    @app.route('/predict-tsm', methods=['POST'])
    def predict_tsm():
        
        csv_data = BytesIO(request.data)
        column_names = ['time', 'SDLR310','SDBB91','SDKC91','SDKC9','ROC','ATR54','ATR53','ATR52','ATR51','ATR5','ATR21','ATR2','RSI','STOK1', 'output', 'outputC', 'actual']
        data_df = pd.read_csv(csv_data, header=None, names=column_names)
        data_df.drop(columns=['time', 'actual', 'output', 'outputC'], inplace=True)
        X = data_df    
        
        #aX = data_df
        #anom_score = get_anomaly_score(aX)
        anom_score = False
        
        if anom_score == False:
            X_scaled = ts_mixer.saved_scaler.transform(X)
            X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # [batch_size, seq_length, num_features]
            y_val = ts_mixer.model.predict(X_scaled)
            y_raw = y_val[0][0]
        else:
            y_raw = 0.0            
            
        predicts = [y_raw]
        out_data = {
            "agg_prediction" : round(y_raw,6),
            "agg_weighted_prediction" : round(y_raw,6),
            "all_predicts": predicts
        }
        
        out_data = format_json(out_data)
        jd = json.dumps(out_data, indent=4)        
        jd = json.dumps(out_data, indent=4)
        print(jd)
        return jd




        
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
    