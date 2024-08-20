import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # For saving scaler and IsolationForest

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from ml_model.model_stats import gen_reg_stats_x 
tf.config.set_visible_devices([], 'GPU')

class TSMixer(tf.keras.Model):
    def __init__(self, input_dim, output_dim=1, hidden_units=128, num_layers=2, dropout_rate=0.1, **kwargs):
        super(TSMixer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers_list = []
        for _ in range(num_layers):
            self.layers_list.append(layers.Dense(hidden_units, activation='relu'))
            self.layers_list.append(layers.Dropout(dropout_rate))
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        outputs = self.output_layer(x)
        return outputs

    def get_config(self):
        config = super(TSMixer, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_units": self.hidden_units,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)




class TemporalFusionTransformer(tf.keras.Model):
    def __init__(self, input_dim, output_dim, hidden_units=128, num_heads=4, dropout_rate=0.1, **kwargs):
        super(TemporalFusionTransformer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.embedding = layers.Dense(hidden_units)
        self.dropout = layers.Dropout(dropout_rate)
        self.lstm = layers.LSTM(hidden_units, return_sequences=True)
        self.multi_head_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_units)
        self.dense1 = layers.Dense(hidden_units, activation='relu')
        self.dense2 = layers.Dense(output_dim)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dropout(x)
        
        # Reshape input to add sequence dimension
        x = tf.expand_dims(x, axis=1)  # Shape becomes (batch_size, 1, hidden_units)
        
        x = self.lstm(x)
        x = self.multi_head_attention(x, x)
        x = self.dense1(x)
        outputs = self.dense2(x[:, -1, :])  # Take the output from the last time step
        return outputs

    def get_config(self):
        config = super(TemporalFusionTransformer, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_units": self.hidden_units,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



@tf.keras.utils.register_keras_serializable()
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], 16))
    return z_mean + tf.keras.backend.exp(z_log_var) * epsilon


class Anomaly_Ensemble:
    def __init__(self, epochs=50, batch_size=32):
        
        self.input_dim = None
        self.epochs = epochs
        self.batch_size = batch_size
       
        self.lstm_model = None
        self.iso_forest_model = None
        self.tft_model = None
        self.ts_mixer_model = None
        self.scaler = None
        self.ref_model = None

        self.checkpoint_dir = 'checkpoints/'
        
        self.saved_lstm_model = os.path.join(self.checkpoint_dir, 'lstm_model.keras')
        self.saved_tft_model = os.path.join(self.checkpoint_dir, 'tft_model.keras')
        self.saved_ts_mixer_model = os.path.join(self.checkpoint_dir, 'ts_mixer_model.keras')
        self.saved_scaler = os.path.join(self.checkpoint_dir, 'scaler.pkl') 
        self.saved_iso_forest_model = os.path.join(self.checkpoint_dir, 'iso_forest.pkl') 
        self.saved_ref_model = os.path.join(self.checkpoint_dir, 'xgb_model.json')  # Path to save XGB model

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.ens_patience = 5
        print(f"Anomaly Ensemble Init")

    def build_isolation_forest(self):
        return IsolationForest(contamination=0.06, n_estimators=100, random_state=42)

    def build_tft_model(self, input_dim):
        model = TemporalFusionTransformer(input_dim=input_dim, output_dim=1)
        model.compile(optimizer='adam', loss='mse')
        return model

    def build_ts_mixer_model(self, input_dim):
        model = TSMixer(input_dim=input_dim, output_dim=1)
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_tft(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.ens_patience, restore_best_weights=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(self.saved_tft_model, save_best_only=True)
        
        self.tft_model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2, 
                           callbacks=[model_checkpoint, early_stopping])

    def train_ts_mixer(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.ens_patience, restore_best_weights=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(self.saved_ts_mixer_model, save_best_only=True)
        
        self.ts_mixer_model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2, 
                                callbacks=[model_checkpoint, early_stopping])

    def train_isolation_forest(self):
        self.iso_forest_model.fit(self.X_train)
        joblib.dump(self.iso_forest_model, self.saved_iso_forest_model )
        print(f"Train Forest")

    def train_baseline(self):
        ref_model = XGBRegressor()
        ref_model.fit(self.X_train, self.y_train)
        ref_model.save_model(self.saved_ref_model)
        self.ref_model = ref_model  
        print(f"Train Baseline")
        

    def training_setup(self, data_file):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        df = pd.read_csv(data_file)
        df = df.drop(columns=['outputC'])  # Drop the binary output if it exists
        features = df.drop(columns=['output']).values
        target = df['output'].values
    
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(features)
        joblib.dump(self.scaler, self.saved_scaler)

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)    
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train  
        self.y_test = y_test
        self.input_dim = X_train.shape[1]

        print(f"Anomaly Training Ensemble Setup")

    def test_setup(self, data_file):

        df = pd.read_csv(data_file)
        df = df.drop(columns=['outputC'])  # Drop the binary output if it exists
        features = df.drop(columns=['output']).values
        target = df['output'].values
        
        self.scaler = joblib.load(self.saved_scaler)
        features = self.scaler.fit_transform(features)
        self.X_test = features
        self.y_test = target
        self.input_dim = features.shape[1]

        print(f"Anomaly Test Setup")

    def build_ensemble_models(self):
        
        iso_forest_model = self.build_isolation_forest()
        tft_model = self.build_tft_model(self.input_dim)
        ts_mixer_model = self.build_ts_mixer_model(self.input_dim)
       
        self.iso_forest_model = iso_forest_model
        self.tft_model = tft_model
        self.ts_mixer_model = ts_mixer_model
        
        print(f"Anomaly Ensemble Build")
        return iso_forest_model, tft_model, ts_mixer_model

    def train_ensemble(self):
        self.train_isolation_forest()
        self.train_lstm()
        self.train_tft()
        self.train_ts_mixer()
        print(f"Anomaly Ensemble Train")


    def detect_anomalies_batch(self, threshold_percentile=95):
       
        iso_anomaly_score = self.iso_forest_model.decision_function(self.X_test)
        tft_prediction = self.tft_model.predict(self.X_test)
        tft_anomaly_score = np.abs(self.y_test - tft_prediction.flatten())
        
        ts_mixer_prediction = self.ts_mixer_model.predict(self.X_test)
        ts_mixer_anomaly_score = np.abs(self.y_test - ts_mixer_prediction.flatten())
        
        print(f"TFT Mean  {np.mean(tft_anomaly_score.flatten())}    ")
        print(f"TSM Mean  {np.mean(ts_mixer_anomaly_score.flatten())}  ")
        
        weighted_anomaly_scores = (
            0.7 * iso_anomaly_score +
            0.15 * tft_anomaly_score +
            0.15 * ts_mixer_anomaly_score
        )
       
        threshold = np.percentile(weighted_anomaly_scores, threshold_percentile)
        print(f"Test Threshold {threshold}   ")
        
        final_scores = weighted_anomaly_scores > threshold
        return final_scores



    def detect_anomalies_single(self, X):
        
        weighted_anomaly_scores = 0.0
        
        # from Test set  
        #TFT Mean  1.705082390692624
        #TSM Mean  1.6680918936122138
        #Test Threshold 1.5248480003779112
        
        #threshold = 1.5248480003779112
        threshold = 1.8
        
        ts_mixer_mean =   1.6680918936122138
        tft_mean = 1.705082390692624
        
        X_scaled = self.scaler.transform([X.values])
        iso_anomaly_score = self.iso_forest_model.decision_function(X_scaled)
        tft_prediction = self.tft_model.predict(X_scaled)[0]
        ts_mixer_prediction = self.ts_mixer_model.predict(X_scaled)[0]
            
        ts_mixer_anomaly_score = np.abs(ts_mixer_mean - ts_mixer_prediction.flatten()[0])
        tft_anomaly_score = np.abs(tft_mean - tft_prediction.flatten()[0])
        
        weighted_anomaly_scores = (
            0.7 * iso_anomaly_score +
            0.15 * tft_anomaly_score +
            0.15 * ts_mixer_anomaly_score
            )

        is_anomaly = weighted_anomaly_scores > threshold
        
        #print(f" ISO                          {iso_anomaly_score} ")
        #print(f" TFT        {tft_prediction}  {tft_anomaly_score}  ")
        #print(f" TSMIxer    {ts_mixer_prediction}  {ts_mixer_anomaly_score} ")
        print(f" Final Anomaly Score  {weighted_anomaly_scores}   {threshold}   {is_anomaly[0]} ")
        
        return is_anomaly[0]



    def anomaly_baseline(self, threshold_percentile=95):

        anomalies = self.detect_anomalies_batch(threshold_percentile)
        y_pred = self.ref_model.predict(self.X_test)
        correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(self.y_test, y_pred)

        X_test_filtered = self.X_test[~anomalies]
        y_test_filtered = self.y_test[~anomalies]
        filtered_predictions = self.ref_model.predict(X_test_filtered)

        print(" ")
        print(f"Detected {np.sum(anomalies)} anomalies out of {len(self.X_test)} samples.")
        print(" ")
        print(f"Test MSE: {mse}  Test MAE: {mae}  Test RMSE: {rmse}  R2: {r2}")
        print(f"Total Wins: {correct}  Total Losses: {total-correct}  Win Percentage: {perf:.2f}%")
        print(f"Number of Samples: {total}")
        
        correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(y_test_filtered, filtered_predictions)

        print(" ")
        print(f"Filtered Test MSE: {mse}   Test MAE: {mae}  Test RMSE: {rmse}  R2: {r2}")
        print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.2f}%")
        print(f"Number of Samples: {total}")
        print(" ")       
        
        
    def load_saved_ensemble(self):
        
        self.tft_model = tf.keras.models.load_model(self.saved_tft_model,
            custom_objects={'TemporalFusionTransformer': TemporalFusionTransformer})
        
        self.ts_mixer_model = tf.keras.models.load_model(self.saved_ts_mixer_model)
        self.iso_forest_model = joblib.load(self.saved_iso_forest_model)
       
        self.scaler = joblib.load(self.saved_scaler)
        xgb = XGBRegressor()
        xgb.load_model(self.saved_ref_model)
        self.ref_model = xgb
