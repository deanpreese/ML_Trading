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

from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor
from lightgbm  import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from ml_model.model_stats import gen_reg_stats_x 

tf.config.set_visible_devices([], 'GPU')

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
        self.vae_model = None
        self.lstm_model = None
        self.iso_forest_model = None
        self.scaler = None
        self.ref_model = None

        self.checkpoint_dir = 'checkpoints/'
        
        self.saved_vae_model = os.path.join(self.checkpoint_dir, 'vae_model.keras')
        self.saved_lstm_model = os.path.join(self.checkpoint_dir, 'lstm_model.keras')
        self.saved_scaler = os.path.join(self.checkpoint_dir, 'scaler.pkl') 
        self.saved_iso_forest_model = os.path.join(self.checkpoint_dir, 'iso_forest.pkl') 
        self.saved_ref_model = os.path.join(self.checkpoint_dir, 'xgb_model.json')  # Path to save XGB model

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.ens_patience = 5
        
        print(f"Anomaly Ensemble Init")


    def build_vae(self, input_dim):
        
        inputs = layers.Input(shape=(input_dim,))
        h = layers.Dense(64, activation='relu')(inputs)
        h = layers.Dense(64, activation='relu')(h)
        z_mean = layers.Dense(16)(h)
        z_log_var = layers.Dense(16)(h)

        z = layers.Lambda(sampling)([z_mean, z_log_var])

        decoder_h = layers.Dense(32, activation='relu')
        decoder_mean = layers.Dense(input_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        outputs = decoder_mean(h_decoded)
        vae = models.Model(inputs, outputs)
        vae.compile(optimizer='adam', loss='mse')
        return vae

    def build_isolation_forest(self):
        return IsolationForest(contamination=0.06, n_estimators=100, random_state=42)

    def build_lstm_model(self, input_dim):
        
        inputs = layers.Input(shape=(None, input_dim))
        x = layers.LSTM(128, return_sequences=True)(inputs)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LSTM(16, return_sequences=True)(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LSTM(32)(x)
        x = layers.Dense(1)(x)

        model = models.Model(inputs, x)
        model.compile(optimizer='adam', loss='mse')
        return model


    def train_vae(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.ens_patience, restore_best_weights=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(self.saved_vae_model , save_best_only=True)
        
        self.vae_model.fit(self.X_train, self.X_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2, 
                callbacks=[model_checkpoint, early_stopping])

    def train_lstm(self):
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.ens_patience, restore_best_weights=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(self.saved_lstm_model,  save_best_only=True)
        
        X_train_lstm = np.expand_dims(self.X_train, axis=1)
        
        self.lstm_model.fit(X_train_lstm, self.y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2, 
                            callbacks=[model_checkpoint, early_stopping])
        
        #X_train_lstm = self.X_train.reshape(self.X_train.shape[0], 1, self.X_train.shape[1])
        #self.lstm_model.fit(X_train_lstm, self.y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2, 
        #        callbacks=[model_checkpoint, early_stopping])

    def train_isolation_forest(self):
        self.iso_forest_model.fit(self.X_train)
        joblib.dump(self.iso_forest_model, self.saved_iso_forest_model )
        print(f"Train Forest")


    def train_baseline(self):
        ref_model = XGBRegressor()
        ref_model.fit(self.X_train,self.y_train)
        ref_model.save_model(self.saved_ref_model)
        self.ref_model = ref_model  
        print(f"Train Baseline")
        

    def training_setup(self, data_file):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        df = pd.read_csv(data_file)
        df = df.drop(columns=['outputC'])  # Drop the binary output if it exists
        features = df.drop(columns=['output']).values
        target = df['output'].values
    
        self.scaler = joblib.load(self.saved_scaler)
        features = self.scaler.fit_transform(features)

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

        print(f"Anomaly Training Ensemble Setup")


    def build_ensemble_models(self):
        vae_model = self.build_vae(self.input_dim)
        iso_forest_model = self.build_isolation_forest()
        lstm_model = self.build_lstm_model(self.input_dim)
        self.vae_model = vae_model
        self.iso_forest_model = iso_forest_model
        self.lstm_model = lstm_model
        
        print(f"Anomaly Ensemble Build")
        return vae_model, lstm_model, iso_forest_model

    def train_ensemble(self):
        self.train_isolation_forest()
        self.train_lstm()
        self.train_vae()
        print(f"Anomaly Ensemble Train")


    def detect_anomalies_batch(self, threshold_percentile=95):
        
        #vae_reconstruction = self.vae_model.predict(self.X_test)
        #vae_anomaly_score = np.mean(np.square(self.X_test - vae_reconstruction), axis=1)
        
        
        iso_anomaly_score = -self.iso_forest_model.decision_function(self.X_test)
        #lstm_predictions = self.lstm_model.predict(self.X_test.reshape(self.X_test.shape[0], 1, self.X_test.shape[1]))
        #lstm_anomaly_score = np.abs(self.y_test - lstm_predictions.flatten())
        
        #weighted_anomaly_scores = 0.0 * vae_anomaly_score + 1.0 * iso_anomaly_score + 0.0 * lstm_anomaly_score
        weighted_anomaly_scores = iso_anomaly_score 
        
        threshold = np.percentile(weighted_anomaly_scores, threshold_percentile)
        final_scores = weighted_anomaly_scores > threshold
        return final_scores

   
    def detect_anomalies_single(self, X):
        
        if isinstance(X, pd.Series):
            X = X.values
        
        X_scaled = self.scaler.transform([X])  
        X_window = np.expand_dims(X_scaled, axis=1)
        #vae_reconstruction = self.vae_model.predict(X_scaled)
        #vae_anomaly_score = np.mean(np.square(X_scaled - vae_reconstruction), axis=1)
        
        iso_anomaly_score = -self.iso_forest_model.decision_function(X_scaled)
        #lstm_prediction = self.lstm_model.predict(X_scaled.reshape(1, 1, len(X)))
        #lstm_anomaly_score = np.abs(X[-1] - lstm_prediction.flatten()[0])

        #weighted_anomaly_score = 0.0 * vae_anomaly_score + 1.0 * iso_anomaly_score + 0.0 * lstm_anomaly_score
        weighted_anomaly_score =  iso_anomaly_score 
        
        threshold = np.percentile(weighted_anomaly_score, 94)  
        is_anomaly = weighted_anomaly_score > threshold
        
        return is_anomaly
       
       
       
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
        print(f"Test MSE: {mse}   Test MAE: {mae}  Test RMSE: {rmse}  R2: {r2}")
        print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.2f}%")
        print(f"Number of Samples: {total}")
        print(" ")       
       

    def load_saved_ensemble(self):
        self.vae_model = tf.keras.models.load_model(self.saved_vae_model, custom_objects={'sampling': sampling})
        self.lstm_model = tf.keras.models.load_model(self.saved_lstm_model)
        self.iso_forest_model = joblib.load(self.saved_iso_forest_model)
        self.scaler = joblib.load(self.saved_scaler)
        xgb = XGBRegressor()
        xgb.load_model(self.saved_ref_model)
        self.ref_model = xgb