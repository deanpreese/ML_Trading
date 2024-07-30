import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Define Composite Model Components
class PrebuiltTSMixerModel(tf.keras.Model):
    def __init__(self, input_dim):
        super(PrebuiltTSMixerModel, self).__init__()
        self.model = tf.keras.Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])

    def call(self, inputs):
        return self.model(inputs)

class PrebuiltTFTModel(tf.keras.Model):
    def __init__(self, input_dim):
        super(PrebuiltTFTModel, self).__init__()
        self.model = tf.keras.Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])

    def call(self, inputs):
        return self.model(inputs)

class PrebuiltTiDEModel(tf.keras.Model):
    def __init__(self, input_dim):
        super(PrebuiltTiDEModel, self).__init__()
        self.model = tf.keras.Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])

    def call(self, inputs):
        return self.model(inputs)

# Define Composite Model
class CompositeModel(tf.keras.Model):
    def __init__(self, input_dim):
        super(CompositeModel, self).__init__()
        self.ts_mixer = PrebuiltTSMixerModel(input_dim)
        self.tft = PrebuiltTFTModel(input_dim)
        self.tide = PrebuiltTiDEModel(input_dim)
        self.final_dense = Dense(1)

    def call(self, inputs):
        ts_mixer_output = self.ts_mixer(inputs)
        tft_output = self.tft(inputs)
        tide_output = self.tide(inputs)
        concatenated = tf.keras.layers.concatenate([ts_mixer_output, tft_output, tide_output])
        return self.final_dense(concatenated)

# Function to compute evaluation metrics
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    return rmse, r2, pearson_corr

# Function to train and evaluate XGBoost model
def train_evaluate_xgboost(X_train, X_test, y_train, y_test):
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)
    xgb_rmse, xgb_r2, xgb_pearson = compute_metrics(y_test, xgb_predictions)
    return xgb_rmse, xgb_r2, xgb_pearson

# Function to train and evaluate LightGBM model
def train_evaluate_lightgbm(X_train, X_test, y_train, y_test):
    lgb_model = lgb.LGBMRegressor()
    lgb_model.fit(X_train, y_train)
    lgb_predictions = lgb_model.predict(X_test)
    lgb_rmse, lgb_r2, lgb_pearson = compute_metrics(y_test, lgb_predictions)
    return lgb_rmse, lgb_r2, lgb_pearson

# Function to train and evaluate CatBoost model
def train_evaluate_catboost(X_train, X_test, y_train, y_test):
    cat_model = CatBoostRegressor(logging_level='Silent')
    cat_model.fit(X_train, y_train)
    cat_predictions = cat_model.predict(X_test)
    cat_rmse, cat_r2, cat_pearson = compute_metrics(y_test, cat_predictions)
    return cat_rmse, cat_r2, cat_pearson

# Function to train and evaluate XGBRFRegressor model
def train_evaluate_xgbrf(X_train, X_test, y_train, y_test):
    xgbrf_model = xgb.XGBRFRegressor(objective='reg:squarederror')
    xgbrf_model.fit(X_train, y_train)
    xgbrf_predictions = xgbrf_model.predict(X_test)
    xgbrf_rmse, xgbrf_r2, xgbrf_pearson = compute_metrics(y_test, xgbrf_predictions)
    return xgbrf_rmse, xgbrf_r2, xgbrf_pearson

# Function to train and evaluate Composite model
def train_evaluate_composite(X_train, X_test, y_train, y_test, input_dim):
    composite_model = CompositeModel(input_dim)
    composite_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_composite_model.keras', save_best_only=True, monitor='val_loss')

    composite_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])
    
    composite_model.load_weights('best_composite_model.keras')
    composite_predictions = composite_model.predict(X_test)
    composite_rmse, composite_r2, composite_pearson = compute_metrics(y_test, composite_predictions)
    return composite_rmse, composite_r2, composite_pearson

def main():
    datafile = [ 
        'data/buildSeqInd_Lucky13_5M_3070.csv',   #0
        'data/buildSeqInd_Lucky13_5M_ALL.csv',  #1
        'data/buildSeqInd_Lucky13_F.csv',  #2
        'data/buildSeqInd_Lucky13_D.csv',  #3
        'data/buildSeqInd_Lucky13_F_3070.csv',  #4
        'data/Expanded_Lucky13_070.csv',  #5
        'data/ndata_3070.csv', #6
        'data/ndata_3070_alt.csv', #7
        'data/ym_ndata_3070_alt.csv', #8
    ]

    file_loaded = pd.read_csv(datafile[0])
    X = file_loaded.drop(columns=['output', 'outputC'])
    y = file_loaded['output'].values
        
    num_features = len(X.columns)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate models
    xgb_metrics = train_evaluate_xgboost(X_train, X_test, y_train, y_test)
    lgb_metrics = train_evaluate_lightgbm(X_train, X_test, y_train, y_test)
    cat_metrics = train_evaluate_catboost(X_train, X_test, y_train, y_test)
    xgbrf_metrics = train_evaluate_xgbrf(X_train, X_test, y_train, y_test)
    composite_metrics = train_evaluate_composite(X_train, X_test, y_train, y_test, num_features)
    
    print(f'XGBoost Metrics: RMSE: {xgb_metrics[0]}, R2: {xgb_metrics[1]}, Pearson: {xgb_metrics[2]}')
    print(f'LightGBM Metrics: RMSE: {lgb_metrics[0]}, R2: {lgb_metrics[1]}, Pearson: {lgb_metrics[2]}')
    print(f'CatBoost Metrics: RMSE: {cat_metrics[0]}, R2: {cat_metrics[1]}, Pearson: {cat_metrics[2]}')
    print(f'XGBRFRegressor Metrics: RMSE: {xgbrf_metrics[0]}, R2: {xgbrf_metrics[1]}, Pearson: {xgbrf_metrics[2]}')
    print(f'Composite Model Metrics: RMSE: {composite_metrics[0]}, R2: {composite_metrics[1]}, Pearson: {composite_metrics[2]}')

if __name__ == "__main__":
    main()



    """
    XGBoost Metrics: RMSE: 3.153298615724688, R2: 0.41745894956140106, Pearson: 0.6512327637859829
LightGBM Metrics: RMSE: 3.0139309603377327, R2: 0.46781462707187893, Pearson: 0.6840028407024694
CatBoost Metrics: RMSE: 3.075953071126159, R2: 0.4456861309778014, Pearson: 0.6683683273300914
XGBRFRegressor Metrics: RMSE: 3.0646974531400275, R2: 0.44973543197710586, Pearson: 0.6714257575827112
Composite Model Metrics: RMSE: 3.1789355388332896, R2: 0.4079481024355691, Pearson: 0.6387103530019005
    
    
    
    
    """