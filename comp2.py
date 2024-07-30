import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
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
    def __init__(self, input_dim, xgb_model, lgb_model, cat_model):
        super(CompositeModel, self).__init__()
        self.ts_mixer = PrebuiltTSMixerModel(input_dim)
        self.tft = PrebuiltTFTModel(input_dim)
        self.tide = PrebuiltTiDEModel(input_dim)
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.cat_model = cat_model
        self.final_dense = Dense(1)

    def call(self, inputs):
        ts_mixer_output = self.ts_mixer(inputs)
        tft_output = self.tft(inputs)
        tide_output = self.tide(inputs)

        # Convert inputs to NumPy arrays for XGBoost, LightGBM, and CatBoost
        inputs_np = inputs.numpy() if isinstance(inputs, tf.Tensor) else inputs

        xgb_output = tf.convert_to_tensor(self.xgb_model.predict(inputs_np).reshape(-1, 1), dtype=tf.float32)
        lgb_output = tf.convert_to_tensor(self.lgb_model.predict(inputs_np).reshape(-1, 1), dtype=tf.float32)
        cat_output = tf.convert_to_tensor(self.cat_model.predict(inputs_np).reshape(-1, 1), dtype=tf.float32)

        concatenated = Concatenate()([ts_mixer_output, tft_output, tide_output, xgb_output, lgb_output, cat_output])
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
    return xgb_model, xgb_rmse, xgb_r2, xgb_pearson

# Function to train and evaluate LightGBM model
def train_evaluate_lightgbm(X_train, X_test, y_train, y_test):
    lgb_model = lgb.LGBMRegressor()
    lgb_model.fit(X_train, y_train)
    lgb_predictions = lgb_model.predict(X_test)
    lgb_rmse, lgb_r2, lgb_pearson = compute_metrics(y_test, lgb_predictions)
    return lgb_model, lgb_rmse, lgb_r2, lgb_pearson

# Function to train and evaluate CatBoost model
def train_evaluate_catboost(X_train, X_test, y_train, y_test):
    cat_model = CatBoostRegressor(logging_level='Silent')
    cat_model.fit(X_train, y_train)
    cat_predictions = cat_model.predict(X_test)
    cat_rmse, cat_r2, cat_pearson = compute_metrics(y_test, cat_predictions)
    return cat_model, cat_rmse, cat_r2, cat_pearson

# Function to train and evaluate XGBRFRegressor model
def train_evaluate_xgbrf(X_train, X_test, y_train, y_test):
    xgbrf_model = xgb.XGBRFRegressor(objective='reg:squarederror')
    xgbrf_model.fit(X_train, y_train)
    xgbrf_predictions = xgbrf_model.predict(X_test)
    xgbrf_rmse, xgbrf_r2, xgbrf_pearson = compute_metrics(y_test, xgbrf_predictions)
    return xgbrf_model, xgbrf_rmse, xgbrf_r2, xgbrf_pearson

# Custom training loop for CompositeModel
def train_evaluate_composite(X_train, X_test, y_train, y_test, input_dim):
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
    lgb_model = lgb.LGBMRegressor()
    cat_model = CatBoostRegressor(logging_level='Silent')

    # Fit the models
    xgb_model.fit(X_train, y_train)
    lgb_model.fit(X_train, y_train)
    cat_model.fit(X_train, y_train)

    composite_model = CompositeModel(input_dim, xgb_model, lgb_model, cat_model)
    optimizer = Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Custom training loop
    for epoch in range(50):
        print(f"Epoch {epoch+1}/50")
        with tf.GradientTape() as tape:
            predictions = composite_model(X_train, training=True)
            loss = loss_fn(y_train, predictions)
        gradients = tape.gradient(loss, composite_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, composite_model.trainable_variables))
        
        val_predictions = composite_model(X_test, training=False)
        val_loss = loss_fn(y_test, val_predictions)
        print(f"Validation loss: {val_loss.numpy()}")

        if epoch % 10 == 0:
            composite_model.save_weights('best_composite_model.weights.h5')

    composite_model.load_weights('best_composite_model.weights.h5')
    composite_predictions = composite_model(X_test, training=False)
    composite_rmse, composite_r2, composite_pearson = compute_metrics(y_test, composite_predictions.numpy())
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
    xgb_model, xgb_rmse, xgb_r2, xgb_pearson = train_evaluate_xgboost(X_train, X_test, y_train, y_test)
    lgb_model, lgb_rmse, lgb_r2, lgb_pearson = train_evaluate_lightgbm(X_train, X_test, y_train, y_test)
    cat_model, cat_rmse, cat_r2, cat_pearson = train_evaluate_catboost(X_train, X_test, y_train, y_test)
    xgbrf_model, xgbrf_rmse, xgbrf_r2, xgbrf_pearson = train_evaluate_xgbrf(X_train, X_test, y_train, y_test)
    composite_rmse, composite_r2, composite_pearson = train_evaluate_composite(X_train, X_test, y_train, y_test, num_features)
    
    print(f'XGBoost Metrics: RMSE: {xgb_rmse}, R2: {xgb_r2}, Pearson: {xgb_pearson}')
    print(f'LightGBM Metrics: RMSE: {lgb_rmse}, R2: {lgb_r2}, Pearson: {lgb_pearson}')
    print(f'CatBoost Metrics: RMSE: {cat_rmse}, R2: {cat_r2}, Pearson: {cat_pearson}')
    print(f'XGBRFRegressor Metrics: RMSE: {xgbrf_rmse}, R2: {xgbrf_r2}, Pearson: {xgbrf_pearson}')
    print(f'Composite Model Metrics: RMSE: {composite_rmse}, R2: {composite_r2}, Pearson: {composite_pearson}')

if __name__ == "__main__":
    main()
