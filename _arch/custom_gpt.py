from pandas import date_range
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from lightgbm import LGBMRegressor
from darts.models import NHiTSModel
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

def create_lstm_model(input_dim):
    model = Sequential([
        Input(shape=(1, input_dim)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return model


# Function to create sequences
def create_sequences(data_in, seq_length_in):
    print("create_sequences ")
    xs = [data_in.iloc[i:i + seq_length_in, :-1].values for i in range(len(data_in) - seq_length_in)]
    ys = data_in.iloc[seq_length_in:].values
    xr = [data_in.iloc[i,].values for i in range(len(data_in) - seq_length_in)]
    
    return np.array(xs), np.array(ys), np.array(xr)

def main():
    
    sequence_length = 25
    test_split = 0.85
    
    data = pd.read_csv('data/buildSeqInd_Lucky13_5M_ALL.csv')
    data = data.drop(columns=['outputC'])
    series = TimeSeries.from_dataframe(data, value_cols='output').astype(np.float32)
    train_data, test_data = series.split_after(test_split)
    
    print(" Train Data ")
    print(train_data.pd_dataframe().shape)
    print(train_data.pd_dataframe())
    print("  ")
    
    scaler = Scaler()
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.fit_transform(test_data)
    
    features_cols = list(train_data.columns)  # Get all column names
    X_train = scaled_train_data[features_cols]  # Select features using column names
    y_train = scaled_train_data["output"]  # Select target variable column by name
    X_test = scaled_test_data[features_cols]  # Select features using column names
    y_test = scaled_test_data["output"]  # Select target variable column by name
        
    print(" Scaled ")
    print(X_train.pd_dataframe().shape)
    print(X_train.pd_dataframe())
    print(X_test.pd_dataframe().shape)
    print(X_train.pd_dataframe().shape)
    print(y_test.pd_dataframe().shape)
    print("  ")
    
    X_train_lstm, y_train_lstm, xr_train = create_sequences(train_data.pd_dataframe(), sequence_length)
    X_test_lstm, y_test_lstm, xr_test = create_sequences(test_data.pd_dataframe(), sequence_length)

    print(" Seq ")
    print(X_train_lstm.shape)
    print(xr_train.shape)
    print(y_train_lstm.shape)

    print(" Seq Removed")
    print(X_test_lstm.shape)
    print(xr_test.shape)    
    print(xr_test)
    
    print(y_test_lstm.shape)    
    print(y_test_lstm)    
    

    exit()
    lstm_model = create_lstm_model(X_train.shape[1])
    lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.1)

    lgbm_model = LGBMRegressor(n_estimators=100)
    lgbm_model.fit(X_train, y_train)

    y_train_ts = TimeSeries.from_dataframe(pd.DataFrame(data={'value': y_train}, index=train_dates))
    n_hits_model = NHiTSModel(input_chunk_length=15, output_chunk_length=5, n_epochs=100)
    n_hits_model.fit(series=y_train_ts)

    def ensemble_predictions(X, X_dates):
        X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
        lstm_pred = lstm_model.predict(X_lstm).flatten()
        lgbm_pred = lgbm_model.predict(X)
        X_ts = TimeSeries.from_dataframe(pd.DataFrame(data={'value': np.zeros_like(X[:, 0])}, index=X_dates))
        n_hits_pred = n_hits_model.predict(n=len(X), series=X_ts).values().flatten()
        return np.mean([lstm_pred, lgbm_pred, n_hits_pred], axis=0)

    predictions = ensemble_predictions(X_test, test_dates)
    mse = mean_squared_error(y_test, predictions)
    print(f'Ensemble MSE: {mse}')

if __name__ == "__main__":
    main()
