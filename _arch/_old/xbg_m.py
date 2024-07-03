import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler



# Read the data into a DataFrame
df = pd.read_csv('data/buildSeqInd_Lucky13_5M_ALL.csv')

# Drop 'outputC' column
df = df.drop(columns=['outputC'])

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Create lag features
def create_lag_features(df, lag=1):
    lag_features = df.copy()
    for col in df.columns:
        for l in range(1, lag + 1):
            lag_features[f'{col}_lag_{l}'] = df[col].shift(l)
    lag_features = lag_features.dropna()
    return lag_features

lagged_data = create_lag_features(df, lag=1)
X = lagged_data.drop(columns=['output'])
y = lagged_data['output']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=75, learning_rate=0.05)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Test MSE:', mse)
