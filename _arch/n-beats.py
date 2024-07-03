import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from sklearn.model_selection import train_test_split


# Read the data into a DataFrame
df = pd.read_csv('data/13_10K.csv')

# Preprocess the data
scaler = Scaler()
series = TimeSeries.from_dataframe(df, value_cols=['SDLR310','SDBB91','SDKC91','SDKC9','ROC','ATR34','ATR32','ATR31','ATR3','ATR21','ATR2','RSI','STOK1','output', 'outputC'])
series = scaler.fit_transform(series)

# Split the data into training and validation sets
train, val = series.split_before(0.8)

# Define and train the N-BEATS model
model = NBEATSModel(input_chunk_length=2, output_chunk_length=1, n_epochs=100, random_state=42)
model.fit(train, val_series=val)

# Make predictions
pred = model.predict(n=10, series=train)

# Print predictions
print(pred)
