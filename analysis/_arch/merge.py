import pandas as pd
from datetime import datetime, timedelta

data1 = 'y_1.csv'  #indicator data      
data2 = 'y_2.csv'  #NT Trade Data

# Define the correct headers for csv1 and csv2
ind_header = [
    'tradedatetime', 'qty', 'signal', 'tdate', 'ttime', 'directon', 
    'entryprice1', 'q2', 'fulldt', 'sdlr310', 'sdbb91', 'sdkc91', 
    'sdkc9', 'roc', 'atr34', 'atr32', 'atr31', 'atr3', 'atr21', 'atr2', 
    'RSI', 'stok1', 'output', 'outputC', 'cp'
]

trade_header = [
    'tradenumber', 'instrument', 'account', 'strategy', 'marketpos', 'qty', 
    'entryprice', 'exitprice', 'entrytime', 'exittime', 'entryname', 'exitname', 
    'profit', 'cumnetprofit', 'commission', 'mae', 'mfe', 'etd', 'bars', 'xxx'
]

# Read the CSV files without headers
ind_data = pd.read_csv(data1, header=None)
trade_data = pd.read_csv(data2, skiprows=0)
ind_data.columns = ind_header
trade_data.columns = trade_header

ind_data['tradedatetime'] = pd.to_datetime(ind_data['tradedatetime'])
ind_data['tdate'] = pd.to_datetime(ind_data['tdate'])
ind_data['fulldt'] = pd.to_datetime(ind_data['fulldt'])

ind_data['tradedatetime'] = ind_data['tradedatetime'] + timedelta(minutes=1)

trade_data['entrytime'] = pd.to_datetime(trade_data['entrytime'])
trade_data['exittime'] = pd.to_datetime(trade_data['exittime'])

data1 = pd.merge(ind_data, trade_data, how='inner', left_on='tradedatetime', right_on='entrytime')

# Select and rename columns to match the output structure
data1 = data1[['tradedatetime', 'signal', 'directon', 
               'sdlr310', 'sdbb91', 'sdkc91', 'sdkc9', 'roc', 'atr34', 'atr32', 'atr31', 
               'atr3', 'atr21', 'atr2', 'RSI', 'stok1', 'entrytime', 'exittime', 'entryprice', 
               'exitprice', 'profit', 'mae', 'mfe' ]]

# Rename columns to match output format
data1.columns = ['tradedatetime', 'signal_v', 'directon', 
                 'sdlr310', 'sdbb91', 'sdkc91', 'sdkc9', 'roc', 'atr34', 'atr32', 'atr31', 
                 'atr3', 'atr21', 'atr2', 'rsi', 'stok1', 'entrytime', 'exittime', 'entryprice', 
                 'exitprice', 'profit', 'mae', 'mfe']

# Save the result to a new CSV file
data1.to_csv('data1.csv', index=False)