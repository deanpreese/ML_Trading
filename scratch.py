import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from datetime import datetime
import warnings
import mplfinance as mpf

# Ticker and date range
ticker = '^GSPC'  # S&P 500 Index
start_date = '2010-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

# Fetch the data
ydata = yf.download(ticker, start=start_date, end=end_date)

data = pd.DataFrame()
data['Open'] = ydata['Open']
data['High'] = ydata['High']
data['Low'] = ydata['Low']
data['Close'] = ydata['Close']

print(data.head())

# Plot using mplfinance
mpf.plot(data, type='candle', volume=False, style='yahoo')
