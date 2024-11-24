import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
import warnings
from datetime import datetime


np.random.seed(42)
warnings.filterwarnings('ignore')

def download_and_prepare_data(ticker, start_date, end_date):

    raw_data = yf.download(ticker, start=start_date, end=end_date)
    data = raw_data
    
    with np.errstate(divide='ignore', invalid='ignore'):
        data['LogReturn'] = np.log(raw_data['Adj Close'] / raw_data['Adj Close'].shift(1))
        
    return data['LogReturn'].dropna(), raw_data



def combined_plot(data, result, k_regimes, start_date, end_date):
    
    print("Creating combined plot...")
    
    # Prepare data for plotting
    smoothed_probs = result.smoothed_marginal_probabilities
    smoothed_probs.columns = [f'Regime {i}' for i in range(k_regimes)]
    most_probable_regimes = smoothed_probs.idxmax(axis=1)
    regime_changes = most_probable_regimes != most_probable_regimes.shift(1)
    transition_points = most_probable_regimes[regime_changes]

    if start_date:
        start_date = pd.to_datetime(start_date)
        data = data[data.index >= start_date]
        transition_points = transition_points[transition_points.index >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        data = data[data.index <= end_date]
        transition_points = transition_points[transition_points.index <= end_date]
    
    price_data = data['Adj Close'].reindex(most_probable_regimes.index)
    regime_colors = {f'Regime {i}': plt.cm.tab10(i) for i in range(k_regimes)}

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(price_data, label="Adjusted Close Price", color='black', linewidth=1.5)

    for idx in transition_points.index:
        ax.axvline(x=idx, color='red', linestyle='--', alpha=0.8 )

    ax.set_title("Adjusted Close Price with Regime Overlay and Transition Points")
    ax.set_ylabel("Adjusted Close Price")
    ax.legend()

    ax.set_xlabel("Date")
    plt.tight_layout()
    plt.show()




# Main Workflow
if __name__ == "__main__":
    # Number of regimes (parameter-driven)
    k_regimes = 4

    em_iter = 200
    search_reps = 20
    tcker = '^GSPC'  # S&P 500 Index
    start_date = '2010-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    plot_start_date = '2024-01-01'

    returns, data = download_and_prepare_data(tcker, start_date, end_date)
    
    print(f"Fitting Markov Regime-Switching model with {k_regimes} regimes...")
    model = MarkovAutoregression(returns, k_regimes=k_regimes, trend='c', 
        order=2, switching_variance=True
    )
    
    result = model.fit(em_iter=em_iter, search_reps=search_reps, em_verbose=True)
    combined_plot(data, result, k_regimes, start_date=plot_start_date, end_date=end_date)
