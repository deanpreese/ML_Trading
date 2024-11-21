import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from datetime import datetime
import warnings

def download_and_prepare_data(ticker, start_date, end_date, data_col):
    """
    Download historical price data for a given ticker.
    """
    ydata = yf.download(ticker, start=start_date, end=end_date)
    
    data = ydata
    raw_data = pd.DataFrame()
    raw_data['Open'] = ydata['Open']
    raw_data['High'] = ydata['High']
    raw_data['Low'] = ydata['Low']
    raw_data['Close'] = ydata['Close']
    raw_data['Adj Close'] = ydata['Adj Close']
    
    data['LogReturn'] = np.log(raw_data[data_col] / raw_data[data_col].shift(1))
    return data['LogReturn'].dropna(), raw_data

def fit_hmm(returns, n_components, n_iter=100):
    """
    Fit a Gaussian Hidden Markov Model (HMM) to the returns data.
    """
    print(f"Fitting Gaussian HMM with {n_components} regimes...")
    # Normalize the data for better convergence
    returns_normalized = (returns - returns.mean()) / returns.std()
    model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=n_iter, random_state=42, verbose=2)
    model.fit(returns_normalized.values.reshape(-1, 1))  # Reshape data for HMM input
    return model

def get_regime_probabilities(model, returns):
    """
    Get smoothed regime probabilities and most probable regimes.
    """
    returns_normalized = (returns - returns.mean()) / returns.std()
    hidden_states = model.predict(returns_normalized.values.reshape(-1, 1))
    smoothed_probs = pd.DataFrame(
        model.predict_proba(returns_normalized.values.reshape(-1, 1)),
        index=returns.index,
        columns=[f"Regime {i}" for i in range(model.n_components)]
    )
    return pd.Series(hidden_states, index=returns.index), smoothed_probs


def combined_hlc_plot(data,
                    high_returns, low_returns, close_returns,
                    high_hidden_states, low_hidden_states, close_hidden_states,
                    high_smoothed_probs, low_smoothed_probs, close_smoothed_probs,
                    n_components, start_date, end_date):
    print("Creating combined plot...")
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    data = data[data.index >= start_date]
    data = data[data.index <= end_date]
    
    high_returns = high_returns[high_returns.index >= start_date]
    high_smoothed_probs = high_smoothed_probs[high_smoothed_probs.index >= start_date]
    high_hidden_states = high_hidden_states[high_hidden_states.index >= start_date]
    high_returns = high_returns[high_returns.index <= end_date]
    high_smoothed_probs = high_smoothed_probs[high_smoothed_probs.index <= end_date]
    high_hidden_states = high_hidden_states[high_hidden_states.index <= end_date]

    low_returns = low_returns[low_returns.index >= start_date]
    low_smoothed_probs = low_smoothed_probs[low_smoothed_probs.index >= start_date]
    low_hidden_states = low_hidden_states[low_hidden_states.index >= start_date]
    low_returns = low_returns[low_returns.index <= end_date]
    low_smoothed_probs = low_smoothed_probs[low_smoothed_probs.index <= end_date]
    low_hidden_states = low_hidden_states[low_hidden_states.index <= end_date]

    close_returns = close_returns[close_returns.index >= start_date]
    close_smoothed_probs = close_smoothed_probs[close_smoothed_probs.index >= start_date]
    close_hidden_states = close_hidden_states[close_hidden_states.index >= start_date]
    close_returns = close_returns[close_returns.index <= end_date]
    close_smoothed_probs = close_smoothed_probs[close_smoothed_probs.index <= end_date]
    close_hidden_states = close_hidden_states[close_hidden_states.index <= end_date]

    price_data = data['Adj Close'].reindex(high_returns.index)
    regime_colors = {f'Regime {i}': plt.cm.tab10(i) for i in range(n_components)}

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)

    axes[0].plot(price_data, label="Adjusted High Price", color='black', linewidth=1.5)
    for regime, color in regime_colors.items():
        regime_mask = (high_hidden_states == int(regime.split()[-1]))
        axes[0].fill_between(
            price_data.index,
            price_data.min(),
            price_data.max(),
            where=regime_mask,
            color=color,
            alpha=0.2,
            label=regime
        )
    axes[0].set_title("Adjusted High Price with Regime Overlay")
    axes[0].set_ylabel("Adjusted High Price")
    axes[0].legend()

    axes[1].plot(price_data, label="Adjusted Low Price", color='black', linewidth=1.5)
    for regime, color in regime_colors.items():
        regime_mask = (low_hidden_states == int(regime.split()[-1]))
        axes[1].fill_between(
            price_data.index,
            price_data.min(),
            price_data.max(),
            where=regime_mask,
            color=color,
            alpha=0.2,
            label=regime
        )
    axes[1].set_title("Adjusted Low Price with Regime Overlay")
    axes[1].set_ylabel("Adjusted Low Price")
    axes[1].legend()

    axes[2].plot(price_data, label="Adjusted Close Price", color='black', linewidth=1.5)
    for regime, color in regime_colors.items():
        regime_mask = (close_hidden_states == int(regime.split()[-1]))
        axes[2].fill_between(
            price_data.index,
            price_data.min(),
            price_data.max(),
            where=regime_mask,
            color=color,
            alpha=0.2,
            label=regime
        )
    axes[2].set_title("Adjusted Close Price with Regime Overlay")
    axes[2].set_ylabel("Adjusted Close Price")
    axes[2].legend()

    # Set common x-label
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.show()
    

def identify_state_changes(hidden_states, window_size):
    """
    Identify where the hidden states change within a given window size.
    """
    state_changes = []
    for i in range(len(hidden_states) - window_size):
        if not all(hidden_states[i:i + window_size] == hidden_states[i]):
            state_changes.append(hidden_states.index[i])
    return pd.Series(state_changes, name="State Change Date")    
    
    
def plot_state_changes(data, high_hidden_states, low_hidden_states, close_hidden_states, plot_start_date):
    """
    Plot the dates where hidden states change within X periods.
    """
    plot_start_date = pd.to_datetime(plot_start_date)
    price_data = data[['Open','High', 'Low', 'Adj Close', 'Close']]
    price_data = price_data[price_data.index >= plot_start_date]

    n_periods = 2        

    high_hidden_states_x = high_hidden_states[high_hidden_states.index >= plot_start_date]
    low_hidden_states_x = low_hidden_states[low_hidden_states.index >= plot_start_date]
    close_hidden_states_x = close_hidden_states[close_hidden_states.index >= plot_start_date]

    # Detect state changes within n periods
    high_changes_x = high_hidden_states_x.diff(n_periods).abs() > 0
    low_changes_x = low_hidden_states_x.diff(n_periods).abs() > 0
    close_changes_x = close_hidden_states_x.diff(n_periods).abs() > 0        
        
    # Identify state changes
    high_changes = identify_state_changes(high_hidden_states, window_size=n_periods)
    low_changes = identify_state_changes(low_hidden_states, window_size=n_periods)
    close_changes = identify_state_changes(close_hidden_states, window_size=n_periods)

    # Filter changes based on plot_start_date
    high_changes_filtered = high_changes[high_changes >= plot_start_date]
    low_changes_filtered = low_changes[low_changes >= plot_start_date]
    close_changes_filtered = close_changes[close_changes >= plot_start_date]

    
    fig, ax = plt.subplots(figsize=(16, 4))

    
    # Plot the price data
    ax.plot(price_data['High'], label="High", color='red', linewidth=1.0)
    ax.plot(price_data['Low'], label="Low", color='green', linewidth=1.0)
    ax.plot(price_data['Adj Close'], label="Close", color='black', linewidth=1.2)
    
    plt.scatter(price_data.index[high_changes_x], 
                price_data['High'][high_changes_x], 
                label=f"High State Changes (within {n_periods} periods)", 
                color='r', s=50, alpha=0.7)

    # Mark changes for Low states
    plt.scatter(price_data.index[low_changes_x], 
                price_data['Low'][low_changes_x], 
                label=f"Low State Changes (within {n_periods} periods)", 
                color='g', s=50, alpha=0.7)

    # Mark changes for Close states
    plt.scatter(price_data.index[close_changes_x], 
                price_data['Close'][close_changes_x], 
                label=f"Close State Changes (within {n_periods} periods)", 
                color='b', s=50, alpha=0.7)
    
    # Highlight state changes
    for change in high_changes_filtered:
        ax.axvline(x=change, color='r', linestyle='--', alpha=0.6, label='High State Change')
    
    for change in low_changes_filtered:
        ax.axvline(x=change, color='g', linestyle='--', alpha=0.6, label='Low State Change')

    for change in close_changes_filtered:
        ax.axvline(x=change, color='b', linestyle='--', alpha=0.6, label='Close State Change')


    # Set labels and title
    ax.set_title("State Changes in High, Low, and Close Prices")
    ax.set_ylabel("HLC Price")
    ax.set_xlabel("Date")
    
    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.show()



# Main Workflow
if __name__ == "__main__":
    n_components = 3  # Number of regimes
    ticker = '^GSPC'  # S&P 500 Index
    start_date = '2010-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    plot_start_date = '2024-01-01'
    hmm_iter = 200
    
    high_returns, data = download_and_prepare_data(ticker, start_date, end_date, 'High')
    low_returns, data = download_and_prepare_data(ticker, start_date, end_date, 'Low')
    close_returns, data = download_and_prepare_data(ticker, start_date, end_date, 'Close')

    high_model = fit_hmm(high_returns, n_components, n_iter=hmm_iter)
    low_model = fit_hmm(low_returns, n_components, n_iter=hmm_iter)
    close_model = fit_hmm(close_returns, n_components, n_iter=hmm_iter)

    high_hidden_states, high_smoothed_probs = get_regime_probabilities(high_model, high_returns)
    low_hidden_states, low_smoothed_probs = get_regime_probabilities(low_model, low_returns)
    close_hidden_states, close_smoothed_probs = get_regime_probabilities(close_model, close_returns)

    #combined_hlc_plot(data, high_returns, low_returns, close_returns,
    #                  high_hidden_states, low_hidden_states, close_hidden_states,
    #                  high_smoothed_probs, low_smoothed_probs, close_smoothed_probs,
    #                  n_components, plot_start_date, end_date)

    plot_state_changes(data, high_hidden_states, low_hidden_states, close_hidden_states, plot_start_date)