import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from datetime import datetime
import warnings

def download_and_prepare_data(ticker, start_date, end_date, data_col):
    
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

def compute_bic(model, returns):
    """Computes BIC for a given HMM model."""
    n_params = (
        model.n_components**2  # Transition matrix parameters
        + model.n_components - 1  # Initial state probabilities
        + model.n_components * model.n_features  # Emission means
        + model.n_components * model.n_features  # Emission variances
    )
    log_likelihood = model.score(returns)
    bic = np.log(returns.shape[0]) * n_params - 2 * log_likelihood
    return bic

def select_optimal_hmm(returns, n_components_range, n_iter=100, verbose=False):
    """
    Selects the optimal HMM based on BIC.

    Parameters:
        returns (pd.Series): Time series of returns.
        n_components_range (range): Range of number of regimes to test.
        n_iter (int): Number of iterations for HMM training.
        verbose (bool): If True, print detailed log.

    Returns:
        GaussianHMM: The optimal HMM model.
    """
    best_model, best_bic, best_n_components = None, float('inf'), None
    returns_normalized = (returns - returns.mean()) / returns.std()
    for n in n_components_range:
        if verbose:
            print(f"Training HMM with {n} regimes...")
        model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=n_iter, random_state=42)
        model.fit(returns_normalized.values.reshape(-1, 1))
        bic = compute_bic(model, returns_normalized.values.reshape(-1, 1))
        if bic < best_bic:
            best_model, best_bic, best_n_components = model, bic, n
    print(f"Optimal model: {best_n_components} regimes (BIC: {best_bic:.2f})")
    return best_model

def get_regime_probabilities(model, returns):
    """
    Get smoothed regime probabilities and most probable regimes.

    Parameters:
        model (GaussianHMM): Fitted HMM model.
        returns (pd.Series): Time series of returns.

    Returns:
        pd.Series: Most probable regimes.
        pd.DataFrame: Smoothed probabilities.
    """
    returns_normalized = (returns - returns.mean()) / returns.std()
    hidden_states = model.predict(returns_normalized.values.reshape(-1, 1))
    smoothed_probs = pd.DataFrame(
        model.predict_proba(returns_normalized.values.reshape(-1, 1)),
        index=returns.index,
        columns=[f"Regime {i}" for i in range(model.n_components)]
    )
    return pd.Series(hidden_states, index=returns.index), smoothed_probs

def filter_data_within_date_range(data, start_date, end_date):
    """Filters data within the specified date range."""
    return data[(data.index >= start_date) & (data.index <= end_date)]

def plot_state_changes(data, high_hidden_states, low_hidden_states, close_hidden_states, plot_start_date, n_periods=2):
    """
    Plot the dates where hidden states change within specified periods.

    Parameters:
        data (pd.DataFrame): Historical price data.
        high_hidden_states (pd.Series): Hidden states for high prices.
        low_hidden_states (pd.Series): Hidden states for low prices.
        close_hidden_states (pd.Series): Hidden states for close prices.
        plot_start_date (str): Start date for plotting.
        n_periods (int): Window size for state change detection.
    """
    plot_start_date = pd.to_datetime(plot_start_date)
    price_data = data[['High', 'Low', 'Adj Close']]
    price_data = price_data[price_data.index >= plot_start_date]

    fig, ax = plt.subplots(figsize=(16, 4))

    # Plot the price data
    ax.plot(price_data['High'], label="High", color='red', linewidth=1.0)
    ax.plot(price_data['Low'], label="Low", color='green', linewidth=1.0)
    ax.plot(price_data['Adj Close'], label="Close", color='black', linewidth=1.2)

    # Highlight state changes for high, low, and close
    for label, states, color in [("High", high_hidden_states, 'r'),
                                 ("Low", low_hidden_states, 'g'),
                                 ("Adj Close", close_hidden_states, 'b')]:
        states = states[states.index >= plot_start_date]
        state_changes = states.diff(n_periods).abs() > 0
        ax.scatter(price_data.index[state_changes],
                   price_data[label][state_changes],
                   label=f"{label} State Changes", color=color, s=50, alpha=0.7)

    # Set labels and title
    ax.set_title("State Changes in High, Low, and Close Prices")
    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    
    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    #ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()
    plt.show()

# Main Workflow
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Parameters
    ticker = '^GSPC'
    start_date = '2010-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    plot_start_date = '2024-01-01'
    n_components_range = range(3, 10)
    hmm_iter = 200

    # Prepare data
    high_returns, data = download_and_prepare_data(ticker, start_date, end_date, 'High')
    low_returns, _ = download_and_prepare_data(ticker, start_date, end_date, 'Low')
    close_returns, _ = download_and_prepare_data(ticker, start_date, end_date, 'Close')

    # Fit models
    high_model = select_optimal_hmm(high_returns, n_components_range, n_iter=hmm_iter)
    low_model = select_optimal_hmm(low_returns, n_components_range, n_iter=hmm_iter)
    close_model = select_optimal_hmm(close_returns, n_components_range, n_iter=hmm_iter)

    # Get regime probabilities
    high_hidden_states, high_smoothed_probs = get_regime_probabilities(high_model, high_returns)
    low_hidden_states, low_smoothed_probs = get_regime_probabilities(low_model, low_returns)
    close_hidden_states, close_smoothed_probs = get_regime_probabilities(close_model, close_returns)

    # Plot state changes
    plot_state_changes(data, high_hidden_states, low_hidden_states, close_hidden_states, plot_start_date)
