import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
import warnings
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def download_and_prepare_data(ticker, start_date, end_date):
    """
    Download historical price data for a given ticker.
    """
    raw_data = yf.download(ticker, start=start_date, end=end_date)
    data = raw_data
    
    with np.errstate(divide='ignore', invalid='ignore'):
        data['LogReturn'] = np.log(raw_data['Adj Close'] / raw_data['Adj Close'].shift(1))
        
    return data['LogReturn'].dropna(), raw_data


def fit_markov_model(returns, iter, reps, k_regimes, order=1):
    """
    Fit a Markov Regime-Switching model with a parameterized number of regimes.
    """
    print(f"Fitting Markov Regime-Switching model with {k_regimes} regimes...")
    model = MarkovAutoregression(
        returns, k_regimes=k_regimes, order=order, switching_variance=True
    )
    return model.fit(em_iter=iter, search_reps=reps, em_verbose=True)


def analyze_results(result, k_regimes):
    """
    Extract and analyze model parameters dynamically based on the number of regimes.
    """
    params = result.params
    ar_params = params.filter(regex=r'ar\.L1\[\d+\]').index.tolist()
    intercept_params = params.filter(regex=r'const\[\d+\]').index.tolist()
    variance_params = params.filter(regex=r'sigma2\[\d+\]').index.tolist()

    if not (len(ar_params) == len(intercept_params) == len(variance_params) == k_regimes):
        raise ValueError("Mismatch in number of parameters or regimes.")

    regime_numbers = range(k_regimes)
    regime_params = pd.DataFrame({
        'Regime': regime_numbers,
        'Intercept': params[intercept_params].values,
        'AR_Coefficient': params[ar_params].values,
        'Variance': params[variance_params].values
    })

    # Calculate implied mean
    regime_params['Mean'] = np.where(
        regime_params['AR_Coefficient'] != 1,
        regime_params['Intercept'] / (1 - regime_params['AR_Coefficient']),
        np.nan
    )
    regime_params_sorted = regime_params.sort_values(by='Mean').reset_index(drop=True)

    # Assign dynamic market states
    market_states = [f'Regime {i}' for i in range(k_regimes)]
    regime_params_sorted['Market State'] = market_states

    return regime_params.merge(
        regime_params_sorted[['Regime', 'Market State']],
        on='Regime',
        how='left'
    )


def combined_plot(data, returns, result, k_regimes, start_date=None, end_date=None, num_values=None):
    """
    Create a combined plot with:
    1. Smoothed probabilities
    2. Daily log returns by regime
    3. Adjusted close price with regime overlay
    Optionally, limit the plot to a given number of values.
    """
    print("Creating combined plot...")
    
    # Prepare data for plotting
    smoothed_probs = result.smoothed_marginal_probabilities
    smoothed_probs.columns = [f'Regime {i}' for i in range(k_regimes)]
    most_probable_regimes = smoothed_probs.idxmax(axis=1)
    
    if start_date:
        start_date = pd.to_datetime(start_date)
        smoothed_probs = smoothed_probs[smoothed_probs.index >= start_date]
        most_probable_regimes = most_probable_regimes[most_probable_regimes.index >= start_date]
        returns = returns[returns.index >= start_date]
        data = data[data.index >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        smoothed_probs = smoothed_probs[smoothed_probs.index <= end_date]
        most_probable_regimes = most_probable_regimes[most_probable_regimes.index <= end_date]
        returns = returns[returns.index <= end_date]
        data = data[data.index <= end_date]

    if num_values:
        smoothed_probs = smoothed_probs.tail(num_values)
        most_probable_regimes = most_probable_regimes.tail(num_values)
        returns = returns.tail(num_values)
        data = data.tail(num_values)
    
    price_data = data['Adj Close'].reindex(most_probable_regimes.index)
    regime_colors = {f'Regime {i}': plt.cm.tab10(i) for i in range(k_regimes)}

    # Create subplots for the plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    
    # Plot 1: Smoothed Probabilities
    for column in smoothed_probs.columns:
        axes[0].plot(smoothed_probs[column], label=column)
    axes[0].set_title("Smoothed Probabilities of Each Regime")
    axes[0].set_ylabel("Probability")
    axes[0].legend()

    # Plot 2: Daily Log Returns by Regime
    for regime, color in regime_colors.items():
        axes[1].scatter(
            returns[most_probable_regimes == regime].index,
            returns[most_probable_regimes == regime],
            color=color,
            label=regime,
            s=10
        )
    axes[1].set_title("Daily Log Returns by Most Probable Regime")
    axes[1].set_ylabel("Log Return")
    axes[1].legend()

    # Plot 3: Adjusted Close Price with Regime Overlay
    axes[2].plot(price_data, label="Adjusted Close Price", color='black', linewidth=1.5)
    for regime, color in regime_colors.items():
        regime_mask = most_probable_regimes == regime
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
    
    # Adjust layout
    plt.tight_layout()
    plt.show()



def calculate_transition_matrix(result, k_regimes):
    """
    Calculate the transition matrix from the model parameters for MarkovAutoregression.
    """
    print("Calculating transition matrix...")
    
    # Initialize the transition matrix
    transition_matrix = np.zeros((k_regimes, k_regimes))
    params = result.params

    # Extract transition probabilities
    for j in range(k_regimes):
        sum_probs = 0
        for i in range(k_regimes - 1):
            param_name = f'p[{j}->{i}]'
            prob = params.get(param_name, 0)  # Get the parameter value
            transition_matrix[j, i] = prob
            sum_probs += prob

        # Compute the last probability as 1 - sum of other probabilities
        transition_matrix[j, k_regimes - 1] = 1 - sum_probs

    return pd.DataFrame(
        transition_matrix,
        columns=[f'To Regime {i}' for i in range(k_regimes)],
        index=[f'From Regime {i}' for i in range(k_regimes)]
    )


def display_transition_matrix(result, k_regimes):
    """
    Display the transition matrix of the Markov Switching Model.
    """
    transition_matrix = calculate_transition_matrix(result, k_regimes)
    print("\n### Transition Matrix ###\n")
    print(transition_matrix)
    return transition_matrix


# Main Workflow
if __name__ == "__main__":
    # Number of regimes (parameter-driven)
    k_regimes = 4

    em_iter = 100
    search_reps = 20
    tcker = '^GSPC'  # S&P 500 Index
    start_date = '2010-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    plot_start_date = '2024-01-01'

    returns, data = download_and_prepare_data(tcker, start_date, end_date)
    result = fit_markov_model(returns, em_iter, search_reps, k_regimes=k_regimes, order=1)
    regime_params = analyze_results(result, k_regimes)
    print("\nRegime Parameters:\n")
    print(regime_params)

    transition_matrix = display_transition_matrix(result, k_regimes)

    # Step 5: Combined Plot
    # data, returns, result, k_regimes, start_date=None, end_date=None, num_values=None):

    combined_plot(data, returns, result, k_regimes, 
                  start_date=plot_start_date, end_date=end_date,num_values=100)
