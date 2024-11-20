# Markov Regime-Switching Model with 3 States
# States: Uptrend, Downtrend, Sideways

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

import warnings
import re
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# 1. Data Collection and Preparation
# ----------------------------------

# Define the ticker symbol and date range
ticker = '^GSPC'  # S&P 500 Index
start_date = '2010-01-01'

# Suggestion 1: Adjust 'end_date' to today's date to avoid future dates
end_date = datetime.today().strftime('%Y-%m-%d')

# Download historical price data using yfinance
try:
    data = yf.download(ticker, start=start_date, end=end_date)
except Exception as e:
    print(f"Error downloading data: {e}")
    exit()

# Check if data is not empty
if data.empty:
    print("No data downloaded. Please check the ticker symbol and date range.")
    exit()

# Calculate daily log returns
data['LogReturn'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))

# Drop missing values resulting from the shift
returns = data['LogReturn'].dropna()
# 2. Implement the Markov Regime-Switching Autoregressive Model
# -------------------------------------------------------------

model = MarkovAutoregression(
    returns,
    k_regimes=3,
    order=1,                # Include AR(1) term
    switching_variance=True # Allow variance to switch between regimes
)

result = model.fit(
    em_iter=100,
    search_reps=20,
    em_verbose=True
)


# 3. Analyze and Interpret Results
# --------------------------------

print(result.summary())

params = result.params

# Print the model parameters to inspect their names
print("\nModel Parameters:\n")
print(params)

# Extract the AR coefficients
ar_params = [name for name in params.index if re.match(r'ar\.L1\[\d+\]', name)]
ar_params.sort()

# Extract the intercepts
intercept_params = [name for name in params.index if re.match(r'const\[\d+\]', name)]
intercept_params.sort()

# Extract the variances
variance_params = [name for name in params.index if re.match(r'sigma2\[\d+\]', name)]
variance_params.sort()

# Verify the parameter names
print("\nIntercept parameters:", intercept_params)
print("AR parameters:", ar_params)
print("Variance parameters:", variance_params)

# Ensure that all lists have the same length
if not (len(ar_params) == len(intercept_params) == len(variance_params)):
    raise ValueError("Mismatch in number of parameters.")

# Extract regime numbers from parameter names
def extract_regime_number(param_name):
    match = re.search(r'\[(\d+)\]', param_name)
    if match:
        return int(match.group(1))
    else:
        return None

regime_numbers = [extract_regime_number(name) for name in intercept_params]

# Check for None values
if None in regime_numbers:
    raise ValueError("Failed to extract regime numbers from parameter names.")

# Create the DataFrame
regime_params = pd.DataFrame({
    'Regime': regime_numbers,
    'Intercept': params[intercept_params].values,
    'AR_Coefficient': params[ar_params].values,
    'Variance': params[variance_params].values
})

# Calculate the implied mean of each regime
regime_params['Mean'] = regime_params['Intercept'] / (1 - regime_params['AR_Coefficient'])

print("\nRegime Parameters:\n")
print(regime_params)

# 3.1 Map Regimes to Market States
# --------------------------------

# Sort regimes by the implied mean
regime_params_sorted = regime_params.sort_values(by='Mean').reset_index(drop=True)

# Assign market states based on sorted mean returns
market_states = ['Downtrend', 'Sideways', 'Uptrend']

# Ensure lengths match
if len(regime_params_sorted) != len(market_states):
    raise ValueError("Number of regimes and market states do not match.")

regime_params_sorted['Market State'] = market_states

# Merge 'Market State' back to 'regime_params' based on 'Regime' column
regime_params = regime_params.merge(
    regime_params_sorted[['Regime', 'Market State']],
    on='Regime',
    how='left'
)

print("\nRegime Parameters with Market States:\n")
print(regime_params)



# 4. Determine the Current Market State
# -------------------------------------

# Get smoothed probabilities of each regime over time
smoothed_probs = result.smoothed_marginal_probabilities

# Rename columns for clarity
smoothed_probs.columns = ['Regime %d' % i for i in range(result.k_regimes)]

# Get the latest date from the data
latest_date = smoothed_probs.index[-1]

# Identify the regime with the highest probability at the latest date
current_regime = smoothed_probs.loc[latest_date].idxmax()

# Extract the corresponding market state
regime_index = int(current_regime.split()[-1])
current_state = regime_params.loc[regime_index, 'Market State']

print(f"\nAs of {latest_date.date()}, the most probable regime is: {current_regime}")
print(f"The current market state is: {current_state}")

# 5. Visualize the Regimes
# ------------------------

# Suggestion 4: Adjust 'plot_start_date' to a date within your data range
# Define the date range for plotting (set to None to plot all data)
plot_start_date = '2024-01-01'  # Adjusted to a past date within data range
plot_end_date = end_date

# 5.1 Plot smoothed probabilities of each regime over time
# --------------------------------------------------------

# Filter smoothed probabilities based on plotting date range
smoothed_probs_filtered = smoothed_probs.copy()
if plot_start_date is not None:
    plot_start_date_pd = pd.to_datetime(plot_start_date)
    smoothed_probs_filtered = smoothed_probs_filtered[smoothed_probs_filtered.index >= plot_start_date_pd]

if plot_end_date is not None:
    plot_end_date_pd = pd.to_datetime(plot_end_date)
    smoothed_probs_filtered = smoothed_probs_filtered[smoothed_probs_filtered.index <= plot_end_date_pd]

plt.figure(figsize=(12, 6))
for i in range(result.k_regimes):
    plt.plot(
        smoothed_probs_filtered['Regime %d' % i],
        label='Regime %d' % i
    )
plt.title('Smoothed Probabilities of Each Regime')
plt.xlabel('Date')
plt.ylabel('Probability')
plt.legend()
plt.show()

# 5.2 Plot daily log returns colored by the most probable regime
# --------------------------------------------------------------

# Determine the most probable regime at each time point
most_probable_regimes = smoothed_probs.idxmax(axis=1)

# Filter based on plotting date range
most_probable_regimes_filtered = most_probable_regimes.copy()
if plot_start_date is not None:
    most_probable_regimes_filtered = most_probable_regimes_filtered[most_probable_regimes_filtered.index >= plot_start_date_pd]

if plot_end_date is not None:
    most_probable_regimes_filtered = most_probable_regimes_filtered[most_probable_regimes_filtered.index <= plot_end_date_pd]

returns_filtered = returns.reindex(most_probable_regimes_filtered.index)

# Map regimes to colors
regime_colors = {
    'Regime 0': 'red',
    'Regime 1': 'orange',
    'Regime 2': 'green'
}

# Plot returns colored by regime
plt.figure(figsize=(12, 6))
for regime in smoothed_probs.columns:
    plt.scatter(
        returns_filtered[most_probable_regimes_filtered == regime].index,
        returns_filtered[most_probable_regimes_filtered == regime],
        color=regime_colors[regime],
        label=regime,
        s=10
    )
plt.title('Daily Log Returns Colored by Most Probable Regime')
plt.xlabel('Date')
plt.ylabel('Log Return')
plt.legend()
plt.show()

# 5.3 Plot Regimes on the Actual Close Data
# -----------------------------------------

# Reindex most_probable_regimes to match data index
most_probable_regimes = most_probable_regimes.reindex(data.index, method='ffill')

# Combine adjusted close prices and regimes into a DataFrame
price_regime_df = data[['Adj Close']].copy()
price_regime_df['Regime'] = most_probable_regimes

# Drop any rows with missing values
price_regime_df.dropna(inplace=True)

# Reset the index to have 'Date' as a column
price_regime_df.reset_index(inplace=True)

# Filter data based on the plotting date range
if plot_start_date is not None:
    plot_start_date_pd = pd.to_datetime(plot_start_date)
    price_regime_df = price_regime_df[price_regime_df['Date'] >= plot_start_date_pd]

if plot_end_date is not None:
    plot_end_date_pd = pd.to_datetime(plot_end_date)
    price_regime_df = price_regime_df[price_regime_df['Date'] <= plot_end_date_pd]

# Map regimes to colors
regime_colors = {
    'Regime 0': 'red',       # Downtrend
    'Regime 1': 'orange',    # Sideways
    'Regime 2': 'green'      # Uptrend
}

# Map regimes to market states for labeling
regime_to_state = dict(zip(['Regime %d' % i for i in regime_params['Regime']], regime_params['Market State']))

# Import required modules
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

# Prepare data for LineCollection
dates = mdates.date2num(price_regime_df['Date'])
prices = price_regime_df['Adj Close'].to_numpy().reshape(-1)
colors = price_regime_df['Regime'].map(regime_colors).values

# Ensure that dates, prices, and colors are all 1D NumPy arrays
dates = np.array(dates)
prices = np.array(prices)
colors = np.array(colors)

# Remove any NaN values just in case
valid_mask = ~np.isnan(dates) & ~np.isnan(prices)
dates = dates[valid_mask]
prices = prices[valid_mask]
colors = colors[valid_mask]

# Create segments for LineCollection
points = np.array([dates, prices]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create a color array for the segments
color_list = colors[:-1]  # Exclude the last color to match the number of segments

# Create LineCollection
lc = LineCollection(segments, colors=color_list, linewidth=2)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.add_collection(lc)
ax.set_xlim(dates.min(), dates.max())
ax.set_ylim(prices.min(), prices.max())

# Format x-axis with dates
ax.xaxis_date()
fig.autofmt_xdate()

# Add legend
legend_elements = [Line2D([0], [0], color=regime_colors[regime], lw=2, label=regime_to_state[regime]) for regime in regime_colors]
ax.legend(handles=legend_elements)

ax.set_title('S&P 500 Adjusted Closing Prices with Regimes')
ax.set_xlabel('Date')
ax.set_ylabel('Adjusted Close Price')
plt.show()

# 6. Display the Transition Matrix
# --------------------------------

k_regimes = result.k_regimes

# Collect transition probabilities from result.params
trans_params = {name: result.params[name] for name in result.params.index if 'p[' in name}

# Initialize the transition matrix
transition_matrix = np.zeros((k_regimes, k_regimes))

# Construct the transition matrix
for j in range(k_regimes):
    sum_probs = 0
    for i in range(k_regimes - 1):
        param_name = f'p[{j}->{i}]'
        prob = trans_params.get(param_name, 0)
        transition_matrix[j, i] = prob
        sum_probs += prob
    # Compute the last probability for state k_regimes - 1
    transition_matrix[j, k_regimes - 1] = 1 - sum_probs

# Create a DataFrame for better readability
transition_matrix_df = pd.DataFrame(
    transition_matrix,
    columns=[f'To Regime {i}' for i in range(k_regimes)],
    index=[f'From Regime {i}' for i in range(k_regimes)]
)

print("\nTransition Matrix:\n")
print(transition_matrix_df)

# Suggestion 5: Verify that each row sums to 1
row_sums = transition_matrix_df.sum(axis=1)
print("\nRow sums (should be close to 1.0):\n", row_sums)
