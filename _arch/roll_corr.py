import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#file_loaded = pd.read_csv('data/sm13_3070.csv')
file_loaded = pd.read_csv('data/buildSeqInd_Lucky13_5M_ALL.csv')
data_loaded = file_loaded.drop(columns=['outputC'])

output_column = 'output'  

#data_loaded = data_loaded.drop(columns=['STOK1'])
#data_loaded = data_loaded.drop(columns=['RSI'])
#data_loaded = data_loaded.drop(columns=['ATR2'])
data_loaded = data_loaded.drop(columns=['ATR21'])
#data_loaded = data_loaded.drop(columns=['ATR3'])
data_loaded = data_loaded.drop(columns=['ATR31'])
data_loaded = data_loaded.drop(columns=['ATR32'])
data_loaded = data_loaded.drop(columns=['ATR34'])
#data_loaded = data_loaded.drop(columns=['ROC'])
#data_loaded = data_loaded.drop(columns=['SDKC9'])
#data_loaded = data_loaded.drop(columns=['SDKC91'])
#data_loaded = data_loaded.drop(columns=['SDBB91'])
#data_loaded = data_loaded.drop(columns=['SDLR310'])

df = data_loaded

# Ensure the output column is in the dataframe
if output_column not in df.columns:
    raise ValueError(f"The specified output column '{output_column}' does not exist in the dataset.")

# Initialize DataFrames to store rolling correlations
rolling_correlations_21 = pd.DataFrame()
rolling_correlations_50 = pd.DataFrame()
rolling_correlations_1000 = pd.DataFrame()

# Define the rolling window size
window_size = 1000
ema_span_1 = 21
ema_span_2 = 50

# Calculate rolling correlation for each feature against the output
for feature in df.columns:
    if feature != output_column:
        rolling_correlation = df[feature].rolling(window=window_size).corr(df[output_column])
        rolling_correlations_1000[feature] = rolling_correlation
        rolling_correlations_21[feature] = rolling_correlation.ewm(span=ema_span_1).mean()
        rolling_correlations_50[feature] = rolling_correlation.ewm(span=ema_span_2).mean()

# Calculate the rolling summation of the output values
rolling_sum_output = df[output_column].rolling(window=window_size).sum()

# Calculate the composite correlations
composite_correlation_1000 = rolling_correlations_1000.mean(axis=1)
composite_correlation_21 = rolling_correlations_21.mean(axis=1)
composite_correlation_50 = rolling_correlations_50.mean(axis=1)

# Save the rolling correlations to new CSV files
rolling_correlations_1000.to_csv(f'rolling_correlations_{window_size}_period.csv', index=False)
rolling_correlations_21.to_csv(f'rolling_correlations_{ema_span_1}_period_ema.csv', index=False)
rolling_correlations_50.to_csv(f'rolling_correlations_{ema_span_2}_period_ema.csv', index=False)

num_features = len(rolling_correlations_1000.columns)
cols = 3
rows = 3
#rows = (num_features // cols) + (num_features % cols > 0)

fig, axes = plt.subplots(rows, cols, figsize=(16, 9), sharex=True)

# Plot each feature
for i, feature in enumerate(rolling_correlations_1000.columns):
    row, col = divmod(i, cols)
    ax = axes[row, col]
    corr_1000 = rolling_correlations_1000[feature].dropna().tail(100)
    ema_21 = rolling_correlations_21[feature].dropna().tail(100)
    ema_50 = rolling_correlations_50[feature].dropna().tail(100)
    comp_corr_1000 = composite_correlation_1000.dropna().tail(100)
    comp_corr_21 = composite_correlation_21.dropna().tail(100)
    comp_corr_50 = composite_correlation_50.dropna().tail(100)
    sum_output = rolling_sum_output.dropna().tail(100)
    
    ax.plot(corr_1000.index, corr_1000.values, color='blue', label=f'{window_size}-Period Correlation')
    ax.plot(ema_21.index, ema_21.values, color='green', linestyle='dashed', label=f'{ema_span_1}-Period EMA')
    ax.plot(ema_50.index, ema_50.values, color='red', linestyle='dotted', label=f'{ema_span_2}-Period EMA')
    
    # Plotting composite correlations
    #ax.plot(comp_corr_1000.index, comp_corr_1000.values, color='black', linestyle='solid', label='1000-Period Composite')
    #ax.plot(comp_corr_21.index, comp_corr_21.values, color='cyan', linestyle='dashed', label='21-Period Composite')
    #ax.plot(comp_corr_50.index, comp_corr_50.values, color='magenta', linestyle='dotted', label='50-Period Composite')

    # Secondary y-axis for the rolling summation of output
    ax2 = ax.twinx()
    ax2.plot(sum_output.index, sum_output.values, color='black', label='Rolling Sum Output')

    ax.set_title(f'Last {window_size} Corr {feature} ')
    ax.set_xlabel('Row Index')
    ax.set_ylabel('Correlation')
    ax2.set_ylabel('Rolling Summation of Output', color='black')
    ax.legend(loc='upper left')
    #ax2.legend(loc='upper right')

# Adjust layout and remove any empty subplots
plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.6)
plt.show()