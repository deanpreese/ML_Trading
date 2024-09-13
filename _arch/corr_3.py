import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Load the dataset

data_loaded = pd.read_csv('data/sm13_3070.csv')
#data_loaded = pd.read_csv('data/buildSeqInd_Lucky13_5M_ALL.csv')

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
data_loaded = data_loaded.drop(columns=['SDKC91'])
#data_loaded = data_loaded.drop(columns=['SDBB91'])
#data_loaded = data_loaded.drop(columns=['SDLR310'])

data_loaded['ATR2'] = data_loaded['ATR2'] * -1
data_loaded['ATR3'] = data_loaded['ATR3'] * -1
data_loaded['SDKC9'] = data_loaded['SDKC9'] * -1
data_loaded['SDBB91'] = data_loaded['SDBB91'] * -1

data = data_loaded.drop(columns=['outputC'])




# Assuming the last column is the output
output = data.iloc[:, -1]
features = data.iloc[:, :-1]

results = pd.DataFrame(index=data.index)

window_size = 250
plot_size = 2000

# Calculate rolling correlations for each feature
for feature in features.columns:
    rolling_corr = features[feature].rolling(window=window_size).corr(output)
    results[f'{feature}_corr'] = rolling_corr

# Calculate the cumulative sum of the output
output_cumsum = output.cumsum()


# Only keep the last 500 values
results = results.tail(plot_size)
output_cumsum = output_cumsum.tail(plot_size)

# Determine the grid size
num_features = len(features.columns)
grid_size = math.ceil(num_features ** 0.5)

# Plotting each raw correlation and the cumulative sum of the output on separate axes
fig, axes = plt.subplots(grid_size, grid_size, figsize=(18, 8), sharex=True)
axes = axes.flatten()

for i, feature in enumerate(features.columns):
    ax1 = axes[i]
    ax2 = ax1.twinx()
    
     # Plotting correlation with colors based on their value
    corr = results[f'{feature}_corr']
    
    #ax1.plot(corr.index, corr, label=f'{feature} Correlation', color='blue')
    ax1.fill_between(corr.index, corr, where=(corr > 0), color='green', alpha=0.5, interpolate=True)
    ax1.fill_between(corr.index, corr, where=(corr < 0), color='red', alpha=0.5, interpolate=True)
    
    
    #ax1.plot(results.index, results[f'{feature}_corr'], label=f'{feature} Correlation', color='blue')
    ax2.plot(results.index, output_cumsum, label='Cumulative Sum', color='black', alpha=0.6)

    ax1.set_title(f'{feature}')
    ax1.set_ylabel('Correlation')
    ax2.set_ylabel('Cumulative Sum')
    
    #ax1.legend(loc='upper left')
    #ax2.legend(loc='upper right')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()