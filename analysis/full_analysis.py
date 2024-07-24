import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def merge_data(ind_file, nt_file):

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
    ind_data = pd.read_csv(ind_file, header=None)
    trade_data = pd.read_csv(nt_file, skiprows=0)
    ind_data.columns = ind_header
    trade_data.columns = trade_header

    ind_data['tradedatetime'] = pd.to_datetime(ind_data['tradedatetime'])
    ind_data['tdate'] = pd.to_datetime(ind_data['tdate'])
    ind_data['fulldt'] = pd.to_datetime(ind_data['fulldt'])

    ind_data['tradedatetime'] = ind_data['tradedatetime'] + timedelta(minutes=1)

    trade_data['entrytime'] = pd.to_datetime(trade_data['entrytime'])
    trade_data['exittime'] = pd.to_datetime(trade_data['exittime'])

    merged_data = pd.merge(ind_data, trade_data, how='inner', left_on='tradedatetime', right_on='entrytime')

    # Select and rename columns to match the output structure
    merged_data = merged_data[['tradedatetime', 'signal', 'directon', 
                'sdlr310', 'sdbb91', 'sdkc91', 'sdkc9', 'roc', 'atr34', 'atr32', 'atr31', 
                'atr3', 'atr21', 'atr2', 'RSI', 'stok1', 'entrytime', 'exittime', 'entryprice', 
                'exitprice', 'profit', 'mae', 'mfe' ]]

    # Rename columns to match output format
    merged_data.columns = ['tradedatetime', 'signal_v', 'directon', 
                    'sdlr310', 'sdbb91', 'sdkc91', 'sdkc9', 'roc', 'atr34', 'atr32', 'atr31', 
                    'atr3', 'atr21', 'atr2', 'rsi', 'stok1', 'entrytime', 'exittime', 'entryprice', 
                    'exitprice', 'profit', 'mae', 'mfe']

    return merged_data

def process_data(merged_data):
    merged_data['entry_hour'] = merged_data['entrytime'].dt.hour
    merged_data['entry_day'] = merged_data['entrytime'].dt.dayofweek  # Monday=0, Sunday=6
    merged_data['entry_minute'] = merged_data['entrytime'].dt.minute
    merged_data['directon'] = merged_data['directon'].str.strip().map({'Buy': 1, 'Sell': -1})
    return merged_data

def calculate_distribution_stats(data, indicators):
    """Calculate distribution statistics for indicators among winning trades."""
    winning_trades = data[data['profit'] > 0]
    distribution_stats = winning_trades[indicators].describe().T
    distribution_stats['IQR'] = distribution_stats['75%'] - distribution_stats['25%']
    return distribution_stats

def calculate_time_day_effectiveness(data):
    """Calculate profit statistics based on entry time (hour, day of the week, minute)."""
    day_stats = data.groupby(['entry_day'])['profit'].agg(['count', 'mean', 'sum']).reset_index()
    day_hour_stats = data.groupby(['entry_hour', 'entry_day'])['profit'].agg(['count', 'mean', 'sum']).reset_index()
    hour_stats = data.groupby(['entry_hour'])['profit'].agg(['count', 'mean', 'sum']).reset_index()
    minute_stats = data.groupby(['entry_minute'])['profit'].agg(['count', 'mean', 'sum']).reset_index()
    hour_minute_stats = data.groupby(['entry_hour', 'entry_minute'])['profit'].agg(['count', 'mean', 'sum']).reset_index()
    return day_hour_stats, hour_stats, minute_stats, hour_minute_stats, day_stats

def calculate_profit_statistics(data, indicator, distribution_stats):
    """Calculate profit statistics based on the given indicator thresholds."""
    buy_cond = (
        (data[indicator] >= distribution_stats.loc[indicator, '25%']) & 
        (data[indicator] <= distribution_stats.loc[indicator, '75%']) & 
        (data['directon'] == 1)
    )
    sell_cond = (
        (data[indicator] >= distribution_stats.loc[indicator, '25%']) & 
        (data[indicator] <= distribution_stats.loc[indicator, '75%']) & 
        (data['directon'] == -1)
    )
    filtered_buy_trades = data[buy_cond]
    filtered_sell_trades = data[sell_cond]
    filtered_trades = pd.concat([filtered_buy_trades, filtered_sell_trades])
    
    total_profit = filtered_trades['profit'].sum()
    win_rate = len(filtered_trades[filtered_trades['profit'] > 0]) / len(filtered_trades) if len(filtered_trades) > 0 else 0
    profit_stats = filtered_trades['profit'].describe().to_frame().T
    profit_stats['total_profit'] = total_profit
    profit_stats['win_rate'] = win_rate
    profit_stats['rule_set'] = indicator
    
    return profit_stats

def calculate_correlations_by_group(data, indicators, group_by):
    """Calculate correlations of each indicator with profit by group (day or hour)."""
    grouped = data.groupby(group_by)
    correlation_results = {}
    
    for name, group in grouped:
        correlation_matrix = group[['profit'] + indicators].corr()
        correlation_results[name] = correlation_matrix['profit'][indicators]
    
    return pd.DataFrame(correlation_results)

    # Run the main function
if __name__ == "__main__":
    
    ind_data = 'ind.csv'  #indicator data      
    nt_data = 'trades.csv'  #NT Trade Data
    merged_data = merge_data(ind_data, nt_data)
    full_data = process_data(merged_data)
    
    full_data.to_csv("full_data.csv", index=False)
    
    indicators = ['sdlr310', 'sdbb91', 'sdkc91', 'sdkc9', 'roc', 'atr34', 'atr32', 'atr31', 'atr3', 'atr21', 'atr2', 'rsi', 'stok1']
    
    # 1. Calculate distribution statistics
    distribution_stats = calculate_distribution_stats(full_data, indicators)
    # 2. Calculate time and day effectiveness
    day_hour_stats, hour_stats, minute_stats, hour_minute_stats, day_stats = calculate_time_day_effectiveness(full_data)
    # 3. Calculate profit statistics for each indicator
    results = []
    for indicator in indicators:
        result = calculate_profit_statistics(full_data, indicator, distribution_stats)
        results.append(result)
    # Combine results into a single DataFrame
    individual_results = pd.concat(results).reset_index(drop=True)
    
        # 4. Calculate correlations by day and by hour
    correlations_by_day = calculate_correlations_by_group(full_data, indicators, 'entry_day')
    correlations_by_hour = calculate_correlations_by_group(full_data, indicators, 'entry_hour')
    

        
    # Display results
    print("\nDistribution Results")
    print(distribution_stats.sort_values(by='IQR', ascending=False))
    print("\nIndividual Results:")
    print(individual_results)
    print("\nDay Stats:")
    print(day_stats)
    print("\nDay Hour Stats:")
    print(day_hour_stats)
    print("\nHour Stats:")
    print(hour_stats)    
    print("\nMinute Stats:")
    print(minute_stats)
    print("\nHour Minute Stats:")
    print(hour_minute_stats)    
    print("\nCorrelation Matrix by Day:")
    print(correlations_by_day)
    print("\nCorrelation Matrix by Hour:")
    print(correlations_by_hour)
    
    

    # Plot heatmap for correlations by day
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlations_by_day, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Indicators with Profit by Day of the Week')
    plt.xlabel('Day of the Week (Monday=0, Sunday=6)')
    plt.ylabel('Indicators')
    
    # Plot heatmap for correlations by hour
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlations_by_hour, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Indicators with Profit by Hour of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Indicators')
    
    time_day_pivot = pd.pivot_table(day_hour_stats, values='sum', index='entry_hour', columns='entry_day')
    plt.figure(figsize=(16, 8))
    sns.heatmap(time_day_pivot, annot=True, cmap='Spectral', fmt=".2f")
    plt.title('Profit by Time of Day and Day of the Week')
    plt.xlabel('Day of the Week (Monday=0, Sunday=6)')
    plt.ylabel('Hour of the Day')

    # Plot heatmap for time and minute effectiveness
    time_minute_pivot = pd.pivot_table(hour_minute_stats, values='sum', index='entry_hour', columns='entry_minute')
    plt.figure(figsize=(16, 8))
    sns.heatmap(time_minute_pivot, annot=False, cmap='Spectral', fmt=".2f")
    plt.title('Profit by Time of Day and Minute')
    plt.xlabel('Minute of the Hour')
    plt.ylabel('Hour of the Day')
    

    
    plt.show()

