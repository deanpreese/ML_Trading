import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load the dataset and prepare necessary columns."""
    data = pd.read_csv(file_path)
    data['entrytime'] = pd.to_datetime(data['entrytime'])
    data['entry_hour'] = data['entrytime'].dt.hour
    data['entry_day'] = data['entrytime'].dt.dayofweek  # Monday=0, Sunday=6
    data['entry_minute'] = data['entrytime'].dt.minute
    data['directon'] = data['directon'].str.strip().map({'Buy': 1, 'Sell': -1})
    return data

def calculate_distribution_stats(data, indicators):
    """Calculate distribution statistics for indicators among winning trades."""
    winning_trades = data[data['profit'] > 0]
    distribution_stats = winning_trades[indicators].describe().T
    distribution_stats['IQR'] = distribution_stats['75%'] - distribution_stats['25%']
    return distribution_stats

def calculate_time_day_effectiveness(data):
    """Calculate profit statistics based on entry time (hour, day of the week, minute)."""
    day_stats = data.groupby(['entry_hour', 'entry_day'])['profit'].agg(['count', 'mean', 'sum']).reset_index()
    hour_stats = data.groupby(['entry_hour'])['profit'].agg(['count', 'mean', 'sum']).reset_index()
    time_minute_stats = data.groupby(['entry_minute'])['profit'].agg(['count', 'mean', 'sum']).reset_index()
    return day_stats, hour_stats, time_minute_stats

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

def main(file_path):
    # Load and prepare data
    data = load_data(file_path)
    
    # Define indicators
    indicators = ['sdlr310', 'sdbb91', 'sdkc91', 'sdkc9', 'roc', 'atr34', 'atr32', 'atr31', 'atr3', 'atr21', 'atr2', 'rsi', 'stok1']
    
    # 1. Calculate distribution statistics
    distribution_stats = calculate_distribution_stats(data, indicators)
    
    # 2. Calculate time and day effectiveness
    day_stats, hour_stats, time_minute_stats = calculate_time_day_effectiveness(data)
    
    # 3. Calculate profit statistics for each indicator
    results = []
    for indicator in indicators:
        result = calculate_profit_statistics(data, indicator, distribution_stats)
        results.append(result)
    
    # Combine results into a single DataFrame
    individual_results = pd.concat(results).reset_index(drop=True)
    
    # Display results
    print("Individual Results:")
    print(individual_results)
    print("\nDay Stats:")
    print(day_stats)
    print("\nTime Stats:")
    print(hour_stats)    
    print("\nTime Minute Stats:")
    print(time_minute_stats)
    
    
    # Run the main function
if __name__ == "__main__":
    file_path = 'data1.csv'  # Update with your file path
    main(file_path)
