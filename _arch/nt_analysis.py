import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def import_trade_data(nt_file):
    trade_header = [
        'tradenumber', 'instrument', 'account', 'strategy', 'marketpos', 'qty', 
        'entryprice', 'exitprice', 'entrytime', 'exittime', 'entryname', 'exitname', 
        'profit', 'cumnetprofit', 'commission', 'mae', 'mfe', 'etd', 'xxx'
    ]

    # Read the CSV files without headers
    trade_data = pd.read_csv(nt_file, skiprows=0)
    trade_data.columns = trade_header
    trade_data['entrytime'] = pd.to_datetime(trade_data['entrytime'])
    trade_data['exittime'] = pd.to_datetime(trade_data['exittime'])
    return trade_data

def process_data(data):
    data['entry_hour'] = data['entrytime'].dt.hour
    data['entry_day'] = data['entrytime'].dt.dayofweek  # Monday=0, Sunday=6
    data['entry_minute'] = data['entrytime'].dt.minute
    data['directon'] = data['marketpos'].str.strip().map({'Buy': 1, 'Sell': -1})
    data['profit'] = data['profit'].astype(float)
    data['mae'] = data['mae'].astype(float)
    data['mfe'] = data['mfe'].astype(float)
    return data

def calculate_time_day_effectiveness(data):
    """Calculate profit statistics based on entry time (hour, day of the week, minute)."""
    day_stats = data.groupby(['entry_day'])['profit'].agg(['count', 'mean', 'sum']).reset_index()
    day_hour_stats = data.groupby(['entry_hour', 'entry_day'])['profit'].agg(['count', 'mean', 'sum']).reset_index()
    hour_stats = data.groupby(['entry_hour'])['profit'].agg(['count', 'mean', 'sum']).reset_index()
    minute_stats = data.groupby(['entry_minute'])['profit'].agg(['count', 'mean', 'sum']).reset_index()
    hour_minute_stats = data.groupby(['entry_hour', 'entry_minute'])['profit'].agg(['count', 'mean', 'sum']).reset_index()
    return day_hour_stats, hour_stats, minute_stats, hour_minute_stats, day_stats

def generate_histogram_data(data, column_name, bins=10):
    counts, bin_edges = np.histogram(data[column_name], bins=bins)
    histogram_df = pd.DataFrame({
        'Bin Start': bin_edges[:-1],
        'Bin End': bin_edges[1:],
        'Frequency': counts
    })
    return histogram_df

def plot_histograms(nt_data, day_stats, hour_stats):
    fig, axes = plt.subplots(3, 1, figsize=(16, 7))

    # Plot Profit Distribution
    profit_hist_data = generate_histogram_data(nt_data, "profit", 50)
    axes[0].bar(profit_hist_data['Bin Start'], profit_hist_data['Frequency'], width=profit_hist_data['Bin End'] - profit_hist_data['Bin Start'], align='edge',
                color=['green' if x >= 0 else 'red' for x in profit_hist_data['Bin Start']])
    axes[0].set_title('Profit Distribution')
    axes[0].set_xlabel('Profit')
    axes[0].set_ylabel('Frequency')

    # Plot MAE Distribution
    mae_hist_data = generate_histogram_data(nt_data, "mae", 50)
    axes[1].bar(mae_hist_data['Bin Start'], mae_hist_data['Frequency'], width=mae_hist_data['Bin End'] - mae_hist_data['Bin Start'], align='edge',
                color=['green' if x >= 0 else 'red' for x in mae_hist_data['Bin Start']])
    axes[1].set_title('MAE Distribution')
    axes[1].set_xlabel('MAE')
    axes[1].set_ylabel('Frequency')

    # Plot MFE Distribution
    mfe_hist_data = generate_histogram_data(nt_data, "mfe", 50)
    axes[2].bar(mfe_hist_data['Bin Start'], mfe_hist_data['Frequency'], width=mfe_hist_data['Bin End'] - mfe_hist_data['Bin Start'], align='edge',
                color=['green' if x >= 0 else 'red' for x in mfe_hist_data['Bin Start']])
    axes[2].set_title('MFE Distribution')
    axes[2].set_xlabel('MFE')
    axes[2].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plot_day_hour_minute( day_stats, hour_stats, minute_stats):

    fig, axes = plt.subplots(3, 1, figsize=(16, 7))
    sns.barplot(x='entry_day', y='sum', data=day_stats, ax=axes[0], 
                hue='entry_day', dodge=False, palette=['green' if x >= 0 else 'red' for x in day_stats['sum']], legend=False)
    axes[0].set_title('Day Stats')
    axes[0].set_xlabel('Day of the Week')
    axes[0].set_ylabel('Total Profit')

    sns.barplot(x='entry_hour', y='sum', data=hour_stats, ax=axes[1], 
                hue='entry_hour', dodge=False, palette=['green' if x >= 0 else 'red' for x in hour_stats['sum']], legend=False)
    axes[1].set_title('Hour Stats')
    axes[1].set_xlabel('Hour of the Day')
    axes[1].set_ylabel('Total Profit')

    sns.barplot(x='entry_minute', y='sum', data=minute_stats, ax=axes[2],
                hue='entry_minute', dodge=False, palette=['green' if x >= 0 else 'red' for x in minute_stats['sum']], legend=False)
    axes[2].set_title('Minute Stats')
    axes[2].set_xlabel('Minute of the Hour')
    axes[2].set_ylabel('Total Profit')


    plt.tight_layout()
    plt.show()

# Run the main function
if __name__ == "__main__":
    nt_file = 'x.csv'  # NT Trade Data
    nt_data = import_trade_data(nt_file)
    nt_data = process_data(nt_data)

    day_hour_stats, hour_stats, minute_stats, hour_minute_stats, day_stats = calculate_time_day_effectiveness(nt_data)

    # Debugging: Print the data to ensure it's being processed correctly
    print("Profit Distribution:")
    print(generate_histogram_data(nt_data, "profit", 50))
    print("\nMAE Distribution:")
    print(generate_histogram_data(nt_data, "mae", 50))
    print("\nMFE Distribution:")
    print(generate_histogram_data(nt_data, "mfe", 50))
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

    # Generate plots
    plot_histograms(nt_data, day_stats, hour_stats)
    plot_day_hour_minute(day_stats, hour_stats, minute_stats)