import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN


def read_data(file_path: str, col_name: str) -> pd.DataFrame:
    """
    Reads a CSV file, filters by col_name, and returns a cleaned DataFrame.
    """
    df = pd.read_csv(file_path)
    df = df[[col_name]].dropna()
    return df


def detect_outliers_std(df: pd.DataFrame, col_name: str, std_threshold: float):
    """
    Computes mean ± X2*STD bounds and returns:
      - lower_bound_Xstd, upper_bound_2std
      - outliers_Xstd (rows outside these bounds)
    """
    mean_val = df[col_name].mean()
    std_val = df[col_name].std(ddof=1)  # Sample standard deviation
    lower_bound_2std = mean_val - std_threshold * std_val
    upper_bound_2std = mean_val + std_threshold * std_val
    outliers_2std = df[(df[col_name] < lower_bound_2std) | (df[col_name] > upper_bound_2std)]
    return lower_bound_2std, upper_bound_2std, outliers_2std


def detect_outliers_iqr(df: pd.DataFrame, col_name: str):
    """
    Computes the IQR bounds (Q1 − 1.5×IQR, Q3 + 1.5×IQR) for col_name and returns:
      - lower_bound_iqr, upper_bound_iqr
      - outliers_iqr (rows outside these bounds)
    """
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound_iqr = Q1 - 1.5 * IQR
    upper_bound_iqr = Q3 + 1.5 * IQR
    outliers_iqr = df[(df[col_name] < lower_bound_iqr) | (df[col_name] > upper_bound_iqr)]
    return lower_bound_iqr, upper_bound_iqr, outliers_iqr


def plot_hist_2std(df: pd.DataFrame, col_name: str,
                   lower_bound_2std: float, upper_bound_2std: float):
    """
    Plots a histogram of col_name with vertical lines for mean ± 2*STD.
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(df[col_name], bins=30, kde=True, color='blue', alpha=0.6)
    plt.axvline(lower_bound_2std, color='red', linestyle='--', label='2 STD Lower Bound')
    plt.axvline(upper_bound_2std, color='red', linestyle='--', label='2 STD Upper Bound')
    plt.title(f"Histogram of {col_name} with Mean ± 2*STD Outlier Boundaries")
    plt.legend()
    plt.show()


def plot_box_iqr(df: pd.DataFrame, col_name: str,
                 lower_bound_iqr: float, upper_bound_iqr: float):
    """
    Plots a boxplot of col_name with vertical lines for IQR fences.
    """
    plt.figure(figsize=(6, 5))
    sns.boxplot(x=df[col_name], color='green')
    plt.axvline(lower_bound_iqr, color='orange', linestyle='--', label='IQR Lower Fence')
    plt.axvline(upper_bound_iqr, color='orange', linestyle='--', label='IQR Upper Fence')
    plt.title(f"Boxplot of {col_name} with IQR Outlier Fences")
    plt.legend()
    plt.show()


def main():
    
    lucky13_all = [
                    'SDLR310','SDBB91',
                    'SDKC91','SDKC9',
                    'ROC',
                    'ATR54','ATR53',
                    'ATR52','ATR51','ATR5',
                    'ATR21',
                    'ATR2','RSI','STOK1'
                ]
        
    model_m_1_all = [
    #"Year", "Month", "Day", "DayOfWeek", "HourOfDay", "MinOfHour",
    #"SeqClose", "RSIRAW", 
    "SDBB9L", "SDBB9U", "SDKC7U", "SDKC7L",
    "ROC14", "ROC9", "ATR5", "ATR2", "RSI14", "RSI14Avg", "RSIH14", "RSIL14",
    "RSI9", "RSI9Avg", "RSIH9", "RSIL9", "STOK721", "STOK513", "STOK721D", "STOK513D",
    "TV1", "TV2", "TV3", "TV4", "TV5", "TV6", "COMP0", "COMP1", "COMP2", "COMP3"
    ]
    
    
    # Adjust paths and column name as needed
    datafile = [
        'data/Lucky13_3070.csv',   # index 0
        'data/Model_M_1_3070.csv' # index 1
    ]
    col_name = 'RSI'

    # 1. Read Data
    df = read_data(datafile[0], col_name)
    print(f"Data loaded: {len(df)} rows, column = {col_name}")

    # 2. Mean ± 2 STD
    std_threshold = 2.0
    lower_2std, upper_2std, outliers_2std = detect_outliers_std(df, col_name, std_threshold)
    print(f"\n=== Mean ± {std_threshold}*STD method ===")
    print(f"Lower bound: {lower_2std:.4f}, Upper bound: {upper_2std:.4f}")
    print(f"Number of outliers: {len(outliers_2std)}")
    print(f"Percent of outliers: {len(outliers_2std)/len(df)}")

    # 3. IQR
    lower_iqr, upper_iqr, outliers_iqr = detect_outliers_iqr(df, col_name)
    print("\n=== IQR method ===")
    print(f"Lower bound: {lower_iqr:.4f}, Upper bound: {upper_iqr:.4f}")
    print(f"Number of outliers: {len(outliers_iqr)}")
    print(f"Percent of outliers: {len(outliers_iqr)/len(df)}")


    # Visualizations
    #sns.set(style="whitegrid")
    #plot_hist_2std(df, col_name, lower_2std, upper_2std)
    #plot_box_iqr(df, col_name, lower_iqr, upper_iqr)

if __name__ == "__main__":
    main()
