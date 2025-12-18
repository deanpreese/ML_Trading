
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def pareto_cutoffs(data, print_rows=True):
    
    data = data.astype(np.float32)
    data = data.values.flatten()
    
    
    total_count = len(data)
    if total_count == 0:
        print("No valid data in column; cannot proceed.")
        return None

    # --- 2. Build histogram (counts only) ---
    hist_count, bin_edges = np.histogram(data, bins=100)
    # hist_count[i] is how many data points fall in [bin_edges[i], bin_edges[i+1])

    # --- 3. Cumulative counts ---
    cumulative_counts = np.cumsum(hist_count)

    # --- 4. Function for linear interpolation of cutoff values ---
    def find_cutoff_value(cutoff_fraction):
        """
        cutoff_fraction: (e.g. 0.10 for 10%, 0.90 for 90%)
        Returns the numeric value at which the cumulative distribution
        crosses cutoff_fraction * total_count.
        Uses linear interpolation in the bin where that fraction lies.
        """

        target_count = cutoff_fraction * total_count
        # Find bin i where cumsum >= target_count
        i = np.searchsorted(cumulative_counts, target_count, side="left")

        # Edge cases
        if i == 0:
            # If the fraction is below or within the first bin
            # we'll interpolate within bin 0, if it has any points
            if hist_count[0] == 0:
                # If the first bin is empty, just return its start
                return bin_edges[0]
        elif i >= len(hist_count):
            # If we are beyond the last bin, just return the last edge
            return bin_edges[-1]

        # Values for bin i
        c_prev = cumulative_counts[i - 1] if i > 0 else 0
        in_bin = hist_count[i]
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]

        if in_bin == 0:
            # If this bin is empty, no interpolation possible; return start
            return bin_start

        needed_from_bin = target_count - c_prev
        frac_in_bin = needed_from_bin / in_bin
        cutoff_val = bin_start + frac_in_bin * (bin_end - bin_start)
        return cutoff_val

        # If total_count == 0, the entire function would be moot,
        # but we've already handled that at the top.

    # Compute 10% and 90% cutoffs
    cutoff_10 = find_cutoff_value(0.10)
    cutoff_90 = find_cutoff_value(0.90)

    # --- 5. Mean and Standard Deviation / ±1σ and ±2σ ---
    mean_val = np.mean(data)
    std_val = np.std(data, ddof=1)  # sample std dev (N-1 in denominator)

    std_2_above = mean_val + 2 * std_val
    std_2_below = mean_val - 2 * std_val
    std_x_above = mean_val + 3 * std_val
    std_x_below = mean_val - 3 * std_val

    # --- 6. Print histogram table ---
    print(f"{'Bin':>4} | {'Range Start':>12} | {'Range End':>12} | {'Count':>8} "
          f"| {'% of Total':>10} | {'Cumul.Count':>12} | {'Cumul.%':>9}")
    print("-" * 79)

    for i in range(len(hist_count)):
        bin_start = bin_edges[i]
        bin_end   = bin_edges[i + 1]
        count     = hist_count[i]
        cume      = cumulative_counts[i]

        bin_pct = (count / total_count) * 100
        cume_pct = (cume / total_count) * 100
        print(f"{i:4d} | "
              f"{bin_start:12.4f} | "
              f"{bin_end:12.4f} | "
              f"{count:8d} | "
              f"{bin_pct:10.2f} | "
              f"{cume:12d} | "
              f"{cume_pct:9.2f}")

    # Print summary
    print("\n--- Summary ---")
    print(f"Total count: {total_count}")
    print(f"Min value:   {bin_edges[0]:.4f}")
    print(f"Max value:   {bin_edges[-1]:.4f}")
    print(f"Mean:        {mean_val:.4f}")
    print(f"Std Dev:     {std_val:.4f}")

    # Cutoff Values
    print("\n--- Cutoff Values ---")
    print(f"10% cutoff (cumulative) value: {cutoff_10:.4f}")
    print(f"90% cutoff (cumulative) value: {cutoff_90:.4f}")

    # ±1σ and ±2σ
    print("\n--- ± Standard Deviations ---")
    print(f"2σ below: {std_2_below:.4f}   2σ above: {std_2_above:.4f}")
    print(f"Xσ below: {std_x_below:.4f}   Xσ above: {std_x_above:.4f}")

    print(" ")

    # Return values if further analysis is desired
    return {
        #"hist_count": hist_count,
        #"bin_edges": bin_edges,
        #"cumulative_counts": cumulative_counts,
        "cutoff_10": cutoff_10,
        "cutoff_90": cutoff_90,
        "mean": mean_val,
        "std_dev": std_val,
        "std_2_below": std_2_below,
        "std_2_above": std_2_above,
        "std_x_below": std_x_below,
        "std_x_above": std_x_above,
    }


if __name__ == "__main__":
    # Example usage:


    datafile = [ 
            'data/Lucky13_3070.csv',  #0
            'data/Model_M_1_3070.csv' #1
    ] 
    

    lucky_13_columns = [
        "SDLR310", "SDBB91", "SDKC91", "SDKC9", "ROC", "ATR54", "ATR53", "ATR52", 
        "ATR51", "ATR5", "ATR21", "ATR2", "RSI", "STOK1", "output", "outputC"
    ]

    model_m_1_columns = [
    #"Year", "Month", "Day", "DayOfWeek", "HourOfDay", "MinOfHour",
    #"SeqClose", "RSIRAW", 
    "SDBB9L", "SDBB9U", "SDKC7U", "SDKC7L",
    "ROC14", "ROC9", "ATR5", "ATR2", "RSI14", "RSI14Avg", "RSIH14", "RSIL14",
    "RSI9", "RSI9Avg", "RSIH9", "RSIL9", "STOK721", "STOK513", "STOK721D", "STOK513D",
    "TV1", "TV2", "TV3", "TV4", "TV5", "TV6", "COMP0", "COMP1", "COMP2", "COMP3"
    ]
        

    data = pd.read_csv(datafile[1])
    df = data [['RSI9']]
    
    results = pareto_cutoffs(df, print_rows=True)

    print(results)  


"""

Model M 1

Bell Curve High and Low Boundaries 
SDBB9L  4.99  91.76 
SDBB9U  10.08  94.01 
SDKC7U  6.00  95.00 
SDKC7L  5.00  93.98 
ROC14  24.90  72.77 
ROC9  28.33  69.88 
ATR5  35.48  79.72 
ATR2  41.11  68.96 
RSI14  25.55  73.81 
RSI14Avg  30.22  70.43 
RSIH14  22.96  76.85 
RSIL14  23.25  77.49 
RSI9  20.47  78.94 
RSI9Avg  23.33  76.84 
RSIH9  17.00  82.00 
RSIL9  17.78  82.99 
STOK721  10.00  91.00 
STOK513  10.00  91.00 
STOK721D  14.00  89.00 
STOK513D  13.00  89.00 
TV1  -9.08  6.69 
TV2  -13.70  11.74 
TV3  -14.28  8.72 
TV4  -4.37  3.10 
TV5  -10.01  7.31 
TV6  -10.26  4.81 
COMP0  0.18  0.82 
COMP1  0.18  0.82 
COMP2  0.16  0.84 
COMP3  0.08  0.93 
"""  