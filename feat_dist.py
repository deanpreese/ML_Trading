
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
#df = data[['ATR2',"ATR5"]]
df = data [model_m_1_columns]

# Function to generate a formatted, readable text-based distribution summary with percentages and cumulative percentage
def generate_readable_text_distribution_with_cumulative(df):
    
    distribution_texts = {}
    distribution_bounds = {}
    
    distribution_bounds = f"\nBell Curve High and Low Boundaries \n"
        
    
    for column in df.columns:
        # Get min and max for the column
        min_val = df[column].min()
        max_val = df[column].max()
        
        # Generate 100 bins from min to max
        bins = np.linspace(min_val, max_val, 101)  # 101 edges define 100 bins
        
        # Get counts per bin
        counts, _ = np.histogram(df[column], bins=bins)
        total_count = counts.sum()  # Total number of entries in the column
        
        # Initialize cumulative percentage
        cumulative_percentage = 0
        
        # Generate a formatted text representation
        distribution_text = f"\nDistribution of '{column}':\n"
        distribution_text += "-" * 60 + "\n"
        distribution_text += f"{'Range':<20}{'Count':<10}{'Percentage (%)':<15}{'Cumulative (%)':<15}\n"
        distribution_text += "-" * 60 + "\n"
        
        lv_lower = 0
        lv_upper = 0
        
        for i in range(len(counts)):
            # Only display bins with non-zero counts for clarity
            if counts[i] > 0:
                lower_bound = bins[i]
                upper_bound = bins[i + 1]
                count = counts[i]
                percentage = (count / total_count) * 100  # Calculate percentage
                cumulative_percentage += percentage  # Update cumulative percentage
                
                if cumulative_percentage <= 10:
                    lv_lower = upper_bound
                
                if cumulative_percentage <= 90:
                    lv_upper = upper_bound
                
                distribution_text += f"[{lower_bound:.2f} - {upper_bound:.2f}): {count:<10}{percentage:<15.2f}{cumulative_percentage:<15.2f}\n"
                
                
        distribution_bounds += f"{column}  {lv_lower:.2f}  {lv_upper:.2f} \n"        
        
        # Store the formatted text representation for this column
        distribution_texts[column] = distribution_text

    return distribution_texts, distribution_bounds

# Generate formatted text-based distributions with cumulative percentages
distribution_summaries, distribution_bounds  = generate_readable_text_distribution_with_cumulative(df)

# Display the results
#for column, summary in distribution_summaries.items():
#    print(f" {summary}")
#    print("-" * 60)
    
print(distribution_bounds) 

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