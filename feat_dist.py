
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datafile = [ 
        'data/NewModel_3070_oos.csv',   
        'data/NewModel_3070.csv',  #1
        'data/Model_X_ALL_oos.csv',   
        'data/Model_X_ALL.csv',  #3
        'data/Model_X_3070_oos.csv',  
        'data/Model_X_3070.csv',  #5
        
]   

lucky_13_columns = [
    "SDLR310", "SDBB91", "SDKC91", "SDKC9", "ROC", "ATR54", "ATR53", "ATR52", 
    "ATR51", "ATR5", "ATR21", "ATR2", "RSI", "STOK1", "output", "outputC"
]

model_x_columns = [
    "Year", "Month", "Day", "DayOfWeek", "HourOfDay", "MinOfHour", "SeqClose",
    "SDBB9", "SDBB91", "SDKC9", "SDKC91", "SDBB29", "SDBB291", "SDKC29", "SDKC291",
    "SDBB14CU", "SDBB14CL", "SDBB9CU", "SDBB9CL", "SDKC10CU", "SDKC10CL", "SDKC7CU",
    "SDKC7CL", "ROC14", "ROC9", "ROC7", "ATR14", "ATR9", "ATR5", "ATR2", "RSI14", 
    "RSI9", "ADX14", "ADX9", "STO5135K", "STO5135D", "STO7143K", "STO7143D", "TV1", 
    "TV2", "TV3", "TV4", "TV5", "TV6", "ZH79X", "ZL79X", "ZC79X", "COMP0", "COMP1", 
    "COMP2", "COMP3"
]

data = pd.read_csv(datafile[3])
df = data[['STOK1']]


# Function to generate a formatted, readable text-based distribution summary with percentages and cumulative percentage
def generate_readable_text_distribution_with_cumulative(df):
    distribution_texts = {}
    
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
        distribution_text += "-" * 90 + "\n"
        distribution_text += f"{'Range':<20}{'Count':<10}{'Percentage (%)':<15}{'Cumulative (%)':<15}\n"
        distribution_text += "-" * 90 + "\n"
        
        for i in range(len(counts)):
            # Only display bins with non-zero counts for clarity
            if counts[i] > 0:
                lower_bound = bins[i]
                upper_bound = bins[i + 1]
                count = counts[i]
                percentage = (count / total_count) * 100  # Calculate percentage
                cumulative_percentage += percentage  # Update cumulative percentage
                distribution_text += f"[{lower_bound:.2f} - {upper_bound:.2f}): {count:<10}{percentage:<15.2f}{cumulative_percentage:<15.2f}\n"
        
        # Store the formatted text representation for this column
        distribution_texts[column] = distribution_text

    return distribution_texts

# Generate formatted text-based distributions with cumulative percentages
distribution_summaries = generate_readable_text_distribution_with_cumulative(df)

# Display the results
for column, summary in distribution_summaries.items():
    print(summary)
    print("-" * 90)