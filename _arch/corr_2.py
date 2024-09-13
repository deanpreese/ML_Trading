import pandas as pd

# Load the dataset

#file_loaded = pd.read_csv('data/sm13_3070.csv')
file_loaded = pd.read_csv('data/buildSeqInd_Lucky13_5M_ALL.csv')
data_loaded = file_loaded.drop(columns=['outputC'])

output_column = 'output'  

df = data_loaded

# Ensure the output column is in the dataframe
if output_column not in df.columns:
    raise ValueError(f"The specified output column '{output_column}' does not exist in the dataset.")

# Initialize a DataFrame to store rolling correlations
rolling_correlations_1000 = pd.DataFrame()

window_size = 1000

# Initialize list to store the results
positive_correlation_features = []

# Iterate over the rows starting from row 1001
for start in range(1000, len(df)):
    end = start + window_size
    if end > len(df):
        break

    current_window = df.iloc[start:end]
    
    # Calculate rolling correlation for each feature against the output
    for feature in df.columns:
        if feature != output_column:
            rolling_correlation = current_window[feature].corr(current_window[output_column])
            
            # Check for correlation greater than zero
            if rolling_correlation > 0:
                positive_correlation_features.append((start, feature, rolling_correlation))

# Convert the results to a DataFrame for easier handling and saving
results_df = pd.DataFrame(positive_correlation_features, columns=['Start_Row', 'Feature', 'Correlation'])

# Output the features with positive correlations
print("Features with positive correlations at each rolling window:")
print(results_df)

# Save the results to a new CSV file
results_df.to_csv('positive_correlations_features_stepped.csv', index=False)

