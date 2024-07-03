import pandas as pd
from darts import TimeSeries
from darts.models import NBEATSBlock
from darts.metrics import mse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv("data/buildSeqInd_Lucky13_5M_ALL.csv")

# Drop the 'outputC' column (assuming it's not relevant for regression)
data = data.drop(columns=['outputC'])

# Separate features (X) and target variable (y)
X_data = data.drop(columns=['output'])  # Features
y_data = data['output']

# Preprocessing steps (replace with your specific needs)
categorical_features = X_data.select_dtypes(include=["object"])
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(categorical_features)

scaler = StandardScaler()
scaled_numerical_features = scaler.fit_transform(X_data.select_dtypes(include=[np.number]))

preprocessed_features = np.concatenate((encoded_features, scaled_numerical_features), axis=1)

# Create a Darts TimeSeries object
ts = TimeSeries.from_dataframe(data=data, time_col="timestamp",  # Replace with your timestamp column name
                               value_cols="output",  # Replace with your target variable name if different
                               freq="infer")  # Automatically infer frequency (e.g., daily, hourly)

# Split data into training and testing sets
train, test = ts.split(test_size=0.2, random_state=42)

# N-BEATS Model using Darts
model = NBEATSBlock(
    past_window=2,  # Adjust based on desired historical window (replace with your actual window size)
    future_window=1,  # Predicting 1 step ahead
    hidden_dim=16,  # Adjust hyperparameters as needed
    num_stacks=1,
    share_weights_in_stack=True
)

# Train the model
model.fit(train, verbose=True)  # Set verbose to True for training progress updates

# Prediction on test set
predictions = model.predict(test)

# Calculate Mean Squared Error (MSE)
metric = mse(test, predictions)
print("N-BEATS MSE:", metric)

# Plot actual vs predicted values (for visualization)
plt.figure(figsize=(10, 6))
plt.plot(test.values, label="Actual")
plt.plot(predictions, label="N-BEATS")
plt.legend()
plt.show()

# ... (Implement ensemble logic and further analysis)

