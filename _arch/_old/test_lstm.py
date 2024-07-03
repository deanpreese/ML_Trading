import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


# Read the data into a DataFrame
data = pd.read_csv('data/lucky13_oos.csv')
data.drop(columns=['outputC'])

num_columns = len(data.axes[1]) 


input_features =  num_columns - 2
X = data.iloc[:, 0:input_features]  
y = data['output'].values

print(f"Columns   {input_features}")
print(X.shape[0])
print(X.shape[1])
print("   ")

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(X)

model = tf.keras.models.load_model('lstm-90-65-33-4.keras')

for i in range(len(y)):
    
    
    
    pred = model.predict(X.loc[i])
    print(pred) 
    
    #print(X.iloc[i])   