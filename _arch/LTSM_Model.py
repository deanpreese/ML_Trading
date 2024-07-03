import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input, Embedding, Bidirectional,TimeDistributed
from sklearn.ensemble import RandomForestRegressor
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss',patience=5)
from common_func import ShowImportances, show_stats, calc_MSE
from sklearn.metrics import mean_squared_error

#data = pd.read_csv("data/Seq_30_R2080_TAG.csv")   #83%
#data = pd.read_csv("data/Seq_30_R3070_TAG.csv")   #75%   
#data = pd.read_csv("data/Seq_TopSet_R3070_TAG.csv")   #  
data = pd.read_csv("data/Seq_Ind_R3070_TAG.csv")   #  


num_columns = len(data.axes[1]) 
input_features =  num_columns -1
X = data.iloc[:, 0:input_features]  
y = data['output'].values

num_epocs = 100
run_batch_size = 32
run_test_size = 0.5
units=input_features

#X = data[['feature1', 'feature11','feature14','feature21', 'feature23', 'feature33', 'feature34', 'feature36', 'feature4', 'feature47'  ]].values
#y = data['output'].values
#input_features = 10

#X = data[['feature18','feature17','feature16','feature14','feature9','feature8','feature7','feature3','feature2','feature1']].values
#y = data['output'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=run_test_size, random_state=0)

print("Number of TRAIN patterns:", X_train.shape[0])
print("Size of input patterns:", X_train.shape[1])
print(" ")
print("Number of TEST patterns:", X_test.shape[0])
print("Size of input patterns:", X_test.shape[1])
print(" ")

#sys.exit()

input_features = X_train.shape[1]

# Rescale the data to the range -1 to 1
scaler = MinMaxScaler(feature_range=(-1, 1))
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)
X_train = np.array(X_train).reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = np.array(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])

units = 53
regression= Sequential()
regression.add(LSTM(units=units,return_sequences=True,kernel_initializer='glorot_uniform',input_shape=(1,input_features)))
regression.add(Dropout(0.2))
regression.add(LSTM(units=units,kernel_initializer='glorot_uniform',return_sequences=True))
regression.add(Dropout(0.2))
regression.add(LSTM(units=units,kernel_initializer='glorot_uniform',return_sequences=True))
regression.add(Dropout(0.2))
regression.add(LSTM(units=units,kernel_initializer='glorot_uniform'))
regression.add(Dropout(0.2))
regression.add(Dense(units=7))
regression.add(Dropout(0.2))
regression.add(Dense(units=1))


model = regression

model.compile(optimizer='adam',loss='mean_squared_error')

print(" ")
model.summary()
print(" ")

history = model.fit(X_train, y_train, epochs=num_epocs, batch_size=run_batch_size, validation_split=0.1, callbacks=[early_stopping])


predictions = model.predict(X_test)
#ShowImportances(data.columns[:input_features],model.feature_importances_, False)
show_stats(False, y_test, predictions)

#RunMSE(y_test, predictions)

# Plot loss and accuracy during training
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

