import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input, Embedding, Bidirectional,TimeDistributed, Attention 
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

tf.config.set_visible_devices([], 'GPU')

# Read the data into a DataFrame
data = pd.read_csv('data/buildSeqInd_Lucky13_5M_ALL.csv')
data = data.drop(columns=['outputC'])
#data = data.drop(columns=['STOK1'])
#data = data.drop(columns=['RSI'])
#data = data.drop(columns=['ATR2'])
data = data.drop(columns=['ATR21'])
#data = data.drop(columns=['ATR3'])
data = data.drop(columns=['ATR31'])
data = data.drop(columns=['ATR32'])
data = data.drop(columns=['ATR34'])
#data = data.drop(columns=['ROC'])
#data = data.drop(columns=['SDKC9'])
data = data.drop(columns=['SDKC91'])
#data = data.drop(columns=['SDBB91'])
#data = data.drop(columns=['SDLR310'])


#num_columns = len(data.axes[1]) 
#input_features =  num_columns -1
#X = data.iloc[:, 0:input_features]  
#y = data['output'].values


def create_sequences(data_in, seq_length):
    xs, ys = [], []
    for i in range(len(data_in) - seq_length):
        x = data_in.iloc[i:(i + seq_length), :-1]
        y = data_in.iloc[i + seq_length, -1]  # The 'output' column
        xs.append(x.values)
        ys.append(y)
    return np.array(xs), np.array(ys)

def normalize_sequences(sequences_in):
    scalers_out = {}
    for i in range(sequences_in.shape[0]):
        scalers_out[i] = MinMaxScaler((-1,1))
        sequences_in[i] = scalers_out[i].fit_transform(sequences_in[i])
    return sequences_in, scalers_out

# Function to reverse scaling
def reverse_scaling(preds_in, scalers_in, seq_length_in, feature_dim_in):
    reversed_preds = []
    for i in range(len(preds_in)):
        temp_input = np.zeros((seq_length_in, feature_dim_in))
        temp_input[:, -1] = preds_in[i]
        reversed_pred = scalers_in[i].inverse_transform(temp_input)
        reversed_preds.append(reversed_pred[0, -1])
    return np.array(reversed_preds)


timesteps = 60
X, y = create_sequences(data, timesteps)
X, scalers = normalize_sequences(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

scaler = MinMaxScaler()
#scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#X_train = np.array(X_train).reshape(X_train.shape[0], 1, X_train.shape[1])
#X_test = np.array(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])

layer1 = 50
layer2= 50
layer3= 25


# Build the LSTM model
model = Sequential()
model.add(Input(shape = (X_train.shape[1], X_train.shape[2])))
model.add(LSTM(layer1, kernel_initializer='glorot_uniform', return_sequences=True))
model.add(Dropout(0.3))
model.add(Dropout(0.2))
model.add(LSTM(layer2, kernel_initializer='glorot_uniform'))
model.add(Dropout(0.2))
model.add(Dense(layer3))
model.add(Dropout(0.2))
model.add(Dense(1))



model.compile(optimizer='adam',loss='mean_squared_error')

print(" ")
model.summary()
print(" ")

early_stopping = EarlyStopping(monitor='loss',patience=3)

history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.3, callbacks=[early_stopping])

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 9))
fig.suptitle('Vertically stacked subplots')

# Plot training and validation loss
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.grid(True)

# Evaluate the model
val_loss = model.evaluate(X_test, y_test)
print(f'Validation Loss: {val_loss:.4f}')

# Generate predictions
predictions = model.predict(X_test)

# Plot actual vs predicted values
ax2.scatter(y_test, predictions)
ax2.set_xlabel("Actual Output")
ax2.set_ylabel("Predicted Output")
ax2.set_title("Actual vs. Predicted Output")
ax2.grid(True)


plt.show()
