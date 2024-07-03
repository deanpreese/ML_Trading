import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, TimeDistributed, Flatten, Dropout, BatchNormalization, LeakyReLU, Bidirectional, Concatenate, MultiHeadAttention
from keras.layers import MultiHeadAttention, LayerNormalization, Add, Reshape
from tensorflow.keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.initializers import RandomNormal

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

tf.config.set_visible_devices([], 'GPU')

def set_seeds(seed=42):
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

# Function to create sequences
def create_sequences(data_in, seq_length_in):
    xs = [data_in.iloc[i:i + seq_length_in, :-1].values for i in range(len(data_in) - seq_length_in)]
    ys = data_in.iloc[seq_length_in:, -1].values
    return np.array(xs), np.array(ys)


# Function to normalize sequences
def normalize_sequences(sequences_in):
    scalers_out = {}
    for i in range(sequences_in.shape[0]):
        scalers_out[i] = MinMaxScaler((0,1)) 
        sequences_in[i] = scalers_out[i].fit_transform(sequences_in[i])
    return sequences_in, scalers_out


def sequence_and_normalize(data_in, seq_length_in):
    
    feature_dim = data_in.shape[1] - 1
    X, y = create_sequences(data_in, seq_length_in)
    X, scalers = normalize_sequences(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return feature_dim, scalers, X_train, X_val, y_train, y_val

def reverse_scaling(preds_in, scalers_in, seq_length_in, feature_dim_in):
    reversed_preds = []
    for i in range(len(preds_in)):
        temp_input = np.zeros((seq_length_in, feature_dim_in))
        temp_input[:, -1] = preds_in[i]
        reversed_pred = scalers_in[i].inverse_transform(temp_input)
        reversed_preds.append(reversed_pred[0, -1])
    return np.array(reversed_preds)



def eval_results(history_in, model_in, X_test_in, y_test_in, scalers_in, timesteps_in, num_features_in):
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 9))

    # Plot training and validation loss
    ax1.plot(history_in.history['loss'], label='Training Loss')
    ax1.plot(history_in.history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    #ax1.set_title('Training and Validation Loss')
    ax1.grid(True)

    #plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    #plt.title('model accuracy')
    #plt.ylabel('accuracy')
    #plt.xlabel('epoch')

    # Evaluate the model
    #val_loss = model_in.evaluate(X_test_in, y_test_in)
    #print(f'Validation Loss: {val_loss:.4f}')

    # Generate predictions
    predictions = model_in.predict(X_test_in)

    # Plot actual vs predicted values

    ups = 0
    dwns = 0

    colors = []
    for i in range(len(predictions)):
        #print(f"Predicted: {predictions[i][0]} Actual: {y_test_in[i]}")
        
        if ((predictions[i][0] < 0 and y_test_in[i] > 0) or (predictions[i][0] > 0 and y_test_in[i] < 0)):
            
            if abs(y_test_in[i]) > 1.0:
                colors.append('red')
                dwns += 1
            else:        
                colors.append('red')
                dwns += 1
        
        elif ((predictions[i][0] > 0 and y_test_in[i] > 0) or (predictions[i][0] < 0 and y_test_in[i] < 0)):
            
            if abs(y_test_in[i]) > 1.0:
                colors.append('green')
                ups += 1
            else:        
                colors.append('green')   
                ups += 1
                           
        else:
            colors.append('white')                 
            
    print("ups: " , ups , "        dwns: " , dwns)        
            
    
    ax2.scatter(predictions , y_test_in , color=colors)
    ax2.set_xlabel("Actual Output")
    ax2.set_ylabel("Predicted Output")
    ax2.grid(True)    
        
    # Reverse the scaling of predictions
    predictions_reversed = reverse_scaling(predictions, scalers_in, timesteps_in, num_features_in )

    # Plot actual vs predicted values
    #ax3.plot(range(len(y_test_in)), y_test_in, color='blue', label='Actual Values')
    #ax3.plot(range(len(predictions_reversed)), predictions_reversed, color='red', linestyle='--', label='Predicted Values')
    
    ax3.plot(range(len(predictions_reversed)), (predictions_reversed - y_test_in), color='red', linestyle='--', label='Predicted Values')
    #ax3.set_title('Actual vs Predicted Values')
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Output')
    ax2.grid(True)

    plt.show()



set_seeds(42)

#train_file = pd.read_csv('data/sm13_3070.csv')
train_file = pd.read_csv('data/buildSeqInd_Lucky13_5M_ALL.csv')
data = train_file.drop(columns=['outputC'])
print("Data Shape:", data.shape)


# Generate sequences and split data
n_steps = 13
feature_dim_out, scalers_out, X_train_out, X_test, y_train_out, y_test = sequence_and_normalize(data, n_steps)
print("Train Shape:", X_train_out.shape)

input_layer = Input(shape=(n_steps, feature_dim_out))
print("Input Shape:", input_layer.shape)

init = RandomNormal(stddev=0.02)

reshaped_input = Reshape((n_steps, feature_dim_out, 1))(input_layer)
conv1 = TimeDistributed(Conv1D(filters=32, kernel_size=n_steps, activation='relu', padding='same'))(reshaped_input)
conv1 = Flatten()(conv1)  
conv1 = Reshape((n_steps, -1))(conv1)

x = BatchNormalization()(conv1)
x = Dropout(0.5)(x)
x = LeakyReLU(negative_slope=0.2)(x)

x = LSTM(15, return_sequences=True, activation='tanh', kernel_initializer=init)(x)
x = MultiHeadAttention(num_heads=2, key_dim=25)(x, x)

x = LeakyReLU(negative_slope=0.2)(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = LSTM(8, return_sequences=False, activation='tanh', kernel_initializer=init)(x)

x = Dense(10, activation=LeakyReLU(negative_slope=0.2))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(50, activation=LeakyReLU(negative_slope=0.2))(x)
x = Dropout(0.3)(x)

output_layer = Dense(1, activation='linear')(x)
model = Model(inputs=input_layer, outputs=output_layer)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.compile(optimizer='adam', loss='mse')
model.summary()

history_out = model.fit(X_train_out, y_train_out, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping])
eval_results(history_out, model, X_test, y_test, scalers_out, n_steps, feature_dim_out)