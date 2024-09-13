import pandas as pd
import numpy as np
import seaborn as sns
import visualkeras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, LSTMCell, Dropout, Input,StackedRNNCells, RNN,  Bidirectional, Attention, BatchNormalization
from tensorflow.keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import glob

tf.config.set_visible_devices([], 'GPU')

def review_data(data_co):
    plt.figure(figsize=(16,8))
    #sns.heatmap(data_co.corr(),cmap="YlGnBu",square=False,linewidths=.2,center=0)
    sns.heatmap(data_co.corr(),cmap=sns.cubehelix_palette(as_cmap=True))
        
    plt.show()


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

# Function to reverse scaling
def reverse_scaling(preds_in, scalers_in, seq_length_in, feature_dim_in):
    reversed_preds = []
    for i in range(len(preds_in)):
        temp_input = np.zeros((seq_length_in, feature_dim_in))
        temp_input[:, -1] = preds_in[i]
        reversed_pred = scalers_in[i].inverse_transform(temp_input)
        reversed_preds.append(reversed_pred[0, -1])
    return np.array(reversed_preds)


def sequence_and_normalize(data_in, seq_length_in):
    
    feature_dim = data_in.shape[1] - 1
    X, y = create_sequences(data_in, seq_length_in)
    X, scalers = normalize_sequences(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return feature_dim, scalers, X_train, X_val, y_train, y_val

def eval_win_loss(history_in, model_in, X_test_in, y_test_in, scalers_in, timesteps_in, num_features_in):
    
    ups = 0
    dwns = 0
    
    predictions = model_in.predict(X_test_in)
    for i in range(len(predictions)):
        #print(f"Predicted: {predictions[i][0]} Actual: {y_test_in[i]}")
        
        if ((predictions[i][0] < 0 and y_test_in[i] > 0) or (predictions[i][0] > 0 and y_test_in[i] < 0)):
            dwns += 1
        
        elif ((predictions[i][0] > 0 and y_test_in[i] > 0) or (predictions[i][0] < 0 and y_test_in[i] < 0)):
            ups += 1
                        
            
    print("ups: " , ups , "        dwns: " , dwns)        
    
    return ups, dwns


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
            
            colors.append('red')
            dwns += 1
        
        elif ((predictions[i][0] > 0 and y_test_in[i] > 0) or (predictions[i][0] < 0 and y_test_in[i] < 0)):
            
            colors.append('green')   
            ups += 1
                           
        else:
            colors.append('black')                 
            
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
    

def load_and_predict_oos(file_path, model_in, seq_length, features):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['outputC'])

    X, y = create_sequences(df, seq_length)
    X, scalers_r = normalize_sequences(X)

    predictions = model_in.predict(X)
    predictions_reversed = reverse_scaling(predictions, scalers_r, seq_length, features)

    # Plot actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y)), y, color='blue', label='Actual Values')
    plt.plot(range(len(predictions_reversed)), predictions_reversed, color='red', linestyle='--', label='Predicted Values')
    plt.title(f'Actual vs Predicted Values for {file_path}')
    plt.xlabel('Index')
    plt.ylabel('Output')
    plt.legend()
    plt.show()



def train_modelX(X_train, y_train, X_val, y_val, time_steps_in, epocs, batch):
   
    units=35
    dropout_rate=0.6
    learning_rate=0.00001
  
    # Define the LSTM model
    model = Sequential()
    model.add(Input(shape=(time_steps_in, X_train.shape[2])))
    
    model.add(LSTM(units , return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    #model.add(Dropout(dropout_rate))
    
    #model.add(Bidirectional(LSTM(units ,return_sequences=True, activation='tanh'))) 
    model.add(Bidirectional(LSTM(units //2 ,return_sequences=True)))
    
    #model.add(Dropout(dropout_rate))
    #model.add(BatchNormalization())

    model.add(LSTM(units // 2, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    #model.add(Dropout(dropout_rate))
    #odel.add(BatchNormalization())
    model.add(Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01)))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
   
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    history_out = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epocs, batch_size=batch, callbacks=[early_stopping, reduce_lr])
    
    return model, history_out



def train_model(X_train, y_train, X_val, y_val, time_steps_in, epocs, batch):
   
    layer1 = 30
    layer2 = 20
    
    dropout_rate=0.6
    learning_rate=0.001
    
    model = Model()
    inputs = Input(shape=(time_steps_in, X_train.shape[2]))
    
    #att = Attention()([inputs, inputs])
    
    b1 = Bidirectional(LSTM(layer1 ,return_sequences=True, activation='tanh'))(inputs)
    #dp0 = Dropout(dropout_rate)(b1)
    #bn = BatchNormalization()(dp0)    
    
    lstm1 = LSTM(layer2,activation='tanh')(b1)    
    
    dp0 = Dropout(dropout_rate)(lstm1)
    
    #lstm2 = LSTM(layer2 //2,activation='tanh')(lstm1)
    output = Dense(1)(dp0) # Change activation and size based on your problem
        
    model = Model(inputs=inputs, outputs=output)
   
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    history_out = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epocs, batch_size=batch, callbacks=[early_stopping, reduce_lr])

    return model, history_out




# -----------------------------------------------------------------------


#file_loaded = pd.read_csv("data/IND_LSTM_ALL.csv")
#file_loaded = pd.read_csv('data/sm13_3070.csv')
file_loaded = pd.read_csv('data/buildSeqInd_Lucky13_5M_ALL.csv')
data_loaded = file_loaded.drop(columns=['outputC'])

time_steps = 15
epocs_to_run = 100
batch_size_to_run = 32

feature_dim_out, scalers_out, X_train_out, X_test, y_train_out, y_test = sequence_and_normalize(data_loaded, time_steps)
model_result, history = train_modelX(X_train_out, y_train_out, X_test, y_test, time_steps, epocs_to_run, batch_size_to_run)
eval_results(history, model_result, X_test, y_test, scalers_out, time_steps, feature_dim_out)

#oos_file = 'data/lucky13_oos.csv'
#load_and_predict_oos(oos_file, model_result, time_steps, feature_dim_out)

