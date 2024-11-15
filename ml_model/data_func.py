import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler



def generate_combinations(data_list, min=3, max=7):
    all_combinations = []
    for r in range(min, max + 1):
        combinations = list(itertools.combinations(data_list, r))
        all_combinations.extend(combinations)
    return all_combinations

def simple_split_and_scale(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

#split_three_ways_full(X, y, run_test_size, 0.2, 42)
def split_three_ways_full(X,y,run_test_size, val_size, random_v ):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=run_test_size, random_state=random_v)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_v)
    return X_train, X_val, X_test, y_train, y_val, y_test



def split_three_ways(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def simple_split_and_scale(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train.columns = X_train.columns.str.replace('[^+a-zA-Z0-9]', '_')
    X_test.columns = X_test.columns.str.replace('[^+-a-zA-Z0-9]', '_')
    return X_train, X_test, y_train, y_test


def full_split_and_scale(pd_data, col_offset, size_test, random_state, output_col):
    num_columns = len(pd_data.axes[1]) 
    input_features =  num_columns - col_offset
    X = pd_data.iloc[:, 0:input_features]  
    y = pd_data[output_col].values
    X_train, X_test, y_train, y_test  =  simple_split_and_scale(X, y, size_test, random_state)
    
    return X_train, X_test, y_train, y_test, input_features


def create_sequences(df, seq_length):
    print("Create Sequences")
    xs, ys = [], [] 
    
    for i in range(len(df) - seq_length):
        x = df.iloc[i:(i + seq_length), :-1]
        y = df.iloc[i + seq_length, -1]  
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def normalize_sequences(sequences):
    print("Normalize sequences")
    scalers_out = {}
    for i in range(sequences.shape[0]):
        scalers_out[i] = MinMaxScaler((-1,1))
        sequences[i] = scalers_out[i].fit_transform(sequences[i])
    return sequences, scalers_out

def scale_sequences(sequences):
    print("Scale sequences")
    scalers_out = {}
    for i in range(sequences.shape[0]):
        scalers_out[i] = StandardScaler()
        sequences[i] = scalers_out[i].fit_transform(sequences[i])
    return sequences, scalers_out


# Function to reverse scaling
def reverse_scaling(preds, scalers, seq_length, feature_dim):
    reversed_preds = []
    for i in range(len(preds)):
        temp_input = np.zeros((seq_length, feature_dim))
        temp_input[:, -1] = preds[i]
        reversed_pred = scalers[i].inverse_transform(temp_input)
        reversed_preds.append(reversed_pred[0, -1])
    return np.array(reversed_preds)

def sequence_and_normalize(file_path, timesteps):
    
    df = pd.read_csv(file_path)
    df = df.drop(columns=['outputC'])
    X = df.drop(columns=['output'])
    featrure_dims = len(X.columns)
    
    X, y = create_sequences(df, timesteps)
    X, scalers = normalize_sequences(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return featrure_dims, X_train, X_test, y_train, y_test, scalers


def sequence_and_split(file_path, timesteps):
    
    df = pd.read_csv(file_path)
    df = df.drop(columns=['outputC'])
    X = df.drop(columns=['output'])
    feature_dims = len(X.columns)
    
    X, y = create_sequences(df, timesteps)
    X, scalers = scale_sequences(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return feature_dims, X_train, X_test, y_train, y_test


def sequence_and_split3D(file_path, timesteps):
    
    df = pd.read_csv(file_path)
    df = df.drop(columns=['outputC'])
    X = df.drop(columns=['output'])
    feature_dims = len(X.columns)
    
    X, y = create_sequences(df, timesteps)
    
    X, scalers = scale_sequences(X)
    #X, scalers = normalize_sequences(X)
    """
    The reshaped data has the shape (number of samples, number of time steps, number of features, 1)
    , where:
    X_train.shape[0] is the number of samples in the training set.
    X_train.shape[1] is the number of time steps (sequence length).
    X_train.shape[2] is the number of features per time step.
    1 is the single channel dimension.
    """
    
    X = X.reshape((X.shape[0], timesteps, feature_dims, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return feature_dims, X_train, X_test, y_train, y_test