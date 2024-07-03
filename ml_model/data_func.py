from sklearn.model_selection import train_test_split

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


