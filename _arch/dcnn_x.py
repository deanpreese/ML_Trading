import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv1D, Reshape, Multiply, LSTM, Dense, Dropout, Flatten, Bidirectional, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping,  ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import l2


from ml_model.data_func  import sequence_and_split3D, sequence_and_split
from ml_model.model_stats import gen_reg_stats_x 


tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)


def create_model(timesteps, features):

    drop_out = 0.4
    l2_reg = l2(0.04)
    initializer = GlorotUniform(seed=42)

    input_shape = (timesteps, features, 1)
    inputs = Input(shape=input_shape)
    
    x = Conv2D(filters=64, kernel_size=4, activation='relu', padding="same")(inputs)
    x = Dropout(drop_out)(x)
    x = Conv2D(filters=32, kernel_size=3, activation='relu', padding="same" )(x)
    x = Dropout(drop_out)(x)
    #x = MaxPooling2D(pool_size=(2,1))(x)
    x = Reshape((input_shape[1], -1))(x)
    
    x = Conv1D(filters=64, kernel_size=2,  activation='relu', kernel_initializer=initializer)(x)       
    #x = LSTM(64, kernel_regularizer=l2_reg, activation='relu', return_sequences=True, kernel_initializer=initializer)(x)
    x = Conv1D(filters=32, kernel_size=2,  activation='relu', kernel_initializer=initializer)(x)
    #x = LSTM(32, kernel_regularizer=l2_reg, activation='relu', return_sequences=True, kernel_initializer=initializer)(x)
    #x = Conv1D(filters=16, kernel_size=2, activation='relu', kernel_initializer=initializer)(x)
    #x = Bidirectional(LSTM(32, kernel_regularizer=l2_reg, return_sequences=True, kernel_initializer=initializer))(x)
    #x = Bidirectional(LSTM(64, kernel_regularizer=l2_reg, return_sequences=True, kernel_initializer=initializer))(x)
    
    #x = MaxPooling1D(pool_size=1, strides=1)(x)
    
    x = Bidirectional(LSTM(32, kernel_regularizer=l2_reg, kernel_initializer=initializer))(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2_reg,  kernel_initializer=initializer)(x) 
    attention_x = Dense(32, activation='softmax', kernel_initializer=initializer, name='attention_x')(x)
    x = Multiply()([x, attention_x])                
        
    x = Dense(8, activation='relu', kernel_regularizer=l2_reg, kernel_initializer=initializer)(x) 
    
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def evaluate_model(y_test, y_pred):
    correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(y_test, y_pred)
    print(f"Val MSE: {mse}, Val MAE: {mae}, R2: {r2}")
    print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.3f}")
    print(f"Number of Samples: {total}")


def main():

    checkpoint_dir = 'checkpoints/'
    trained_dir = 'trained_models/'
    
    checkpoint_model = os.path.join(checkpoint_dir, 'dcnn_x_model.keras')
    trained_model = os.path.join(trained_dir, 'dcnn_x_model.keras')
    dot_img_file = os.path.join(checkpoint_dir, 'dcnn_x.png')


    datafile = [ 
        'data/Lucky13_3070_oos.csv',   
        'data/Lucky13_3070.csv',  #1
        'data/ndata_diff_lucky13_3070_oos.csv', 
        'data/ndata_diff_lucky13_3070.csv', #3
        'data/ndata_lucky_13_lag_3070_oos.csv', 
        'data/ndata_lucky13_lag_3070.csv', #5
        'new_model_Z_lucky13_3070_oos.csv',
        'new_model_Z_lucky13_3070.csv', #7,
        'data/Lucky13_3070_oos_3.csv',   
        'data/Lucky13_3070_3.csv',  #8
        'data/Lucky13_3070_oos_5.csv',   
        'data/Lucky13_3070_5.csv',  #10
        'data/new_model_HLC_lucky13.csv', #11
    ]

    file_path = datafile[1]

    df = pd.read_csv(file_path)
    df = df.drop(columns=['outputC'])
    X = df.drop(columns=['output']).values
    y = df['output'].values

    time_steps = 100
    feature_dims, X_train, X_val, y_train, y_val = sequence_and_split3D(file_path, time_steps)
    
    # Create the model
    model = create_model(time_steps, feature_dims)
    
    #model.compile(optimizer=Adam(learning_rate=0.0009), loss='mse',  metrics=['mae', tf.keras.metrics.R2Score()])
    model.compile(optimizer=Adam(learning_rate=0.00085), loss='mse',  metrics=['mae'])
        
    model.summary(expand_nested=True,show_trainable=True)
    tf.keras.utils.plot_model(model, to_file=dot_img_file,             
        show_shapes=True, 
        show_dtype=False, 
        show_layer_names=True,
        expand_nested=True,
        show_layer_activations=False,
        show_trainable=True)

    # Define early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.25, patience=3, verbose=1,
            #mode="auto", 
            #min_delta=0.000001,
            #cooldown=0, 
            min_lr=0.0001,
        )


    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_model, 
            monitor='val_loss', 
                save_best_only=True, 
                    save_weights_only=False, mode='min')
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                            initial_epoch=0, epochs=150, 
                            batch_size=64, callbacks=[
                                early_stopping,
                                reduce_lr,
                                model_checkpoint])

    y_pred = model.predict(X_val)
    evaluate_model(y_val, y_pred)

    # Plot training history
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
