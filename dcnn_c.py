import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib 
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Average, Multiply, GlobalAveragePooling1D, Reshape, Concatenate, ConvLSTM1D, Flatten, SeparableConv1D, LayerNormalization, Bidirectional, Add, Dense,  Dropout, MaxPooling1D, LSTM, MultiHeadAttention, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

from ml_model.model_stats import gen_reg_stats_x, gen_class_stats 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)


class DCNN_C:
    def __init__(self):
        
        self.checkpoint_dir = 'checkpoints/'
        self.trained_dir = 'trained_models/'
       
        self.checkpoint_model = os.path.join(self.checkpoint_dir, 'dcnn_c_model.keras')
        self.trained_model = os.path.join(self.trained_dir, 'dcnn_c_model.keras')
        self.dot_img_file = os.path.join(self.checkpoint_dir, 'dcnn_c.png')

        self.drop_out = 0.3
        self.l2_reg = l2(0.01)
        self.initializer = GlorotUniform(seed=42)        
    
    
    def build_model_h(self, inputs):

        h = Conv1D(filters=64, kernel_size=4, activation='relu', kernel_initializer=self.initializer)(inputs) 
        h = Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer=self.initializer)(h)
        h = Conv1D(filters=16, kernel_size=2, activation='relu', kernel_initializer=self.initializer)(h)
        h = Bidirectional(LSTM(32,name="BIC", kernel_regularizer=self.l2_reg, return_sequences=True, kernel_initializer=self.initializer))(h)
        h = Bidirectional(LSTM(32,name="BIC2", kernel_regularizer=self.l2_reg, kernel_initializer=self.initializer))(h)

        h = Dense(32, activation='relu', kernel_regularizer=self.l2_reg,  kernel_initializer=self.initializer)(h)         
        attention_h = Dense(32, activation='softmax', kernel_initializer=self.initializer, name='attention_h')(h)
        h = Multiply()([h, attention_h])                
        h = Dense(8, activation='relu', kernel_regularizer=self.l2_reg, name="h_out", kernel_initializer=self.initializer)(h)           
        
        return h        
                
    
    def build_model(self, input_shape):
        
        inputs = Input(shape=input_shape)
        h = self.build_model_h(inputs)
        ave_output = h
        outputs = Dense(1, activation='sigmoid')(ave_output)
        model = Model(inputs=inputs, outputs=outputs)
        self.model = model
        return model        
                
        
    def train_model(self, input_shape, X_train, X_test, y_train, y_test ):

        model = self.build_model(input_shape)
    
        loss_function = 'binary_crossentropy'
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss_function, metrics=['accuracy'])
        model.summary(expand_nested=True, show_trainable=True)
            
        tf.keras.utils.plot_model(model, to_file=self.dot_img_file, 
            show_shapes=True, 
            show_dtype=True,
            show_layer_names=True,
            expand_nested=True,
            show_layer_activations=True,
            show_trainable=True
            )
            
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.2,
            patience=5, verbose=1,
            mode="auto", min_delta=0.000001,
            cooldown=0, min_lr=0,
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_model, 
                monitor='val_loss', 
                    save_best_only=True, 
                        save_weights_only=False, mode='min')
        
        history_out = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                                initial_epoch=0, epochs=3000, 
                                batch_size=32, callbacks=[
                                    early_stopping,
                                    reduce_lr,
                                    model_checkpoint])
        
        y_pred = model.predict(X_test)
        return history_out, y_pred, y_test, X_test
    
    
def evaluate_model( y_pred, y_test):
    
    y_pred_classes = (y_pred > 0.5).astype("int32").flatten()
    y_true_classes = y_test.flatten()

    print("Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_classes, y_pred_classes))
    
    perf, correct, total, tn, fp, fn, tp, mse, rmse, mae, r2 = gen_class_stats( y_test, y_pred)
    
    print(" ")
    print(f"Pred MSE: {mse},  MAE: {mae}, R2: {r2}")
    print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.4f}")
    print(f"Number of Samples: {total}")        
    print(" ")    
    
    return mse, mae
        
        
        
def run():

    datafile = [ 
        'data/Lucky13_3070_oos.csv',   
        'data/Lucky13_3070.csv',  #1
        'data/Lucky13_EX_3070_oos.csv',  
        'data/Lucky13_EX_3070.csv',  #3
    ]

    model = DCNN_C()

    train = True
    test = False
    single_item = False


    if train:
        
        file_path = datafile[1]
        print(f"Loading {file_path}" )
        df = pd.read_csv(file_path)
        df = df.drop(columns=['output'])  # Drop regression target
            
        #df = df[(df['RSI'] > 60) & (df['RSI'] < 80)]  #  
        #df = df[(df['RSI'] > 60) & (df['RSI'] < 75)]  #  
        #df = df[(df['RSI'] > 20) & (df['RSI'] < 40)]  #  
        #df = df[(df['RSI'] > 25) & (df['RSI'] < 40)]  #  
            
        X = df.drop(columns=['outputC']).values
        y = df['outputC'].values

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y)
        num_classes = len(np.unique(y))
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {le.classes_}")

        # One-hot encode labels if multiclass classification
        if num_classes > 2:
            y = to_categorical(y, num_classes)
    
        X = X.reshape(X.shape[0], X.shape[1], 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        
        for i in range(5):
            file_path = datafile[1]
            history_out, y_pred, y_test, X_test = model.train_model(input_shape, X_train, X_test, y_train, y_test )
            mse, mae = evaluate_model( y_pred, y_test)
            
            model_file = f"dcnn_c_{mse}_{mae}_model.keras"
            file_path = os.path.join(model.checkpoint_dir, model_file)
            model.model.save(file_path)
        

    if test:
        file_path = datafile[0]
        
        
        df = pd.read_csv(file_path)
        df = df.drop(columns=['output'])
        X = df.drop(columns=['outputC']).values
        y = df['outputC'].values 

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y)
        num_classes = len(np.unique(y))

        if num_classes > 2:
            y = to_categorical(y, num_classes)
               
        X_test = X.reshape(X.shape[0], X.shape[1], 1)
        y_test = y
        model.evaluate_model()
        
        
        
        model.load_saved_model("train")
        model.run_batch_test(file_path)

    if single_item:
        file_path = datafile[0]
        model.load_saved_model("run")

        df = pd.read_csv(file_path)
        df = df.drop(columns=['output'])
        X = df.drop(columns=['outputC']).values
        y = df['outputC'].values 

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y)
        num_classes = len(np.unique(y))

        if num_classes > 2:
            y = to_categorical(y, num_classes)

        model.X_test = X.reshape(X.shape[0], X.shape[1], 1)
        model.y_test = y

        model.evaluate_model()
        

if __name__ == "__main__":
    run()            
