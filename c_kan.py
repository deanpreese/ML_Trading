import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, Average, Conv2D, Reshape, Concatenate, Multiply, BatchNormalization, Bidirectional, Add, Dense, Dropout, MaxPooling1D, LSTM, MultiHeadAttention, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2
from ml_model.model_stats import gen_reg_stats_x, gen_class_stats
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)


class C_KAN:
    def __init__(self, epochs=50, batch_size=32):
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.checkpoint_dir = 'checkpoints/'
        self.trained_dir = 'trained_models/'
       
        self.checkpoint_model = os.path.join(self.checkpoint_dir, 'c_kan_model.keras')
        self.trained_model = os.path.join(self.trained_dir, 'c_kan_model.keras')
        self.model_plot = os.path.join(self.checkpoint_dir, 'c_kan_model.png')

        self.drop_out = 0.2
        self.l2_reg = l2(0.01)
        self.initializer = GlorotUniform(seed=42)


    def preprocess_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler



    def create_feature_model(self, input_shape):
        
        input = Input(shape=input_shape)
        input_dim = input.shape[1]  
        reshaped_inputs = Reshape((input_dim, 1))(input)
        inx = LSTM(16, return_sequences=True, activation='relu')(reshaped_inputs)
        #inx = Dropout(self.drop_out)(inx)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(inx)
       # x = Dropout(self.drop_out)(x)
        x = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer=self.initializer)(x)
        #x = Dropout(self.drop_out)(x)
        x = MaxPooling1D(pool_size=1, strides=1)(x)
        x = LSTM(16, return_sequences=False, activation='relu')(x)
        #x = Dropout(self.drop_out)(x)
        smx_out = Dense(1, activation='linear')(x) 
        subx_model = Model(input, smx_out)
        
        return subx_model


    def create_custom_model(self, num_features):
        inputs = Input(shape=(num_features,))
        
        feature_outputs = []
        for i in range(num_features):
            feature_input = inputs[:, i:i+1]
            feature_model = self.create_feature_model((1,))
            feature_output = feature_model(feature_input)
            feature_outputs.append(feature_output)
        
        concatenated_outputs = Concatenate(axis=1)(feature_outputs)
        reshaped_attention_input = Reshape((num_features, 1))(concatenated_outputs)
        attention = Dense(num_features, activation='softmax', kernel_initializer=self.initializer, name='attention')(reshaped_attention_input)
        weighted = Multiply()([reshaped_attention_input, attention])
        
        weighted = Conv1D(32, 2, activation='relu', kernel_initializer=self.initializer)(weighted)
        
        weighted = Reshape((-1,))(weighted)
        aggregated = Dense(32, activation='relu', kernel_regularizer=self.l2_reg,kernel_initializer=self.initializer)(weighted)
        aggregated = Dropout(self.drop_out)(aggregated)
        
        output = Dense(1, activation='linear')(aggregated)
        
        self.model = Model(inputs=inputs, outputs=output)
        return self.model


    

    def compile_and_train_model(self, X_train, y_train, X_val, y_val):

        self.model.compile(optimizer=Adam(learning_rate=0.001), 
                loss='mse', metrics=['mae', tf.keras.metrics.R2Score()])

        self.model.summary()
        tf.keras.utils.plot_model(self.model, to_file=self.model_plot, show_shapes=True)

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
                    verbose=0,
                    save_best_only=True, 
                        save_weights_only=False, mode='min')
      
      
        history_out = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                                initial_epoch=0, epochs=150, verbose=1,
                                batch_size=32, callbacks=[
                                    early_stopping,
                                    reduce_lr,
                                    model_checkpoint])      
        
        return history_out


    def load_saved_model(self, mode):
        
        #if mode == "run":
        #   self.model = tf.keras.models.load_model(self.trained_model)
        #if mode == "train":
        self.model = tf.keras.models.load_model(self.checkpoint_model)

        return self.model

    def evaluate_model(self, X_train, y_train, X_val, y_val, X_test, y_test, y_pred, cols):
        train_scores = self.model.evaluate(X_train, y_train, verbose=0)
        val_scores = self.model.evaluate(X_val, y_val, verbose=0)
        test_scores =self.model.evaluate(X_test, y_test, verbose=0)
        
        attention_layer = self.model.get_layer('attention')
        feature_weights = attention_layer.get_weights()[0]
        feature_importance = np.mean(feature_weights, axis=0)

        correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(y_test, y_pred)
        print("")

        fi = list(feature_importance)
        cols_fi = list(zip(cols, fi))
        sorted_fi = sorted(cols_fi, key=lambda x: x[1], reverse=True)

        # Print the sorted feature importance
        for col, importance in sorted_fi:
            print(f"{col}    {importance}") 
        
        #print(" ")
        #print(f"Train Loss: {train_scores[0]}, MAE: {train_scores[1]}, R2: {train_scores[2]}")
        #print(f"Val Loss: {val_scores[0]}, MAE: {val_scores[1]}, R2: {val_scores[2]}")
        print(" ")
        print(f"Pred MSE: {mse},  MAE: {mae}, R2: {r2}")
        print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.4f}")
        print(f"Number of Samples: {total}")        
        print(" ")        
        
        
        return train_scores, val_scores, test_scores


    def combined_plots(self, history, num_features, y_true, y_pred):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Subplot 1: Training History (Loss)
        axes[0, 0].plot(history.history['loss'], label='Train Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()

        # Subplot 2: Training History (MAE)
        axes[0, 1].plot(history.history['mae'], label='Train MAE')
        axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Model MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()

        # Subplot 3: Feature Importance
        attention_layer = self.model.get_layer('attention')
        feature_weights = attention_layer.get_weights()[0]
        feature_importance = np.mean(feature_weights, axis=0)
        
        axes[1, 0].bar(range(num_features), feature_importance)
        axes[1, 0].set_title('Feature Importance')
        axes[1, 0].set_xlabel('Feature Index')
        axes[1, 0].set_ylabel('Importance')

        # Subplot 4: Actual vs Predicted
        axes[1, 1].scatter(y_true, y_pred, alpha=0.5)
        axes[1, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Actual vs Predicted Values')

        plt.tight_layout()
        plt.show()



def main():

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
        'data/Lucky13_3070_3.csv',  #9
        'data/Lucky13_3070_oos_5.csv',   
        'data/Lucky13_3070_5.csv',  #11
        'data/new_model_HLC_lucky13.csv', #12
        'data/R_HLC_lucky13.csv', #13
    ]

    file_path = datafile[1]
    
    df = pd.read_csv(file_path)
    df = df.drop(columns=['outputC'])
    X = df.drop(columns=['output'])
    cols = X.columns
    X = X.values
    y = df['output'].values

    c_kan = C_KAN()
    
    # Preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = c_kan.preprocess_data(X, y)
    
    # Create and train model
    num_features = X_train.shape[1]
    model = c_kan.create_custom_model(num_features)
    history = c_kan.compile_and_train_model(X_train, y_train, X_val, y_val)
    
    # Load best model and evaluate
    best_model = c_kan.load_saved_model('train')
    y_pred = best_model.predict(X_test)
    train_mae, val_mae, test_mae = c_kan.evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, y_pred, cols)
    
    # Visualizations
    #c_kan.combined_pßlots(history, num_features, y_test, y_pred)


if __name__ == "__main__":
    main()
    
    
'''


''' 