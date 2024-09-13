import numpy as np
import pandas as pd
import tensorflow as tf
import keras as keras
import random

from models.wrapped_models import TunableLGBMClassifier, TunableLGBMRegressor, TunableXGBClassifier, TunableXGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Conv1D, TimeDistributed, Flatten, Dropout, BatchNormalization, LeakyReLU, Bidirectional, Concatenate, MultiHeadAttention
from keras.layers import MultiHeadAttention, LayerNormalization, Add, Reshape, AdditiveAttention, Attention

from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.initializers import RandomNormal
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

tf.config.set_visible_devices([], 'GPU')

# Function to set seeds for reproducibility
def set_seeds(seed=42):
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def gen_cond_data_c(X_train, y_train):
    y_train[y_train > 0] = 1
    y_train[y_train <= 0] = 0           
    xgb_1 = TunableXGBClassifier()
    xgb_1.fit(X_train, y_train)
    xgb_1_v = xgb_1.predict(X_train)
    p_xgb = TunableXGBClassifier().param_set()
    xgb_cond = TunableXGBClassifier(**p_xgb)
    xgb_cond.fit(X_train, y_train)
    xgb_pred_v = xgb_cond.predict(X_train)
     
    y_pred = (xgb_pred_v.reshape(-1, 1).astype(np.float32)*.25 + xgb_1_v.reshape(-1, 1).astype(np.float32)*.25 + y_train*.5)
    return y_pred 


def gen_cond_data(X_train, y_train):
    xgb_1 = TunableXGBRegressor()
    xgb_1.fit(X_train, y_train)
    xgb_1_v = xgb_1.predict(X_train)
    p_xgb = TunableXGBRegressor().param_set()
    xgb_model = TunableXGBRegressor(**p_xgb)
    xgb_model.fit(X_train, y_train)
    xgb_pred_v = xgb_model.predict(X_train)
     
    y_pred = (xgb_pred_v.reshape(-1, 1).astype(np.float32)*.25 + xgb_1_v.reshape(-1, 1).astype(np.float32)*.25 + y_train*.5)
    return y_pred


def train_final_model_c(X_train, y_train):
    p = TunableLGBMClassifier().get_params()
    lgb_model = TunableLGBMClassifier(*p)
    lgb_model.fit(X_train, y_train)    
    return lgb_model


def train_final_model(X_train, y_train):
    p_lgb = TunableLGBMRegressor().get_params()
    lgb_model = TunableLGBMRegressor(**p_lgb)    
    lgb_model.fit(X_train, y_train)    
    return lgb_model


# Function to create generator model
def create_generator(input_dim, conditioning_dim, lay1, lay2, lay3, out_activation):
    init = RandomNormal(stddev=0.02)
    input_layer = Input(shape=(input_dim + conditioning_dim,))
    x = Dense(lay1, kernel_initializer=init )(input_layer)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(lay2, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.4)(x)
    x = Reshape((lay2, 1))(x)
    attention = Attention()([x, x])
    x = Flatten()(attention)
    x = Dense(lay3, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(input_dim, activation=out_activation)(x)
    return Model(input_layer, x)


# Function to create discriminator model
def create_discriminator(input_dim, conditioning_dim, lay1, lay2, lay3, out_activation):
    init = RandomNormal(stddev=0.02)
    input_layer = Input(shape=(input_dim + conditioning_dim,))
    x = Dense(lay1, kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_layer)
    #x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(lay2, kernel_initializer=init)(x)
    #x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.4)(x)
    x = Dense(lay3, kernel_initializer=init)(x)
    x = Dropout(0.3)(x)
    x = Dense(1, activation=out_activation)(x)
    return Model(input_layer, x)

# Function to create the GAN model
def create_gan(generator, discriminator, input_dim, conditioning_dim):
    discriminator.trainable = False
    noise_input = Input(shape=(input_dim,))
    condition_input = Input(shape=(conditioning_dim,))
    gan_input = tf.keras.layers.Concatenate()([noise_input, condition_input])
    x = generator(gan_input)
    gan_output = discriminator(tf.keras.layers.Concatenate()([x, condition_input]))
    return Model([noise_input, condition_input], gan_output)

# Gradient Penalty Function
def gradient_penalty(discriminator, real_samples, fake_samples, conditioning_samples, batch_size):
    alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
    interpolated = real_samples + alpha * (fake_samples - real_samples)
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        d_interpolated = discriminator(tf.concat([interpolated, conditioning_samples], axis=1))
    gradients = tape.gradient(d_interpolated, [interpolated])[0]
    grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
    gradient_penalty = tf.reduce_mean((grad_l2 - 1.0) ** 2)
    return gradient_penalty

# Function to train the GAN model
def train_gan(generator, discriminator, gan, x_data, y_data, epochs, batch_size, conditioning_dim, patience=10, min_delta=0.0001, steps=1):
    history = {'d_loss': [], 'g_loss': [], 'd_acc': []}
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    best_g_loss = np.inf
    patience_counter = 0

    for epoch in range(epochs):
        for _ in range(steps):  
            
            idx = np.random.randint(0, x_data.shape[0], batch_size)
            real_samples = x_data[idx]
            real_conditions = y_data[idx]
            noise = np.random.normal(0, 1, (batch_size, x_data.shape[1]))
            gen_input = [noise, real_conditions]
            generated_samples = generator.predict(np.concatenate(gen_input, axis=1))
            d_loss_real = discriminator.train_on_batch(np.concatenate([real_samples, real_conditions], axis=1), valid)
            d_loss_fake = discriminator.train_on_batch(np.concatenate([generated_samples, real_conditions], axis=1), fake)
            gp = gradient_penalty(discriminator, real_samples, generated_samples, real_conditions, batch_size)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) + 10 * gp  # Gradient penalty coefficient

        for _ in range(1):  # Train generator more times
            noise = np.random.normal(0, 1, (batch_size, x_data.shape[1]))
            gen_input = [noise, real_conditions]
            g_loss = gan.train_on_batch(gen_input, valid)

        history['d_loss'].append(d_loss)  # Appending d_loss[0] as d_loss contains [loss_value, accuracy]
        history['g_loss'].append(g_loss)
        history['d_acc'].append(100 * d_loss_real)  # Update to d_loss_real[1] to track accuracy

        if epoch % 2 == 0:
            print(" ")
            print(f"Epoch {epoch}/{epochs}  Patience: {patience_counter} ")
            print("Discriminator Loss:", d_loss.numpy())
            print("Generator Loss:", g_loss[0] ,  g_loss[1]  )
            print(" ")

        g_loss_value = np.mean(history['g_loss'])
        
        if g_loss_value < best_g_loss - min_delta:
            best_g_loss = g_loss_value
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
    return history

# Function to evaluate features using LightGBM model
def evaluate_features(gan_vectors, eval_model, y_val):
    print("Evaluating Features ...")
    count, wins_u, wins_d, losses_u, losses_d= 0,0,0,0,0

    model_preds = []
    
    for i in range(len(gan_vectors)):
        vals = gan_vectors[i]
        dv = pd.DataFrame([vals])
        out_put = eval_model.predict(dv)
        #print(f"Prediction for sample {i}: {out_put}  {y_val[i][0]}  {dv.values} ")
        model_preds.append(out_put)
        
        count += 1
        
        if out_put[0] > 0 and y_val[i][0] > 0:
            wins_u += 1 

        elif out_put[0] < 0 and y_val[i][0] < 0:
            wins_d += 1 

        elif out_put[0] > 0 and y_val[i][0] <= 0:
            losses_u += 1
            
        elif out_put[0] < 0 and y_val[i][0] >= 0:
            losses_d += 1            
             

        if i % 4000 == 0:
            if (wins_d == 0  or losses_d == 0):
                print(f"Prediction for sample {i}: {out_put}  {y_val[i][0]} ")
            else:                
                print(f"Intermediate wins_u: {wins_u}  wins_d: {wins_d}  {(wins_u+wins_d)/count} ")
            
                      
    print(f"{count} TotWins {wins_u+wins_d}  wins_u: {wins_u}  wins_d: {wins_d}  TotLosses {losses_u+losses_d}  losses_u: {losses_u}  losses_d: {losses_d}  {(wins_u+wins_d)/count}  {wins_d+wins_u+losses_d+losses_u}  ")
    return model_preds


# Function to plot the results
def plot_results(history_in, model_predictions, y_test, last_x_rows ):
    
    if last_x_rows is not None:
        model_predictions = model_predictions[-last_x_rows:]
        y_test = y_test[-last_x_rows:]    
    
    
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])

    # Top row, left plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history_in['d_loss'], label='Discriminator Loss')
    ax1.plot(history_in['g_loss'], label='Generator Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.grid(True)
    ax1.legend()

    # Top row, right plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history_in['d_acc'], label='Discriminator Accuracy')
    ax2.set_title('Discriminator Accuracy')
    ax2.legend()

    # Bottom row, spanning both columns
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(range(len(model_predictions)), y_test.flatten(), color='black', linestyle='--', label='Y Values')
    ax3.plot(range(len(model_predictions)), model_predictions, color='red', linestyle='-', label='Predicted Values')
    ax3.set_title('Actual vs Predicted Values')
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Output')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()
    plt.show()


# Sample Usage
def run():
    set_seeds(42)
    data = pd.read_csv("data/buildSeqInd_Lucky13_5M_ALL.csv")
    data = data.drop(columns=['outputC'])    
    X_data = data.drop(columns=['output'])    
    y_data = data['output'].values.reshape(-1, 1)
    
    drop_cols = [
        #'STOK1',
        #'RSI',
        #'ATR2',
        'ATR21',
        #'ATR3',
        'ATR31', 
        'ATR32',
        'ATR34',   
        #'ROC',     
        #'SDKC9',   
        'SDKC91',  
        'SDBB91',  
        #'SDLR310'
    ]

    #data = data.drop(columns=drop_cols)
    X_data = data.drop(columns=['output'])    
    y_data = data['output'].values.reshape(-1, 1)
    
    split = 0.2
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=split, random_state=42)
    
    #scaler = StandardScaler()
    #scaler = MinMaxScaler((0,1))
    #scaler = MinMaxScaler((-1,1))
    #scaler = MinMaxScaler()
    
    #X_train = scaler.fit_transform(X_train)
    #X_val = scaler.transform(X_val)
        
    X_train = X_train.values
    X_val = X_val.values
    
    lgb_model = train_final_model(X_train, y_train)
    y_pred = gen_cond_data(X_train, y_train)   
    input_dim = X_train.shape[1]
    conditioning_dim = y_pred.shape[1]

    generator = create_generator(input_dim, conditioning_dim, 8, 64, 256, 'linear')
    #generator = create_generator(input_dim, conditioning_dim, 8, 64, 256, 'tanh')
    
    discriminator = create_discriminator(input_dim, conditioning_dim, 256, 64, 8, 'linear')
    #discriminator = create_discriminator(input_dim, conditioning_dim, 256, 64, 8, 'sigmoid')
        
    discriminator.compile(loss='mean_squared_error', optimizer=Adam(0.003, 0.5), metrics=['mean_squared_error'])
    #discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.002, 0.5), metrics=['accuracy'])
    
    gan = create_gan(generator, discriminator, input_dim, conditioning_dim)
    gan.compile(loss='mean_squared_error', optimizer=Adam(0.001, 0.5))
    #gan.compile(loss='binary_crossentropy', optimizer=Adam(0.001, 0.5))
    
    discriminator.summary()
    gan.summary()
    
    history = train_gan(generator, discriminator, gan, X_train, y_pred, 
                        epochs=1000, batch_size=64, conditioning_dim=conditioning_dim, patience=10, 
                        min_delta=0.001, steps=1)

    print(" ")
    gen_input = np.concatenate([X_val, y_val], axis=1)
    generated_data = generator.predict(gen_input) 
    model_preds = evaluate_features( generated_data, lgb_model, y_val)
    
    plot_results(history, np.array(model_preds), y_val, 500)            
    
    
        
if __name__ == "__main__":
    run()    
    