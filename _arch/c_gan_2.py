import numpy as np
import pandas as pd
import tensorflow as tf
import keras as keras
import random
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

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

from common.model_loader import ModelLoader
from common.order_manager import OrderManager

import logging
logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)
logging.getLogger('mlflow.pyfunc').setLevel(logging.ERROR)
logging.getLogger('mlflow.utils.requirements_utils').setLevel(logging.ERROR)

tf.config.set_visible_devices([], 'GPU')

def load_comp_models(exp_id, n_models, group_id, api_url):
    experiment_id = exp_id
    num_models = n_models    
    model_loader = ModelLoader()
    model_loader.init_api(api_url)
    return model_loader.load_composite_models( experiment_id, num_models, group_id)


def gen_cond_data(gb_data, split):
    
    X_data_xgb = gb_data.drop(columns=['output'])   
    y_data_xgb = gb_data['output'].values.reshape(-1, 1)
    X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(X_data_xgb, y_data_xgb, test_size=split, random_state=17)
    
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train_xgb, y_train_xgb)
    y_pred_v = xgb_model.predict(X_train_xgb)
    y_pred = y_pred_v.reshape(-1, 1).astype(np.float32)
        
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = -1    
    
    return y_pred

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Function to set seeds for reproducibility
def set_seeds(seed=42):
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

# Function to create generator model
def create_generator(input_dim, conditioning_dim, lay1, lay2, lay3):
    init = RandomNormal(stddev=0.02)
    
    input_layer = Input(shape=(input_dim + conditioning_dim,))
    
    print("Input Shape:", input_layer.shape)
    x = Dense(lay1, kernel_initializer=init)(input_layer)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Reshape((lay1, 1))(x)
    attention = Attention()([x, x])
    x = Flatten()(attention)
        
    x = Dense(lay2, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Reshape((lay2, 1))(x)
    attention = Attention()([x, x])
    x = Flatten()(attention)
    
    x = Dense(lay3, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    output_layer = Dense(input_dim, activation='tanh')(x)
    return Model(input_layer, output_layer)


# Function to create discriminator model
def create_discriminator(input_dim, conditioning_dim, lay1, lay2, lay3):
    init = RandomNormal(stddev=0.02)
    input_layer = Input(shape=(input_dim + conditioning_dim,))
    
    x = Dense(lay1, kernel_initializer=init)(input_layer)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Reshape((lay1, 1))(x)
    attention = Attention()([x, x])
    x = Flatten()(attention)
    
    x = Dense(lay2, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Reshape((lay2, 1))(x)
    attention = Attention()([x, x])
    x = Flatten()(attention)
    
    x = Dense(lay3, kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Reshape((lay3, 1))(x)
    attention = Attention()([x, x])
    x = Flatten()(attention)
    
    x = Dense(1, activation='sigmoid')(x)
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
def train_gan(generator, discriminator, gan, x_data, y_data, epochs, batch_size, conditioning_dim, patience=10, min_delta=0.0007, steps=1):
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

        # Training Generator
        #for _ in range(1):  # Train generator more times
            noise = np.random.normal(0, 1, (batch_size, x_data.shape[1]))
            gen_input = [noise, real_conditions]
            g_loss = gan.train_on_batch(gen_input, valid)
            

        history['d_loss'].append(d_loss[0])  # Appending d_loss[0] as d_loss contains [loss_value, accuracy]
        history['g_loss'].append(g_loss)
        history['d_acc'].append(100 * d_loss_real[1])  # Update to d_loss_real[1] to track accuracy

        if epoch % 2 == 0:
            print(" ")
            print(f"Epoch {epoch}/{epochs}  Patience: {patience_counter}  Accuracy: {100 * d_loss_real[1]}")
            converted_values = [float(value) for value in d_loss]
            print("Discriminator Loss:", *converted_values)
            converted_values = [float(value) for value in g_loss]
            print("Generator Loss:", *converted_values)
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


def evaluate_results(pred, y_act):
    accuracy = accuracy_score(pred, y_act)
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(pred, y_act))
    print("Classification Report:")
    print(classification_report(pred, y_act, zero_division=1))

# Function to evaluate features using LightGBM model
def evaluate_features(gan_vectors, eval_model, y_val):
    print("Evaluating Features ...")
    upsx, downsx, zerosx = 0, 0, 0
    model_preds = []

    counter = 0

    for i in range(len(gan_vectors)):
        vals = gan_vectors[i]
        dv = pd.DataFrame([vals])
        out_put = eval_model.predict(dv)
        #print(f"Generated data for sample {i}: {y_val[i]}   {vals}")
        #print(f"Prediction for sample {i}: {out_put[0]}   {y_val[i][0]}")

        if out_put > 0.5:  # Since this is a classifier, use 0.5 threshold
            
            counter +=  1
            
            model_preds.append(1)
            if y_val[i] > 0.5: upsx += 1
            if y_val[i] < 0.5: downsx += 1
            
        elif out_put <= 0.5:
            model_preds.append(-1)
            if y_val[i] > 0.5: downsx += 1
            if y_val[i] < 0.5: upsx += 1

    print(f"Ups: {upsx}  Downs: {downsx}  Zeros: {zerosx}  {upsx/(downsx+upsx):.2f}  {counter} ")
    return model_preds

# Function to plot the results
def plot_results(history_in, model_predictions, y_test):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), gridspec_kw={'height_ratios': [1, 2]})

    axes[0, 0].plot(history_in['d_loss'], label='Discriminator Loss')
    axes[0, 0].plot(history_in['g_loss'], label='Generator Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(history_in['d_acc'], label='Discriminator Accuracy')
    axes[0, 1].set_title('Discriminator Accuracy')
    axes[0, 1].legend()
    
    axes[1, :].plot(range(len(model_predictions)), y_test.flatten(), color='black', linestyle='--', label='Y Values')
    axes[1, :].plot(range(len(model_predictions)), model_predictions, color='red', linestyle='-', label='Predicted Values')
    axes[1, :].set_title('Actual vs Predicted Values')
    axes[1, :].set_xlabel('Index')
    axes[1, :].set_ylabel('Output')
    axes[1, :].legend()
    axes[1, :].grid(True)

    plt.tight_layout()
    plt.show()


# Sample Usage
def run():
    set_seeds(42)

    data = pd.read_csv("data/buildSeqInd_Lucky13_5M_ALL.csv")
    data = data.drop(columns=['outputC'])
    
    X_data = data.drop(columns=['output'])    
    y_data = data['output'].values
    
    exp_id = ["42"]
    n_models = 1
    group_id = 0
    api_u = "http://10.0.0.147:8786/"    
    comp_models = load_comp_models(exp_id, n_models, group_id, api_u)
    
    predicts = []
    predicts_v = []
    
    len_d = len(y_data)

    print(y_data.shape)    
    print(f"Generating Predictions ... on {len_d}  values" )
    
    for i in range(len_d):
            for m in range(len(comp_models)):
                predict = comp_models[m].do_predict(X_data.iloc[i])
                predicts.append(predict)
                
                predict_v = comp_models[m].do_predict_v(X_data.iloc[i])
                predicts_v.append(predict_v)
                
                print(f"{i}/{len_d}  {m}  {predict}")
                
    
    df = pd.DataFrame(predicts)
    df.to_csv("predicts.csv")
    
    df = pd.DataFrame(predicts_v)
    df.to_csv("predicts_v.csv")
    
              
    exit()
    
    split = 0.2
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=split, random_state=42)
    
    
    
    
    y_pred = gen_cond_data(data, split)    
    input_dim = X_train.shape[1]
    conditioning_dim = y_pred.shape[1]

    generator = create_generator(input_dim, conditioning_dim, 4, 16, 32)
    #generator = create_generator(input_dim, conditioning_dim, 8, 32, 64)
    #generator = create_generator(input_dim, conditioning_dim, 32, 64, 128)
    #generator = create_generator(input_dim, conditioning_dim, 64, 128, 256)
    
    discriminator = create_discriminator(input_dim, conditioning_dim, 32, 16, 4)    
    #discriminator = create_discriminator(input_dim, conditioning_dim, 64, 16, 4)
    #discriminator = create_discriminator(input_dim, conditioning_dim, 256, 128, 64)    
    
    
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    gan = create_gan(generator, discriminator, input_dim, conditioning_dim)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5))
    gan.summary()

    #scaler = MinMaxScaler((-1,1))
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    history = train_gan(generator, discriminator, gan, X_train, y_pred, 
                        epochs=1000, batch_size=32, 
                        conditioning_dim=conditioning_dim, patience=10, min_delta=0.0007, steps=1)



    print(" ")
    gen_input = np.concatenate([X_val_sc, y_val], axis=1)
    generated_data = generator.predict(gen_input) 
    x_rev = scaler.inverse_transform(generated_data)  
    model_preds = evaluate_features( x_rev, lgb_model, y_val)
    
    #evaluate_results(model_preds, y_val)
    #plot_results(history, model_preds, y_val)
                
        
    exit()
    oo_df = pd.read_csv("data/lucky13_oos.csv")
    oo_df = oo_df.drop(columns=drop_cols)
    oo_X_data = oo_df.drop(columns=['output']) 
    oo_y_data = oo_df['output'].values.reshape(-1, 1)
    print(" ")
   
    #oo_y_data[oo_y_data > 0] = 1
    #oo_y_data[oo_y_data < 0] = -1    
    #oo_y_data[oo_y_data == 0] = 0            
    
    #scaler_n = MinMaxScaler()
    scaler_n = MinMaxScaler((-1,1))
    oo_X_sc = scaler_n.fit_transform(oo_X_data)
    
    oo_gen_input = np.concatenate([oo_X_sc, oo_y_data], axis=1)
    oo_generated_data = generator.predict(oo_gen_input) 
    oo_gen_rev = scaler_n.inverse_transform(oo_generated_data)

    oos_pred = evaluate_features( oo_gen_rev, lgb_model, oo_y_data)
    #evaluate_results(oos_pred, oo_y_data)
    plot_results(history, oos_pred, oo_y_data)

    print(" ")
        
if __name__ == "__main__":
    run()    
    