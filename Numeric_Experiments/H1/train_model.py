import tensorflow as tf
import numpy as np
import os
import __init__

def load_and_return_data_curve():
    """Loading data from txt. file and return trainingsdata"""
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    file = open("curve_train")
    lines = file.readlines()
    x_list = []
    fx_list = []
    for line in lines:
        try:
            [r, q, sigma, boundary] = eval(line)
            x = [r, q, sigma]
            fx = boundary
            x_list.append(x)
            fx_list.append(fx)
        except: break
    return [x_list, fx_list]

def trainmodell_curve():
    x_list, fx_list = load_and_return_data_curve()
      
    """Creating Modell"""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1024, input_shape=(3,), activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(21, activation='linear'))
    #tf.keras.callbacks.EarlyStopping(
        #monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        #mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics='mean_absolute_percentage_error')
    nepoch = 2999
    nbatch = 64
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save("curve_model")

def load_and_return_data_price():
    """Loading data from txt. file and return trainingsdata"""
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    file = open("price_train")
    lines = file.readlines()
    x_list = []
    fx_list = []
    for line in lines:
        try:
            [r, q, sigma, S, prem] = eval(line)
            x = [r, q, sigma, S]
            fx = prem
            x_list.append(x)
            fx_list.append(fx)
        except: break
    return [x_list, fx_list]

def trainmodell_price():
    x_list, fx_list = load_and_return_data_price()
      
    """Creating Modell"""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1024, input_shape=(4,), activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    #tf.keras.callbacks.EarlyStopping(
        #monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        #mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics='mean_absolute_percentage_error')
    nepoch = 2999
    nbatch = 64
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save("price_model")

if __name__=="__main__":
    #trainmodell_curve()
    trainmodell_price()