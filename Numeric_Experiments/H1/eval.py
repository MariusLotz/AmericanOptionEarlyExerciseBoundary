import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import __init__
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def load_and_return_trainingsdata_curve():
    """Loading data from txt. file and return trainingsdata"""
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    file = open("curve_test")
    lines = file.readlines()
    data = []
    for line in lines:
        try:
            [r, q, sigma, boundary] = eval(line)
            data.append([r, q, sigma, boundary])
        except: break
    return data

def load_and_return_trainingsdata_price():
    """Loading data from txt. file and return trainingsdata"""
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    file = open("price_test")
    lines = file.readlines()
    data = []
    for line in lines:
        try:
            [r, q, sigma, S, price] = eval(line)
            data.append([r, q, sigma, S, price])
        except: break
    return data

def avg_diff(pred_curve, exact_curve):
    sum = 0
    mean = 0 
    n = len(exact_curve)
    for i in range(n):
        mean += exact_curve[i]
        sum += (pred_curve[i] - exact_curve[i])
    mean = mean / n
    sum = sum / n
    return (sum / mean)

def max_diff(pred_curve, exact_curve):
    max_val = 0
    mean = 0
    n = len(exact_curve)
    for i in range(n):
        mean += exact_curve[i]
        err = pred_curve[i] - exact_curve[i]
        if abs(err) > abs(max_val):
            max_val = pred_curve[i] - exact_curve[i]
    mean = mean / n        
    return (max_val / mean)

def calc_df_curve():
    data_curve = load_and_return_trainingsdata_curve()
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    K=100
    T=1
    option_type="Put"
    model_curve = tf.keras.models.load_model('curve_model')
    df = pd.DataFrame(columns=['r', 'q', 'sigma', 'avg_error_curve', 'max_error_curve'])
    for i in range(len(data_curve)):
        ## curve:
        [r, q, sigma, exact_curve] = data_curve[i]
        [pred_curve] = model_curve.predict([[r, q, sigma]])
        avg_error = avg_diff(pred_curve, exact_curve)
        max_error = max_diff(pred_curve, exact_curve)
        ## add entry:
        df = df.append({'r': r, 'q': q, 'sigma': sigma, 'avg_error_curve': avg_error, 'max_error_curve': max_error}, ignore_index=True)
    return df

def calc_df_price():
    data_price = load_and_return_trainingsdata_price()
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    K=100
    T=1
    option_type="Put"
    model_price = tf.keras.models.load_model('price_model')
    df = pd.DataFrame(columns=['r', 'q', 'sigma', 'S','err_price'])
    for i in range(len(data_price)):
        ## price: 
        [r, q, sigma, S, exact_price] = data_price[i]
        [pred_price] = model_price.predict([[r, q, sigma, S]])
        if exact_price < 10**(-6):
            continue
        error = (pred_price - exact_price) / exact_price
        ## add entry:
        df = df.append({'r': r, 'q': q, 'sigma': sigma, 'S': S, 'err_price': error}, ignore_index=True)
    #convert_to_float = lambda x: float(x.strip('[]'))
    #df['err_price'] = df['err_price'].apply(convert_to_float)
    return df



def main():
    #df = calc_df_curve()
    #df.to_csv('curve_errors.csv', index=False)
    df = calc_df_price()
    df.to_csv('price_errors_3.csv', index=False)

if __name__=="__main__":
    main()