import numpy as np
import os
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import __init__

def stat_for_price():
    ##price:
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    df = pd.read_csv("price_errors_3.csv")
    convert_to_float = lambda x: float(x.strip('[]'))
    df['err_price'] = df['err_price'].apply(convert_to_float)
    #print("mean_price: ", df['err_price'].mean())
    price_error_mean = df['err_price'].abs().mean()
    #print("median_price: ", df['err_price'].median())
    price_error_median = df['err_price'].abs().median()
    ##curve
    df = pd.read_csv("curve_errors.csv")
    #print("mean_curve_avg: ", df['avg_error_curve'].mean())
    mean_curve_avg = df['avg_error_curve'].abs().mean()
    #print("median_curve_avg: ", df['avg_error_curve'].median())
    median_curve_avg = df['avg_error_curve'].abs().median()
    #print("mean_curve_max: ", df['max_error_curve'].mean())
    mean_curve_max = df['max_error_curve'].abs().mean()
    #print("median_curve_max: ", df['max_error_curve'].median())
    median_curve_max = df['max_error_curve'].abs().median()

    table = pd.DataFrame(columns=['indicator', 'price_error', 'curve_avg_error', 'curve_max_error'])
    table = table.append({'indicator': "mean", 'price_error': price_error_mean, 'curve_avg_error': mean_curve_avg, 'curve_max_error': mean_curve_max}, ignore_index=True)
    table = table.append({'indicator': "median", 'price_error': price_error_median, 'curve_avg_error': median_curve_avg, 'curve_max_error': median_curve_max}, ignore_index=True)
    print(table.to_latex())

def box_price():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    df = pd.read_csv("price_errors_3.csv")
    convert_to_float = lambda x: float(x.strip('[]'))
    df['err_price'] = df['err_price'].apply(convert_to_float)
    plt.figure()
    fig, ax = plt.subplots()
    sns.boxplot(data=df["err_price"], color='red')
    ax.set_ylabel("(pred_price - exact_price) / exact_price")
    plt.savefig("box_price", dpi=400)

def box_price2():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    df = pd.read_csv("price_errors_3.csv")
    convert_to_float = lambda x: float(x.strip('[]'))
    df['err_price'] = df['err_price'].apply(convert_to_float)
    plt.figure()
    fig, ax = plt.subplots()
    sns.boxplot(data=df["err_price"], color='red')
    ax.set_ylabel("(pred_price - exact_price) / exact_price")
    plt.ylim(-3, 3)
    plt.savefig("box_price2", dpi=400)

def box_curve_avg():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    df = pd.read_csv("curve_errors.csv")
    plt.figure()
    fig, ax = plt.subplots()
    sns.boxplot(data=df["avg_error_curve"])
    ax.set_ylabel("avg_error_curve / avg_curve")
    plt.ylim(-0.1, 0.1)
    plt.savefig("box_curve_avg_error", dpi=400)

def box_curve_max():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    df = pd.read_csv("curve_errors.csv")
    plt.figure()
    fig, ax = plt.subplots()
    sns.boxplot(data=df["max_error_curve"])
    ax.set_ylabel("max_error_curve / avg_curve")
    plt.ylim(-0.1, 0.1)
    plt.savefig("box_curve_max_error", dpi=400)

if __name__=="__main__":
    #box_price2()
    #box_curve_max()
    stat_for_price()