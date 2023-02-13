import numpy as np
import os
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import __init__

def prep_data():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    #data = load_and_return_data()
    df = pd.read_csv("eval_data", header=None)
    df.columns = ["r", "q", "sigma", "S", "prem", "b-prem", "p-prem"]
    df["r"] = df["r"].str.replace("[", "", regex=False)
    df["p-prem"] = df["p-prem"].str.replace("]", "", regex=False)
    df["r"] = df["r"].astype(float)
    df["p-prem"] = df["p-prem"].astype(float)
    df["b-prem_diff"] = df["b-prem"] - df["prem"]
    df["b-prem_diff_rel"] = df["b-prem_diff"] / df["prem"]
    df["p-prem_diff"] = df["p-prem"] - df["prem"]
    df["p-prem_diff_rel"] =  df["p-prem_diff"]  / df["prem"]
    df = df.sort_values("b-prem_diff_rel")
    #df = df [df["prem"]>1e-6] # without very low prices 
    return df

def stat():
    print("start")
    data=prep_data()
    # mean:
    premTFc_diff_abs_mean = data["b-prem_diff"].mean()
    premTFp_diff_abs_mean = data["p-prem_diff"].mean()
    df = df[data["prem"]>1e-6] # without very low prices 
    premTFc_diff_rel_mean = df["b-prem_diff_rel"].mean()
    premTFp_diff_rel_mean = df["p-prem_diff_rel"].mean()
    print(premTFc_diff_abs_mean)
    print(premTFp_diff_abs_mean)
    print(premTFc_diff_rel_mean)
    print(premTFp_diff_rel_mean)
    print(df["p-prem_diff_rel"].describe)

def scatter_sigma_rel():
    data = prep_data()
    data = data[data["prem"]>1e-6] # without very low prices
    plt.figure()
    fig, ax = plt.subplots()
    sns.scatterplot(y=data["p-prem_diff_rel"], x=data["sigma"], color='red', label="y=p")
    sns.scatterplot(y=data["b-prem_diff_rel"], x=data["sigma"], label="y=c")
    ax.set_xlabel("sigma")
    ax.set_ylabel("(premTFy - prem) / prem")
    plt.yscale('symlog')
    plt.legend()
    plt.savefig("scatter_sigma_rel", dpi=400)

def scatter_sigma_rel2():
    data = prep_data()
    data = data[data["prem"]>1e-6] # without very low prices
    plt.figure()
    fig, ax = plt.subplots()
    sns.scatterplot(y=data["p-prem_diff_rel"], x=data["sigma"], color='red', label="y=p")
    sns.scatterplot(y=data["b-prem_diff_rel"], x=data["sigma"], label="y=c")
    ax.set_xlabel("sigma")
    ax.set_ylabel("(premTFy - prem) / prem")
    plt.ylim(-3, 3)
    plt.legend()
    plt.savefig("scatter_sigma_rel2", dpi=400)

def scatter_sigma_abs():
    data = prep_data()
    plt.figure()
    fig, ax = plt.subplots()
    sns.scatterplot(y=data["p-prem_diff"], x=data["sigma"], color='red', label="y=p")
    sns.scatterplot(y=data["b-prem_diff"], x=data["sigma"], label="y=c")
    ax.set_xlabel("sigma")
    ax.set_ylabel("premTFy - prem")
    plt.legend()
    plt.savefig("scatter_sigma_abs", dpi=400)

def scatter_rminusq_abs():
    data = prep_data()
    plt.figure()
    fig, ax = plt.subplots()
    sns.scatterplot(y=data["p-prem_diff"], x=(data["r"] - data["q"]) , color='red', label="y=p")
    sns.scatterplot(y=data["b-prem_diff"], x=(data["r"] - data["q"]), label="y=c")
    ax.set_xlabel("r - q")
    ax.set_ylabel("premTFy - prem")
    plt.legend()
    plt.savefig("scatter_rminusq_abs", dpi=400)

def scatter_rminusq_rel():
    data = prep_data()
    data = data[data["prem"]>1e-6] # without very low prices
    plt.figure()
    fig, ax = plt.subplots()
    sns.scatterplot(y=data["p-prem_diff_rel"], x=(data["r"] - data["q"]), color='red', label="y=p")
    sns.scatterplot(y=data["b-prem_diff_rel"], x=(data["r"] - data["q"]), label="y=c")
    ax.set_xlabel("r - q")
    ax.set_ylabel("(premTFy - prem) / prem")
    plt.yscale('symlog')
    plt.legend()
    plt.savefig("scatter_rminusq_rel", dpi=400)

def scatter_rminusq_rel2():
    data = prep_data()
    data = data[data["prem"]>1e-6] # without very low prices
    plt.figure()
    fig, ax = plt.subplots()
    sns.scatterplot(y=data["p-prem_diff_rel"], x=(data["r"] - data["q"]), color='red', label="y=p")
    sns.scatterplot(y=data["b-prem_diff_rel"], x=(data["r"] - data["q"]), label="y=c")
    ax.set_xlabel("r - q")
    ax.set_ylabel("(premTFy - prem) / prem")
    plt.ylim(-3, 3)
    plt.legend()
    plt.savefig("scatter_rminusq_rel2", dpi=400)

if __name__ == "__main__":
    stat()

     
