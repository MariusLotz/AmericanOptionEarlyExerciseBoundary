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
    data=prep_data()
    data["b-prem_diff"] = abs(data["b-prem_diff"])
    data["p-prem_diff"] = abs(data["p-prem_diff"])
    data["b-prem_diff_rel"] = abs(data["b-prem_diff_rel"])
    data["p-prem_diff_rel"] = abs(data["p-prem_diff_rel"])
    data = data.sort_values("b-prem_diff_rel")
    print(data[["prem", "p-prem_diff_rel", "b-prem_diff_rel", "p-prem_diff", "b-prem_diff"]])
    #print(data)
    # mean:
    print("mean")
    premTFc_diff_abs_mean = data["b-prem_diff"].mean()
    premTFp_diff_abs_mean = data["p-prem_diff"].mean()
    df = data[data["prem"]>1e-6] # without very low prices 
    premTFc_diff_rel_mean = df["b-prem_diff_rel"].mean()
    premTFp_diff_rel_mean = df["p-prem_diff_rel"].mean()
    print(premTFc_diff_abs_mean)
    print(premTFp_diff_abs_mean)
    print(premTFc_diff_rel_mean)
    print(premTFp_diff_rel_mean)
    print()
    # median: 
    print("median")
    premTFc_diff_abs_median = data["b-prem_diff"].median()
    premTFp_diff_abs_median = data["p-prem_diff"].median()
    #df = data[data["prem"]>1e-6] # without very low prices 
    premTFc_diff_rel_median = df["b-prem_diff_rel"].median()
    premTFp_diff_rel_median = df["p-prem_diff_rel"].median()
    print(premTFc_diff_abs_median)
    print(premTFp_diff_abs_median)
    print(premTFc_diff_rel_median)
    print(premTFp_diff_rel_median)
    print()
    # std: 
    print("std")
    premTFc_diff_abs_std = data["b-prem_diff"].std()
    premTFp_diff_abs_std = data["p-prem_diff"].std()
    #df = data[data["prem"]>1e-6] # without very low prices 
    premTFc_diff_rel_std = df["b-prem_diff_rel"].std()
    premTFp_diff_rel_std = df["p-prem_diff_rel"].std()
    print(premTFc_diff_abs_std)
    print(premTFp_diff_abs_std)
    print(premTFc_diff_rel_std)
    print(premTFp_diff_rel_std)
    print()
    print()
    print()
    dd = data[data["prem"]<1e-6]
    #print(dd.size())
    #df = dd.sort_values("p-prem_diff_rel")
    #print(df[["prem", "p-prem_diff_rel", "b-prem_diff_rel", "p-prem_diff", "b-prem_diff"]])
    #df = dd.sort_values("b-prem_diff_rel")
    #print(df[["prem", "p-prem_diff_rel", "b-prem_diff_rel", "p-prem_diff", "b-prem_diff"]])

    """# 5 largest and smallest
    print("5 worse")
    premTFc_diff_abs_worse5 = data["b-prem_diff"].nlargest(5)
    premTFp_diff_abs_worse5 = data["p-prem_diff"].nlargest(5)
    premTFc_diff_rel_worse5 = data["b-prem_diff_rel"].nlargest(5)
    premTFp_diff_rel_worse5 = data["p-prem_diff_rel"].nlargest(5)
    print(premTFc_diff_abs_worse5)
    print(premTFp_diff_abs_worse5)
    print(premTFc_diff_rel_worse5)
    print(premTFp_diff_rel_worse5)
    print("5 smallest")
    premTFc_diff_abs_top5 = data["b-prem_diff"].nsmallest(5)
    premTFp_diff_abs_top5 = data["p-prem_diff"].nsmallest(5)
    premTFc_diff_rel_top5 = data["b-prem_diff_rel"].nsmallest(5)
    premTFp_diff_rel_top5 = data["p-prem_diff_rel"].nsmallest(5)
    print(premTFc_diff_abs_top5)
    print(premTFp_diff_abs_top5)
    print(premTFc_diff_rel_top5)
    print(premTFp_diff_rel_top5)"""

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
    plt.ylim(-1, 1)
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
    plt.ylim(-1, 1)
    plt.legend()
    plt.savefig("scatter_rminusq_rel2", dpi=400)

if __name__ == "__main__":
    stat()
   

     
