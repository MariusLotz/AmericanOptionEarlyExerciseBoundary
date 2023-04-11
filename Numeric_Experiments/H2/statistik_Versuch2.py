import numpy as np
import os
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import __init__


def prep_data(data_name):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    df = pd.read_csv(data_name, header=None)
    df.columns = ["r", "q", "sigma", "p", "prem", "wrong_prem"]
    df["r"] = df["r"].str.replace("[", "", regex=False)
    df["wrong_prem"] = df["wrong_prem"].str.replace("]", "", regex=False)
    df["r"] = df["r"].astype(float)
    df["wrong_prem"] = df["wrong_prem"].astype(float)
    df["diff"] = df["wrong_prem"] - df["prem"]
    df["diff_abs"] = [np.abs(x) for x in df["diff"]]
    df = df[df["prem"] > 1e-6]
    df["diff_rel"] =  [np.abs(y / x) for y, x in zip(df["diff_abs"], df["prem"])]
    df= df.sort_values(by='diff_rel')
    return df

def create_median_for_p(data):
    new_df = pd.DataFrame(columns=["p", "median"])
    for p in [2**(-7), 2**(-5), 2**(-3), 2**(-2), 2**(-1), 1, 2, 4, 8]:
        new_df = new_df.append({"p": p, "median": data[data["p"] == p]["diff_rel"].median()*100}, ignore_index=True)
    return new_df  

def error_median_p():

    plt.figure()
    fig, ax = plt.subplots()

    data_atm = prep_data("error_data_atm_2")
    data = create_median_for_p(data_atm)
    sns.scatterplot(y=data["median"], x=data["p"], label="y=atm, (S/K) = 1", color = "blue")

    data_itm = prep_data("error_data_itm_2")
    data = create_median_for_p(data_itm)
    sns.scatterplot(y=data["median"], x=data["p"], label="y=itm, (S/K) = 0.8", color = "green")

    data_otm = prep_data("error_data_otm_2")
    data = create_median_for_p(data_otm)
    sns.scatterplot(y=data["median"], x=data["p"], label="y=otm, (S/K) = 1.2", color = "red")
    
    ax.set_xlabel("p")
    ax.set_ylabel("Median des Errors in % (Err)")
    plt.legend()
    plt.yscale("log", base=2)
    plt.xscale("log", base=2)
    #plt.xlim(-0.6, 0.6)
    plt.savefig("error_median_p_v2_2", dpi=400)

def create_median_for_sigma(data, p):
    new_df = pd.DataFrame(columns=["sigma", "median"])
    print(data[(data["p"] == p)])
    for sigma in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        new_df = new_df.append({"sigma": sigma, "median": data[(data["sigma"] == sigma) & (data["p"] == p)]["diff_rel"].median()*100}, ignore_index=True)
    return new_df  

def error_median_sigma(p):
    plt.figure()
    fig, ax = plt.subplots()

    data_atm = prep_data("error_data_atm")
    data = create_median_for_sigma(data_atm, p)
    sns.scatterplot(y=data["median"], x=data["sigma"], label="y=atm, (S/K) = 1", color = "blue")

    data_itm = prep_data("error_data_itm")
    data = create_median_for_sigma(data_itm, p)
    sns.scatterplot(y=data["median"], x=data["sigma"], label="y=itm, (S/K) = 0.8", color = "green")

    data_otm = prep_data("error_data_otm")
    data = create_median_for_sigma(data_otm, p)
    sns.scatterplot(y=data["median"], x=data["sigma"], label="y=otm, (S/K) = 1.2", color = "red")
    
    ax.set_xlabel("Sigma")
    ax.set_ylabel("Median des Errors in % (Err)")
    plt.legend()
    plt.yscale("log", base=10)
    plt.savefig("error_median_sigma_p2e3_v2", dpi=400)

if __name__ == "__main__":
    #error_median_p()
    error_median_sigma((2**(3)))