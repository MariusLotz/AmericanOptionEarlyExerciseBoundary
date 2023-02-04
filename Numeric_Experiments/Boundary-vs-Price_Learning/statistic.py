import numpy as np
import os
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import __init__


def main():
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
    df = df [df["prem"]>1e-6] # without very low prices 
    print(df.shape[0])   
if __name__ == "__main__":
    main()
     
