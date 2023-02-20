import numpy as np
import os
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import __init__

def prep_data():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    df = pd.read_csv("error_data_atm", header=None)
    df.columns = ["r", "q", "sigma", "p", "prem", "wrong_prem"]
    df["r"] = df["r"].str.replace("[", "", regex=False)
    df["wrong_prem"] = df["wrong_prem"].str.replace("]", "", regex=False)
    df["r"] = df["r"].astype(float)
    df["wrong_prem"] = df["wrong_prem"].astype(float)
    df["diff"] = df["wrong_prem"] - df["prem"]
    df["diff_rel_abs"] = [0 if x == 0 else np.abs(y / x) for y, x in zip(df["diff"], df["prem"])]
    df = df.sort_values("diff_rel_abs")
    df = df [df["prem"]>1e-6] # without very low prices 
    return df

def stat():
    data=prep_data()
    new_df = pd.DataFrame(columns=["p", "mean", "median"])
    """
    new_df = new_df.append({"p": -2**(-7), "mean": data[data["p"]==-2**(-7)]["diff_rel_abs"].mean(), "median": data[data["p"]==-2**(-7)]["diff_rel_abs"].median()}, ignore_index=True)
    new_df = new_df.append({"p": -2**(-5), "mean": data[data["p"]==-2**(-5)]["diff_rel_abs"].mean(), "median": data[data["p"]==-2**(-5)]["diff_rel_abs"].median()}, ignore_index=True)
    new_df = new_df.append({"p": -2**(-3), "mean": data[data["p"]==-2**(-3)]["diff_rel_abs"].mean(), "median": data[data["p"]==-2**(-3)]["diff_rel_abs"].median()}, ignore_index=True)
    new_df = new_df.append({"p": -2**(-2), "mean": data[data["p"]==-2**(-2)]["diff_rel_abs"].mean(), "median": data[data["p"]==-2**(-2)]["diff_rel_abs"].median()}, ignore_index=True)
    new_df = new_df.append({"p": -2**(-1), "mean": data[data["p"]==-2**(-1)]["diff_rel_abs"].mean(), "median": data[data["p"]==-2**(-1)]["diff_rel_abs"].median()}, ignore_index=True)
    new_df = new_df.append({"p": -1, "mean": data[data["p"]==-1]["diff_rel_abs"].mean(), "median": data[data["p"]==-1]["diff_rel_abs"].median()}, ignore_index=True)
    new_df = new_df.append({"p": -2, "mean": data[data["p"]==-2]["diff_rel_abs"].mean(), "median": data[data["p"]==-2]["diff_rel_abs"].median()}, ignore_index=True)
    new_df = new_df.append({"p": -4, "mean": data[data["p"]==-4]["diff_rel_abs"].mean(), "median": data[data["p"]==-4]["diff_rel_abs"].median()}, ignore_index=True)
    new_df = new_df.append({"p": -8, "mean": data[data["p"]==-8]["diff_rel_abs"].mean(), "median": data[data["p"]==-8]["diff_rel_abs"].median()}, ignore_index=True)
    """ 
    new_df = new_df.append({"p": 2**(-7), "mean": data[data["p"]==2**(-7)]["diff_rel_abs"].mean()*100, "median": data[data["p"]==-2**(-7)]["diff_rel_abs"].median()*100}, ignore_index=True)
    new_df = new_df.append({"p": 2**(-5), "mean": data[data["p"]==2**(-5)]["diff_rel_abs"].mean()*100, "median": data[data["p"]==2**(-5)]["diff_rel_abs"].median()*100}, ignore_index=True)
    new_df = new_df.append({"p": 2**(-3), "mean": data[data["p"]==2**(-3)]["diff_rel_abs"].mean()*100, "median": data[data["p"]==2**(-3)]["diff_rel_abs"].median()*100}, ignore_index=True)
    new_df = new_df.append({"p": 2**(-2), "mean": data[data["p"]==2**(-2)]["diff_rel_abs"].mean()*100, "median": data[data["p"]==2**(-2)]["diff_rel_abs"].median()*100}, ignore_index=True)
    new_df = new_df.append({"p": 2**(-1), "mean": data[data["p"]==2**(-1)]["diff_rel_abs"].mean()*100, "median": data[data["p"]==2**(-1)]["diff_rel_abs"].median()*100}, ignore_index=True)
    new_df = new_df.append({"p": 1, "mean": data[data["p"]==1]["diff_rel_abs"].mean()*100, "median": data[data["p"]==1]["diff_rel_abs"].median()*100}, ignore_index=True)
    new_df = new_df.append({"p": 2, "mean": data[data["p"]==2]["diff_rel_abs"].mean()*100, "median": data[data["p"]==2]["diff_rel_abs"].median()*100}, ignore_index=True)
    new_df = new_df.append({"p": 4, "mean": data[data["p"]==4]["diff_rel_abs"].mean()*100, "median": data[data["p"]==4]["diff_rel_abs"].median()*100}, ignore_index=True)
    new_df = new_df.append({"p": 8, "mean":  data[data["p"]==8]["diff_rel_abs"].mean()*100, "median": data[data["p"]==8]["diff_rel_abs"].median()*100}, ignore_index=True)
    new_df = new_df.sort_values("median")
    #new_df = new_df.reset_index(drop=True)
    print(new_df)
    plt.figure()
    fig, ax = plt.subplots()
    #sns.lineplot(x="p", y="median", data=new_df, color='red', label="y=p")
    plt.plot(new_df["p"], new_df["median"], linestyle='--', marker='o', color='r')
    #sns.scatterplot(y=data["b-prem_diff_rel"], x=(data["r"] - data["q"]), label="y=c")
    #ax.set_xlabel("r - q")
    #ax.set_ylabel("(premTFy - prem) / prem")
    #plt.ylim(-1, 1)
    #plt.legend()
    #plt.yscale("log", base=2)
    #plt.xscale("log", base=2)
    plt.savefig("testt", dpi=400)

if __name__ == "__main__":
    stat()
   

     
