import numpy as np
import sys
import os
myDir = os.getcwd()
sys.path.append(myDir)
from pathlib import Path
path = Path(myDir)
a=str(path.parent.absolute())
sys.path.append(a)
import Solver.Option_Solver as op 
import tensorflow as tf
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import Machine_Learning.Base as Base

def test_data():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    file = open('test_data_1', 'r')
    x_list = []
    fx_list = []
    count = 0
    while True:
        line = file.readline()
        count += 1
        if not line:
            break
        try:
            xline = eval(line)
        except:
            continue
        x = [xline[0],  # r
             xline[1],  # q
             xline[2],  # sigma
             xline[3]]  # S
        x_list.append(x)
        fx_list.append(xline[4]) # price
    file.close()
    return x_list, fx_list

def create_pandas_dataframe():
    x_list, fx_list = test_data()
    model_path = "model1_4"
    model = tf.keras.models.load_model(model_path)

    approx_Early_Exercise_Boundary_list = model.predict([a[:3] for a in x_list])
    r_list = [a[0] for a in x_list]
    q_list = [a[1] for a in x_list]
    sigma_list = [a[2] for a in x_list]
    S_list = [a[3] for a in x_list]
    #tau_vec, boundary, w_vec = option.gaussian_grid_boundary(n=25)
    approx_prem_list = []
    for i in range(len(approx_Early_Exercise_Boundary_list)):
        boundary = approx_Early_Exercise_Boundary_list[i][:25]
        r = r_list[i]
        q = q_list[i]
        sigma = sigma_list[i]
        S = S_list[i]
        option = op.Option_Solver(r_list[i], q_list[i], sigma_list[i], 100, 1, "Put", m=25)
        option.create_boundary()
        approx_prem = option.premium(S, 1, boundary)
        #approx_prem = Base.gaussian_premium(r, q, sigma, 100, S, 1, tau_vec, boundary, w_vec, 1, "Put")
        approx_prem_list.append(approx_prem)
    data = {'r': r_list[:1000], 'q': q_list[:1000], 'sigma': sigma_list[:1000], 'S': S_list[:1000], 'prem': fx_list[:1000], 'approx_prem': approx_prem_list}
    df = pd.DataFrame(data)
    #pd.set_option('float_format', '{:.22f}'.format)
    #print(df)
    df["prem_diff"] = df["approx_prem"] - df["prem"]
    df["abs_prem_diff"] = [np.abs(x) for x in df["prem_diff"]]
    df["rel_prem_diff"] = (df["approx_prem"] - df["prem"]) / df["prem"]
    df["abs_rel_prem_diff"] = [np.abs(x) for x in df["rel_prem_diff"]]
    df.to_csv('pandas_test_2.csv', index=False)
    #print(df)
    #return df

def make_table(df):
    # 3 largest / smallest -- abs:
    largest_abs_price_diff = df.nlargest(3, 'abs_prem_diff')
    smallest_abs_price_diff = df.nsmallest(3, 'abs_prem_diff')
    smallest_abs_price_diff = smallest_abs_price_diff.sort_values(by='abs_prem_diff', ascending=False)
    frames = [largest_abs_price_diff, smallest_abs_price_diff]
    result_abs = pd.concat(frames)
    result_abs = result_abs.drop(['abs_prem_diff', 'abs_rel_prem_diff'], axis=1)
    #print(result_abs.to_latex(index=False))

    rel_largest_abs_price_diff = df.nlargest(3, 'abs_rel_prem_diff')
    rel_smallest_abs_price_diff = df.nsmallest(3, 'abs_rel_prem_diff')
    rel_smallest_abs_price_diff = rel_smallest_abs_price_diff.sort_values(by='abs_rel_prem_diff', ascending=False)
    frames = [rel_largest_abs_price_diff, rel_smallest_abs_price_diff]
    result_rel = pd.concat(frames)
    result_rel = result_rel.drop(['abs_prem_diff', 'abs_rel_prem_diff'], axis=1)
    print(result_rel.to_latex(index=False))

def box_prem_diff(df):
    plt.figure()
    fig, ax = plt.subplots()
    sns.boxplot(data=df["prem_diff"], color='blue')
    ax.set_ylabel("pred_price - exact_price")
    plt.savefig("prem_diff", dpi=400)

def box_prem_diff_rel(df):
    plt.figure()
    fig, ax = plt.subplots()
    sns.boxplot(data=df["rel_prem_diff"], color='red')
    ax.set_ylabel("pred_price - exact_price / exact_price ")
    plt.savefig("prem_diff_rel", dpi=400)

def statistic(df):  
    # median 
    median_abs_prem_diff = df["abs_prem_diff"].median()
    median_rel_abs_prem_diff = df["abs_rel_prem_diff"].median()
    print('median_abs:', median_abs_prem_diff )
    print('median_rel:', median_rel_abs_prem_diff)

    # mean 
    mean_abs_prem_diff = df["abs_prem_diff"].mean()
    mean_rel_abs_prem_diff = df["abs_rel_prem_diff"].mean()
    print('mean_abs:', mean_abs_prem_diff )
    print('mean_rel:', mean_rel_abs_prem_diff)

    # std 
    std_abs_prem_diff = df["abs_prem_diff"].std()
    std_rel_abs_prem_diff = df["abs_rel_prem_diff"].std()
    print('std_abs:', std_abs_prem_diff )
    print('std_rel:', std_rel_abs_prem_diff)

    
if __name__=="__main__":
    #create_pandas_dataframe()
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    df = pd.read_csv('pandas_test_2.csv')
    #print(statistic(df))
    #print(make_table(df))
    #box_prem_diff(df)
    #box_prem_diff_rel(df)
    statistic(df)