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
import pandas as pd
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

def put_call_symmetry_trick(boundary, K=100):
    new_boundary = []
    n = len(boundary)
    for i in range(n):
        new_boundary.append((boundary[i] + (K**2 /boundary[n-1-i]))/2)
    return new_boundary



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
        boundary = put_call_symmetry_trick(approx_Early_Exercise_Boundary_list[i])
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
    df.to_csv('pandas_test_2_trick.csv', index=False)
    #print(df)
    #return df

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
    df = pd.read_csv('pandas_test_2_trick.csv')
    print(statistic(df))