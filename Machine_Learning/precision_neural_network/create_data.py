import numpy as np
import sys
import os
myDir = os.getcwd()
sys.path.append(myDir)
from pathlib import Path
path = Path(myDir)
a=str(path.parent.absolute())
sys.path.append(a)
import Solver.Option_Solver as Solver  

def change_order(vec):
    out_vec = []
    n = len(vec)
    for i in range(n):
        out_vec.append(vec[n-i-1])
    return out_vec


def gen_curve(K, T, r, q, sigma):
    option = Solver.Option_Solver(r, q, sigma, K, T, option_type="Put", m=25)
    option.create_boundary()
    put_vec = option.Early_exercise_vec
    call_vec = [(K**2 / x) for x in put_vec] # Put-Call-Symmetrie
    call_vec.reverse()
    put_vec.extend(call_vec)
    #print(put_vec)
    return r, q, sigma, put_vec


def create_training_data(number_of_datapoints):
    """ 10.000 data points randomly out of
    sigma = [0.1, 0.5]
    r,q = [0.01, 0.1]
    K=100
    T=1
    """
    K=100
    T=1
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    file = open('exercise_curve_train', 'w')
    np.random.seed(1)
    for i in range(number_of_datapoints):
        if i % 100 == 0: print(i)
        r = np.random.uniform(0.01, 0.1)
        q = np.random.uniform(0.01, 0.1)
        sigma = np.random.uniform(0.1, 0.5)
        r, q, sigma, curve = gen_curve(K, T, r, q, sigma)
        entry = [r, q, sigma, curve]
        file.write(str(entry))
        file.write("\n")
    file.close()

def create_test_data(number_of_datapoints):
    """ 1000 data points randomly out of
    sigma = [0.1, 0.5]
    r,q = [0.01, 0.1]
    K=100
    T=1
    """
    K=100
    T=1
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    file = open('test_data_2', 'w')
    np.random.seed(1)
    i = 0
    while i < number_of_datapoints:
        if i % 50 == 0: print(i)
        r = np.random.uniform(0.01, 0.1)
        q = np.random.uniform(0.01, 0.1)
        sigma = np.random.uniform(0.1, 0.5)
        S = np.random.uniform(50, 150)
        option = Solver.Option_Solver(r, q, sigma, K, T, option_type="Put", m=25)
        option.create_boundary()
        prem = option.premium(S, T)
        if prem < 10**(-3):
            continue
        else:
            entry = [r, q, sigma, S, prem]
            file.write(str(entry))
            file.write("\n")
            i += 1
    file.close()


if __name__=="__main__":
    create_test_data(1000)

