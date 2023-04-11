import numpy as np
import os
import __init__
import Solver.Option_Solver as op
import Machine_Learning.Base as Base

def load_and_return_data():
    """Loading data from txt. file and return trainingsdata"""
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    print("hi")
    file = open("exercise_curve")
    lines = file.readlines()
    data = []
    for line in lines:
        try:
            [r, q, sigma, boundary, tau_vec, w_vec] = eval(line)
            data.append([r, q, sigma, boundary, tau_vec, w_vec])
        except: break
    return data

def create_error_data(data):
    K=100
    T=1
    S=80
    file = open('error_data_itm', 'w')
    option_type = "Put"
    count=0
    for d in data:
        count += 1
        print(count)
        [r, q, sigma, boundary, tau_vec, w_vec] = d
        prem = Base.gaussian_premium(r, q, sigma, K, S, T, tau_vec, boundary, w_vec, T, option_type)
        for p in [-2**(-7), -2**(-5), -2**(-3), -2**(-2), -2**(-1), -1, -2, -4, -8,
                  2**(-7), 2**(-5), 2**(-3), 2**(-2), 2**(-1), 1, 2, 4, 8]:
            wrong_boundary = []
            for i in range(len(boundary)):
                wrong_boundary.append(boundary[i] + (p/100) * boundary[i])
            wrong_prem = Base.gaussian_premium(r, q, sigma, K, S, T, tau_vec, wrong_boundary, w_vec, T, option_type)
            entry = [r, q, sigma, p, prem, wrong_prem]
            file.write(str(entry))
            file.write("\n")
    file.close()

def create_error_data_2(data):
    K=100
    T=1
    S=100
    file = open('error_data_atm_2', 'w')
    option_type = "Put"
    np.random.seed(1)
    for d in data:
        [r, q, sigma, boundary, tau_vec, w_vec] = d
        prem = Base.gaussian_premium(r, q, sigma, K, S, T, tau_vec, boundary, w_vec, T, option_type)
        for p in [2**(-7), 2**(-5), 2**(-3), 2**(-2), 2**(-1), 2**(-0), 2**(1), 2**(2), 2**(3)]:
            wrong_boundary = []
            for i in range(len(boundary)):
                c = np.random.normal(loc=0, scale=((p/100) * boundary[i])* 2)
                y = boundary[i] + c
                wrong_boundary.append(y)
            wrong_prem = Base.gaussian_premium(r, q, sigma, K, S, T, tau_vec, wrong_boundary, w_vec, T, option_type)
            entry = [r, q, sigma, p, prem, wrong_prem]
            file.write(str(entry))
            file.write("\n")
    file.close()

if __name__=="__main__":
    data = load_and_return_data()
    #create_error_data(data)
    create_error_data_2(data)