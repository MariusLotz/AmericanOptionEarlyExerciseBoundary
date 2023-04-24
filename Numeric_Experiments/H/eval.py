import tensorflow as tf
import numpy as np
import os
import __init__
import Solver.Option_Solver as op
import Machine_Learning.Base as Base

def load_and_return_trainingsdata():
    """Loading data from txt. file and return trainingsdata"""
    file = open("test_data")
    lines = file.readlines()
    data = []
    for line in lines:
        try:
            [r, q, sigma, boundary, [S, prem]] = eval(line)
            data.append([r, q, sigma, boundary, [S, prem]])
        except: break
    return data

def create_prem_prices():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    data = load_and_return_trainingsdata()
    model_boundary = tf.keras.models.load_model("test_model_boundary")
    model_price = tf.keras.models.load_model("test_model_price")
    T = 1
    K = 100
    option_type = "Put"
    eval_data_list = []
    file = open('eval_data', 'w')
    counter = 0
    for d in data:
        counter += 1
        print(counter)
        [r, q, sigma, boundary, [S, prem]]  = d
        option = op.Option_Solver(r, q, sigma, K, T, option_type, n=11)
        option.create_boundary()
        tau_vec, boundary, w_vec = option.gaussian_grid_boundary(n=11)
        real_prem = Base.gaussian_premium(r, q, sigma, K, S, T, tau_vec, boundary, w_vec, T, option_type)
        #print(real_prem - prem)

        ### approximations:
        #print(model_boundary.predict([[r, q, sigma]])[0].tolist())
        model_boundary_prem = Base.gaussian_premium(r, q, sigma, K, S, T, tau_vec, model_boundary.predict([[r, q, sigma]])[0].tolist(), w_vec, T, option_type)
        eval_data = [r, q, sigma, S, prem, model_boundary_prem, model_price.predict([[r, q, sigma, S]])[0,0]]
        #print(eval_data)
        file.write(str(eval_data))
        file.write("\n")       
        #eval_data_list.append(eval_data)
    file.close()      

if __name__ == "__main__":
    create_prem_prices()
    

