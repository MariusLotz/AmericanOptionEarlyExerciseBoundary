import scipy.stats as stats
import random
import __init__
import os
import Machine_Learning.Base as Base

def create_data():
    """ 1000 data points 
    sigma = 0.05, ..., 0.5
    r,q = 0.01, 0.02, ..., 0.1
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    T = 1
    K = 100
    SIGMA = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    RQ = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    file = open('exercise_curve', 'w')
    counter = 0
    option_type = "Put"
    for sigma in SIGMA:
        for r in RQ:
            for q in RQ:
                counter +=1
                print(counter)
                option = Base.os.Option_Solver(r, q, sigma, K, T, option_type, n=101)
                option.create_boundary()
                tau_vec, boundary_vec, w_vec = option.gaussian_grid_boundary(101)
                entry = [r, q, sigma, list(boundary_vec), list(tau_vec), list(w_vec)]
                file.write(str(entry))
                file.write("\n")
    file.close()

def test():
    """ 1000 data points 
    sigma = 0.05, ..., 0.5
    r,q = 0.01, 0.02, ..., 0.1
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    T = 1
    K = 100
    S = 100
    SIGMA = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    RQ = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    option_type = "Call"
    for sigma in SIGMA:
        for r in RQ:
            for q in RQ:
                option = Base.os.Option_Solver(r, q, sigma, K, T, option_type, n=101)
                option.create_boundary()
                tau_vec, boundary_vec, w_vec = option.gaussian_grid_boundary(101)
                x = Base.gaussian_premium(r, q, sigma, K, S, T, tau_vec, boundary_vec, w_vec, T, option_type)
                print(r, q, sigma, option.premium(S, T), x)
if __name__== "__main__":
    test()
    

    