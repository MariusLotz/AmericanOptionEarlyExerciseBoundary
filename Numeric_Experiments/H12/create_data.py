import scipy.stats as stats
import random
import __init__
import os
import Machine_Learning.Base as Base

def create_trainings_data():
    """ 2750 data points 
    sigma = 0.05, 0.1, ..., 0.05
    r,q = 0.02, 0.04, ..., 0.1
    S = 50, 60, ..., 150 """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    T = 1
    K = 100
    S = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    SIGMA = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    RQ = [0.02, 0.04, 0.06, 0.08, 0.1]
    file = open('training_data', 'w')
    counter = 0
    option_type = "Put"
    for sigma in SIGMA:
        for r in RQ:
            for q in RQ:
                for s in S:
                    counter +=1
                    print(counter, " out of ", len(SIGMA) * len(RQ)**2 * len(S))
                    option = Base.os.Option_Solver(r, q, sigma, K, T, option_type)
                    option.create_boundary()
                    tau_vec, boundary_vec, w_vec = option.gaussian_grid_boundary(n=11)
                    premium = Base.gaussian_premium(r, q, sigma, K, s, T, tau_vec, boundary_vec, w_vec, T, option_type)
                    entry = [r, q, sigma, list(boundary_vec), [s, premium]]
                    file.write(str(entry))
                    file.write("\n")
    file.close()

def create_test_data():
    """1000 datapoints uniform randomly selected out of
    sigma = [0.1, 0.5],
    RQ = [0.02, 0.1],
    S = [50, 150]"""
    random.seed(1)
    SIGMA = [0.1, 0.5]
    RQ = [0.02, 0.1]
    S = [50, 150]
    T = 1
    K = 100
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    file = open('test_data', 'w')
    counter = 0
    option_type = "Put"
    for i in range(1000):
        counter += 1
        print(counter)
        sigma = random.uniform(SIGMA[0], SIGMA[1])
        r = random.uniform(RQ[0], RQ[1])
        q = random.uniform(RQ[0], RQ[1])
        s = random.uniform(S[0], S[1])
        option = Base.os.Option_Solver(r, q, sigma, K, T, option_type)
        option.create_boundary()
        tau_vec, boundary_vec, w_vec = option.gaussian_grid_boundary(n=11)
        premium = Base.gaussian_premium(r, q, sigma, K, s, T, tau_vec, boundary_vec, w_vec, T, option_type)
        entry = [r, q, sigma, list(boundary_vec), [s, premium]]
        file.write(str(entry))
        file.write("\n")
    file.close()

if __name__== "__main__":
    create_test_data()
    create_trainings_data()
    

    