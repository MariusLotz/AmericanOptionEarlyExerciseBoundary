import numpy as np
import scipy.stats as stats
from ML import Base

def create_trainings_data():
    """ 2750 data points 
    sigma = 0.05, 0.1, ..., 0.05
    r,q = 0.02, 0.04, ..., 0.1
    S = 50, 60, ..., 150 """
    T = 1
    K = 100
    S = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    SIGMA = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    RQ = [0.02, 0.04, 0.06, 0.08, 0.1]
    file = open('test_data', 'a')
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
                    premium = gaussian_premium(r, q, sigma, K, S, T, tau_vec, boundary_vec, w_vec, T, option_type='Call')
                    entry = [r, q, sigma, list(boundary_vec), [S, premium]]
                    file.write(str(entry))
                    file.write("\n")
        file.close()

if __name__== "__main__":
    create_trainings_data()