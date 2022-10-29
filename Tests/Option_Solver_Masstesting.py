import Option_Solver as os
import random as rr


def masstesting():
    rr.seed(1)  # seed
    Z = 1 # passes
    for i in range(Z):
        r = rr.uniform(0, 0.1)
        q = rr.uniform(0, 0.1)
        sigma = rr.uniform(0, 1)
        K = rr.uniform(1, 99)
        S = rr.uniform(1, 999)
        T = rr.uniform(0.1, 9)
        tau = rr.uniform(0, T)
        print("r, q, sigma, K, S, T, tau: ", r, q, sigma, K, S, T, tau)
        option = os.Option_Solver(r, q, sigma, K, T)
        #print(option.premium(S, tau))
        option.create_boundary()
        #print(option.premium(S, tau))
        print(option.tau_grid)
        print(option.Early_exercise_vec)
        #print(option.QD_plus_exercise_vec)
        print(option.Early_exercise_curve(0.1))
        print(option._B(option.T))
        #print(option.Early_exercise_curve(2))



if __name__ =="__main__":
    masstesting()




