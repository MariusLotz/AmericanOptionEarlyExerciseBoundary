import numpy as np
import Machine_Learning.B_Nearest_Neighbour as B
import Solver.Option_Solver as os
import Machine_Learning.Base as Base

def first_test():
    r, q, sigma = 0.0345, 0.0478, 0.378
    instance = B.B_Nearest_Neigbour(r, q, sigma)

    print(instance.neighbours)
    print()
    print(instance.boundary)
    print()
    print(instance.price(100, 2))

def mass_test(size=100):
    K = 100
    T = 1
    np.random.seed(seed=1)
    SIGMA = np.random.uniform(0.1, 0.5, size)
    RQ = np.random.uniform(0.02, 0.1, size)
    S = np.random.uniform(80, 120, size)

    #r, q, sigma, S = RQ[0], RQ[-1], SIGMA[0], S[0]
    for r in RQ:
        for i in range(len(RQ)):
            for sigma in SIGMA:
                for s in S:
                    q = RQ[len(RQ) - i - 1]
                    # B_Nearest_Neighbour_price:
                    B_nearest = B.B_Nearest_Neigbour(r, q, sigma)
                    boundary1 = B_nearest.boundary
                    price1 = B_nearest.price(s, T)

                    # real price:
                    option = os.Option_Solver(r, q, sigma, K, T, 'Call')
                    option.create_boundary()
                    tau_vec, boundary_vec, w_vec = option.gaussian_grid_boundary(n=10)
                    boundary2 = boundary_vec
                    price2 = Base.gaussian_premium(r, q, sigma, K, s, T, tau_vec, boundary2, w_vec, T, option_type='Call')

                    print("r=", RQ[0], " q=", RQ[-1], " sigma=", SIGMA[0], " S=", s)
                    print('approx price =', price1, "        ", 'price= ', price2)
                    #print()
                    #print(boundary1)
                    #print(boundary2)
                    #print()
                    #print(B_nearest.neighbouring_boundaries)


if __name__=="__main__":
    mass_test()