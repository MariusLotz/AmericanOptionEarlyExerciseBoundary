import numpy as np
import __init__
import Solver.Option_Solver as Solver        

def gen_curves():
    K=100
    T=1
    option_type = 'Put'

    np.random.seed(1)
    file = open('curve_train', 'w')
    for i_ in range(1000):
        r = np.random.uniform(0, 0.1)
        q = np.random.uniform(0, 0.1)
        sigma = np.random.uniform(0, 0.5)
        option = Solver.Option_Solver(r, q, sigma, K, T, option_type, n=11)
        option.create_boundary()
        curve_vec = option.Early_exercise_vec
        entry = [r, q, sigma, curve_vec]
        file.write(str(entry))
        file.write("\n")
    file.close()

    np.random.seed(2)
    file = open('curve_test', 'w')
    for i_ in range(1000):
        r = np.random.uniform(0, 0.1)
        q = np.random.uniform(0, 0.1)
        sigma = np.random.uniform(0, 0.5)
        option = Solver.Option_Solver(r, q, sigma, K, T, option_type, n=11)
        option.create_boundary()
        curve_vec = option.Early_exercise_vec
        entry = [r, q, sigma, curve_vec]
        file.write(str(entry))
        file.write("\n")
    file.close()

def gen_prices():
    K=100
    T=1
    option_type = 'Put'

    np.random.seed(1)
    file = open('price_train', 'w')
    for i_ in range(1000):
        r = np.random.uniform(0, 0.1)
        q = np.random.uniform(0, 0.1)
        sigma = np.random.uniform(0, 0.5)
        S = np.random.uniform(50, 150)
        option = Solver.Option_Solver(r, q, sigma, K, T, option_type, n=11)
        option.create_boundary()
        American_premium = option.premium(S, T)
        entry = [r, q, sigma, S, American_premium]
        file.write(str(entry))
        file.write("\n")
    file.close()

    np.random.seed(2)
    file = open('price_test', 'w')
    for i_ in range(1000):
        r = np.random.uniform(0, 0.1)
        q = np.random.uniform(0, 0.1)
        sigma = np.random.uniform(0, 0.5)
        S = np.random.uniform(50, 150)
        option = Solver.Option_Solver(r, q, sigma, K, T, option_type, n=11)
        option.create_boundary()
        American_premium = option.premium(S, T)
        entry = [r, q, sigma, S, American_premium]
        file.write(str(entry))
        file.write("\n")
    file.close()
         
if __name__=="__main__":
    gen_curves()
    gen_prices()
