import numpy as np
import scipy.stats as stats
import Option_Solver as os
import tensorflow as tf

def d_minus(r, q, sigma, tau, X):
    """d_- from Black Scholes formula"""
    return (np.log(X) + (r - q) * tau - 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))

def d_plus(r, q, sigma,  tau, X):
    """d_+ from Black Scholes formula"""
    return (np.log(X) + (r - q) * tau + 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))

def gaussian_premium(r, q, sigma, K, S, tau, tau_vec, boundary_vec, w_vec, T, option_type):
    def integrand(u, Bu):
        z = S / Bu
        if option_type == 'Put':
            a = r * K * np.exp(-r * (tau - u)) * stats.norm.cdf(-d_minus(r, q, sigma, tau - u, z))
            b = q * S * np.exp(-q * (tau - u)) * stats.norm.cdf(-d_plus(r, q, sigma, tau - u, z))
        else:  # Call:
            a = q * S * np.exp(-q * (tau - u)) * stats.norm.cdf(d_plus(r, q, sigma, tau - u, z))
            b = r * K * np.exp(-r * (tau - u)) * stats.norm.cdf(d_minus(r, q, sigma, tau - u, z))
        return a - b
    sum = 0
    for i in range(len(w_vec)):
        sum += integrand(tau_vec[i], boundary_vec[i])* w_vec[i]
    return  T/2.0 * sum

def load_and_return_trainingsdata(file_path):
    """Loading data from txt. file and return trainingsdata"""
    file = open(file_path)
    lines = file.readlines()
    x_train = []
    y_train = []
    for line in lines:
        [r, q, sigma, boundary, [s, premium]] = eval(line)
        x_train.append([r,q, sigma])
        y_train.append(boundary)

    return x_train, y_train

def test_model_on_training_data(model, x_train, y_train, S=[80, 120]):
    option_type = 'Call'
    K= 100
    T= 1
    option = os.Option_Solver(0.05, 0.05, 0.35, K, T, option_type)
    option.create_boundary()
    tau_vec, boundary_vec, w_vec = option.gaussian_grid_boundary(n=10)
    for i in range(len(x_train)):
        x = tf.constant([x_train[i]])
        pred_boundary = model(x).numpy()[0]
        boundary = y_train[i]
        print(boundary)
        print()
        print(pred_boundary)
        print()
        print()
        print()
        for s in S:
            pred_prem = gaussian_premium(x_train[i][0], x_train[i][1], x_train[i][2], K, s, T, tau_vec, pred_boundary, w_vec, T, option_type)
            prem = gaussian_premium(x_train[i][0], x_train[i][1], x_train[i][2], K, s, T, tau_vec, boundary, w_vec, T, option_type)
            #print("pred_prem= ", pred_prem, "prem= ", prem, "(pred - prem) / pred = ", (pred_prem-prem)/pred_prem)

def create_trainings_data():
    T = 1
    K = 100
    S = 100
    SIGMA = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    RQ = [0.02, 0.04, 0.06, 0.08, 0.1]
    file = open('Small_Sample_3_11', 'a')
    counter = 0
    for sigma in SIGMA:
        for r in RQ:
            for q in RQ:
                counter +=1
                print(counter, " out of ", len(SIGMA) * len(RQ)**2)
                option = os.Option_Solver(r, q, sigma, K, T, 'Call')
                option.create_boundary()
                tau_vec, boundary_vec, w_vec = option.gaussian_grid_boundary(n=10)
                premium = gaussian_premium(r, q, sigma, K, S, T, tau_vec, boundary_vec, w_vec, T, option_type='Call')
                entry = [r, q, sigma, list(boundary_vec), [S, premium]]
                file.write(str(entry))
                file.write("\n")
    file.close()
if __name__=="__main__":
    create_trainings_data()