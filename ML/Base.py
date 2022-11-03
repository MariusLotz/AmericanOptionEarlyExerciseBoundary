import numpy as np
import scipy.stats as stats
import Option_Solver as os

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