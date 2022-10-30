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

def avg_boundary(boundary):
    return [np.mean(boundary) for x in boundary]

def main():
    r, q, sigma, K, T, option_type = 0.1, 0.1, 0.45, 150, 1, 'Call'
    option = os.Option_Solver(r, q, sigma, K, T, option_type)
    option.create_boundary()
    tau_vec, boundary_vec, w_vec = option.gaussian_grid_boundary(n=12)
    tau, S = T, 100
    """
    print(boundary_vec)
    for i in range(len(boundary_vec)):
        z = np.random.normal(loc=0.0, scale=0.000000001, size=None)
        boundary_vec[i] *= (1+z)
        """
    print(boundary_vec)
    print(avg_boundary(boundary_vec))
    price = gaussian_premium(r, q, sigma, K, S, tau, tau_vec, boundary_vec, w_vec, T, option_type)
    price2 = option.premium(S, tau)
    print((price - price2) / price2)
    price3 = gaussian_premium(r, q, sigma, K, S, tau, tau_vec, avg_boundary(boundary_vec), w_vec, T, option_type)
    print(price, price3)


if __name__=="__main__":
    main()