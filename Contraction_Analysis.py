import numpy as np
from scipy.stats import stats
import scipy.integrate as si
import Option_Solver as OS


def B_plus(option, tau, c, eta=1):
    k = option.K * np.exp(-(option.r - option.q) * tau)
    return (option.B(tau) + c - eta * (option.B(tau) - k * option.N(tau) / option.D(tau)))

def N(option, tau):
    def integrand(u):
        return np.exp(option.r * u) * stats.norm.cdf(option.d_minus(tau - u, option.B(tau) / option.B(u)))
    a = stats.norm.cdf(option.d_minus(tau, option.B(tau) / option.K))
    return a + option.r * si.fixed_quad(integrand, 0, tau, n=option.integration_base)[0]

def D(option, tau):
    def integrand(u):
        return np.exp(option.q * u) * stats.norm.cdf(option.d_plus(tau - u, option.B(tau) / option.B(u)))
    a = stats.norm.cdf(option.d_plus(tau, option.B(tau) / option.K))
    return a + option.q * si.fixed_quad(integrand, 0, tau, n=option.integration_base)[0]

def contraction_val(r, q, sigma, K, T, c, on_QD_plus = True):
    def calc():
        sum1 = 0
        sum2 = 0
        i = -1
        for tau in option.tau_grid:
            i += 1
            sum1 += abs(option.B(tau) - option.B(tau) + c_vec[i])
            sum2 += abs(B_plus(option, tau) - B_plus(option, tau + c_vec[i]))
        k = sum2 / sum1
        return k

    option = OS.Option_Solver(r, q, sigma, K, T)
    c_vec = np.random.normal(loc=0.0, scale=option.B_at_zero * c, size=option.integration_base)
    if on_QD_plus:
        return calc()

    else:
        option.create_boundary()
        return calc()

def main():
    RQ = [0.01, 0.02, 0.03, 0.04, 0.05]
    SIGMA = [0.1, 0.2, 0.3, 0.4, 0.5]
    K, T = 100, 1
    c = 0.01
    for r in RQ:
        for q in RQ:
            for sigma in SIGMA:
                print(r, q, sigma, K, T, c, contraction_val(r, q, sigma, K, T, c))



if __name__ == "__main__":
    main()
