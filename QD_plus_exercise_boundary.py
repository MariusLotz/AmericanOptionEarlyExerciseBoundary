""" QD+ Approximation as in Page 44
sigma, r and tau can not be zero.
Boundary must be lager than P
"""
import numpy as np
import scipy.optimize as sco
import scipy.stats as stats
import BS_formulas as bs

alpha = 1e-7 # minimal boundary value allowed
beta = 1e-07 # minimal value for sigma, r, q
gamma = 1e-06 # minimal value for tau
def l(h, w, r, sigma):
    """lambda in paper"""
    return 0.5 * (-(w - 1) - np.sqrt((w - 1) ** 2 + (8 * r / (sigma ** 2 * h))))

def ll(h, w, r, sigma):
    """lambda differential in paper"""
    return 2 * r / (sigma ** 2 * h ** 2 * np.sqrt((w - 1) ** 2 + (8 * r / (sigma ** 2 * h))))

def c0(h, w, S, K, r, q, sigma, tau):
    """c0 in paper"""
    a = ((1 - h) * 2 * r / sigma ** 2) / (2 * l(h, w, r, sigma) + w - 1)
    b = np.exp(r * tau) * bs.theta(S, K, r, q, sigma, tau) / (r * ((K - S) - bs.put_price(S, K, r, q, sigma, tau)))
    c = ll(h, w, r, sigma) / (2 * l(h, w, r, sigma) + w - 1)
    return -a * ((1 / h) - b + c)

def implicit_exercise_boundary_function(S, K, r, q, sigma, tau):
    """approximate implicit equation for boundary in paper"""
    if sigma < beta: sigma = beta  ##NOTF
    if r < beta: r = beta
    #if q < beta: q = beta
    if tau < gamma: tau = gamma
    h = 1 - np.exp(-r * tau)
    w = 2 * (r - q) * sigma ** (-2)
    a = (l(h, w, r, sigma) + c0(h, w, S, K, r, q, sigma, tau)) * (K - S - bs.put_price(S, K, r, q, sigma, tau))
    return -np.exp(-q * tau) * stats.norm.cdf(-bs.d_plus(S, K, r, q, sigma, tau)) + a / S + 1


def exercise_boundary(K, B_zero, r, q, sigma, tau_points, option_type):
    """Calculating an approximative exercise boundary of an American Option for given points in time"""
    exercise_boundary = []
    try:
        if option_type == 'Put':
            for i in range(len(tau_points)-1):
                solution = sco.root_scalar(f=implicit_exercise_boundary_function,
                                           args=(K, r, q, sigma, tau_points[i]),
                                           method='ridder', bracket=[alpha * K, K])
                exercise_boundary.append(solution.root)
        else:  # Put-Call-Symmetry Trick: B_C(tau, r, q) = K^2 / B_P(tau, q, r)
            for i in range(len(tau_points)-1):
                solution = sco.root_scalar(f=implicit_exercise_boundary_function,
                                           args=(K, q, r, sigma, tau_points[i]),
                                           method='ridder', bracket=[alpha * K, K])  ##NOTF
                bval = K ** 2 / solution.root
                exercise_boundary.append(bval)
    except:
        if option_type == 'Put':
            for i in range(len(tau_points)-1):
                solution = sco.root_scalar(f=implicit_exercise_boundary_function,
                                           args=(K, r, q, sigma, tau_points[i]),
                                           method='secant', x0 =alpha * K, x1= K)
                exercise_boundary.append(solution.root)
        else:  # Put-Call-Symmetry Trick: B_C(tau, r, q) = K^2 / B_P(tau, q, r)
            for i in range(len(tau_points)-1):
                solution = sco.root_scalar(f=implicit_exercise_boundary_function,
                                           args=(K, q, r, sigma, tau_points[i]),
                                           method='secant', x0 =alpha * K, x1= K)  ##NOTF
                bval = K ** 2 / solution.root
                exercise_boundary.append(bval)
    exercise_boundary.append(B_zero)
    return exercise_boundary

def exercise_boundary_singleton(K, r, q, sigma, tau):
    if abs(tau - 0) < 1.49e-07:
        return min(K, r/q * K)
    else:
        try:
            boundary_val = sco.root_scalar(f=implicit_exercise_boundary_function, args=(K, r, q, sigma, tau),
                                   method='ridder', bracket=[alpha * K, K]).root
        except:
            boundary_val =  sco.root_scalar(f=implicit_exercise_boundary_function,
                                           args=(K, r, q, sigma, tau),
                                           method='secant', x0 =alpha * K, x1= K)
        return boundary_val