""" QD+ Approximation as in Page 44
sigma, r and tau can not be zero.
Boundary must be lager than P
"""
import numpy as np
import scipy.optimize as sco
import scipy.stats as stats
import BS_formulas as bs

P = 0.01  # minimal boundary value allowed


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
    h = 1 - np.exp(-r * tau)
    w = 2 * (r - q) * sigma ** (-2)
    a = (l(h, w, r, sigma) + c0(h, w, S, K, r, q, sigma, tau)) * (K - S - bs.put_price(S, K, r, q, sigma, tau))
    return -np.exp(-q * tau) * stats.norm.cdf(-bs.d_plus(S, K, r, q, sigma, tau)) + a / S + 1


def exercise_boundary(K, r, q, sigma, tau_points):
    """Calculating an approximative exercise boundary of an American Option for given points in time"""
    exercise_boundary = []
    for i in range(len(tau_points)-1):
        solution = sco.root_scalar(f=implicit_exercise_boundary_function, args=(K, r, q, sigma, tau_points[i]),
                                   method='ridder', bracket=[P, K])
        exercise_boundary.append(solution.root)
    exercise_boundary.append(min(K, r/q * K))
    return exercise_boundary

def exercise_boundary_singleton(K, r, q, sigma, tau):
    if abs(tau - 0) < 1.49e-05:
        return min(K, r/q * K)
    else:
        return sco.root_scalar(f=implicit_exercise_boundary_function, args=(K, r, q, sigma, tau),
                                   method='ridder', bracket=[P, K]).root