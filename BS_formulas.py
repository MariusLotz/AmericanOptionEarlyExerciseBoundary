"""Option Pricing via Black Scholes for a European Put Option"""
import numpy as np
import scipy.stats as stats


def d_minus(S, K, r, q, sigma, tau):
    return (np.log(S / K) + (r - q) * tau - 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))

def d_plus(S, K, r, q, sigma, tau):
    return (np.log(S / K) + (r - q) * tau + 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))

def put_price(S, K, r, q, sigma, tau):
    a = np.exp(-r * tau) * K * stats.norm.cdf(-d_minus(S, K, r, q, sigma, tau))
    b = np.exp(-q * tau) * S * stats.norm.cdf(-d_plus(S, K, r, q, sigma, tau))
    return a - b

def call_price(S, K, r, q, sigma, tau):
    a = np.exp(-r * tau) * K * stats.norm.cdf(d_minus(S, K, r, q, sigma, tau))
    b = np.exp(-q * tau) * S * stats.norm.cdf(d_plus(S, K, r, q, sigma, tau))
    return b - a

def theta(S, K, r, q, sigma, tau):
    """time derivative of the option price"""
    a = r * K * np.exp(r * tau) * stats.norm.cdf(-d_minus(S, K, r, q, sigma, tau))
    b = q * S * np.exp(q * tau) * stats.norm.cdf(-d_plus(S, K, r, q, sigma, tau))
    c = sigma * S * 0.5 * (1/np.sqrt(tau)) * np.exp(-q * tau) * stats.norm.pdf(d_plus(S, K, r, q, sigma, tau))
    return a - b - c
