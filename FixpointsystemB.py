import scipy.stats as stats
import numpy as np
import scipy.integrate as si

class FixpointsystemB():

    def d_minus(self, tau, X):
        """d_- from Black Scholes formula"""
        return (np.log(X) + (self.r_intern - self.q_intern) * tau - 0.5 * self.sigma ** 2 * tau) / (self.sigma * np.sqrt(tau))

    def d_plus(self, tau, X):
        """d_+ from Black Scholes formula"""
        return (np.log(X) + (self.r_intern - self.q_intern) * tau + 0.5 * self.sigma ** 2 * tau) / (self.sigma * np.sqrt(tau))

    def N(self, tau):
        """N term of fixpoint scheme"""
        def integrand(u):
            return np.exp(self.r_intern * u) * stats.norm.cdf(self.d_minus(tau - u, self.B(tau) / self.B(u)))

        a = stats.norm.cdf(self.d_minus(tau, self.B(tau) / self.K))

        return a + self.r_intern * si.fixed_quad(integrand, 0, tau, n=self.integration_base)[0]

    def D(self, tau):
        """D term of fixpoint scheme"""
        def integrand(u):
            return np.exp(self.q_intern * u) * stats.norm.cdf(self.d_plus(tau - u, self.B(tau) / self.B(u)))
        a = stats.norm.cdf(self.d_plus(tau, self.B(tau) / self.K))

        return a + self.q_intern * si.fixed_quad(integrand, 0, tau, n=self.integration_base)[0]