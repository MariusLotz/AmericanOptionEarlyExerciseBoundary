import scipy.stats as stats
import numpy as np
import scipy.integrate as si

class FixpointsystemB():
    """Collection of functions defining a fixpointsystem
    for the American Option Early Exercise Boundary
    """

    def d_minus(self, tau, X):
        """d_- from Black Scholes formula"""
        return (np.log(X) + (self.r_internal - self.q_internal) * tau - 0.5 * self.sigma ** 2 * tau) / (self.sigma * np.sqrt(tau))

    def d_plus(self, tau, X):
        """d_+ from Black Scholes formula"""
        return (np.log(X) + (self.r_internal - self.q_internal) * tau + 0.5 * self.sigma ** 2 * tau) / (self.sigma * np.sqrt(tau))

    def N(self, tau):
        """N term of fixpoint scheme"""
        def integrand(u):
            if self.internal_option_type == 'Put':
                return np.exp(self.r_internal * u) * stats.norm.cdf(self.d_minus(tau - u, self.B(tau) / self.B(u)))
            else:
                return np.exp(self.r_internal * u) * stats.norm.cdf(-self.d_minus(tau - u, self.B(tau) / self.B(u)))
        if self.internal_option_type == 'Put':
            a = stats.norm.cdf(self.d_minus(tau, self.B(tau) / self.K))
        else:
            a = stats.norm.cdf(-self.d_minus(tau, self.B(tau) / self.K))
        return a + self.r_internal * si.fixed_quad(integrand, 0, tau, n=self.integration_base)[0]

    def D(self, tau):
        """D term of fixpoint scheme"""
        def integrand(u):
            if self.internal_option_type == 'Put':
                return np.exp(self.q_internal * u) * stats.norm.cdf(self.d_plus(tau - u, self.B(tau) / self.B(u)))
            else:
                return np.exp(self.q_internal * u) * stats.norm.cdf(-self.d_plus(tau - u, self.B(tau) / self.B(u)))

        if self.internal_option_type == 'Put':
            a = stats.norm.cdf(self.d_plus(tau, self.B(tau) / self.K))
        else:
            a = stats.norm.cdf(-self.d_plus(tau, self.B(tau) / self.K))
        return a + self.q_internal * si.fixed_quad(integrand, 0, tau, n=self.integration_base)[0]

    def B_plus(self, tau, eta=0.9):
        """fixpoint scheme for one tau, creating B_plus = F(B)"""
        k = self.K * np.exp(-(self.r_internal - self.q_internal) * tau)
        B_plus = self.B(tau) - eta * (self.B(tau) - k * self.N(tau) / self.D(tau))
        return B_plus