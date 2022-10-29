import scipy.stats as stats
import numpy as np
import scipy.integrate as si

class FixpointsystemB():
    """Collection of functions defining a fixpointsystem
    for the American Option Early Exercise Boundary
    """

    def _d_minus(self, tau, X):
        """d_- from Black Scholes formula"""
        return (np.log(X) + (self._r_internal - self._q_internal) * tau - 0.5 * self.sigma ** 2 * tau) / (self.sigma * np.sqrt(tau))

    def _d_plus(self, tau, X):
        """d_+ from Black Scholes formula"""
        return (np.log(X) + (self._r_internal - self._q_internal) * tau + 0.5 * self.sigma ** 2 * tau) / (self.sigma * np.sqrt(tau))

    def _N(self, tau):
        """N term of fixpoint scheme"""
        def integrand(u):
            if self._internal_option_type == 'Put':
                return np.exp(self._r_internal * u) * stats.norm.cdf(self._d_minus(tau - u, self._B(tau) / self._B(u)))
            else:
                return np.exp(self._r_internal * u) * stats.norm.cdf(-self._d_minus(tau - u, self._B(tau) / self._B(u)))
        if self._internal_option_type == 'Put':
            a = stats.norm.cdf(self._d_minus(tau, self._B(tau) / self.K))
        else:
            a = stats.norm.cdf(-self._d_minus(tau, self._B(tau) / self.K))
        return a + self._r_internal * si.fixed_quad(integrand, 0, tau, n=self._integration_base)[0]

    def _D(self, tau):
        """D term of fixpoint scheme"""
        def integrand(u):
            if self._internal_option_type == 'Put':
                return np.exp(self._q_internal * u) * stats.norm.cdf(self._d_plus(tau - u, self._B(tau) / self._B(u)))
            else:
                return np.exp(self._q_internal * u) * stats.norm.cdf(-self._d_plus(tau - u, self._B(tau) / self._B(u)))

        if self._internal_option_type == 'Put':
            a = stats.norm.cdf(self._d_plus(tau, self._B(tau) / self.K))
        else:
            a = stats.norm.cdf(-self._d_plus(tau, self._B(tau) / self.K))
        return a + self._q_internal * si.fixed_quad(integrand, 0, tau, n=self._integration_base)[0]

    def _B_plus(self, tau, eta=0.9):
        """fixpoint scheme for one tau, creating B_plus = F(B)"""
        k = self.K * np.exp(-(self._r_internal - self._q_internal) * tau)
        B_plus = self._B(tau) - eta * (self._B(tau) - k * self._N(tau) / self._D(tau))
        return B_plus