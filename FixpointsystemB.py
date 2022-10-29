import scipy.stats as stats
import numpy as np
import scipy.integrate as si

class FixpointsystemB():
    """Collection of functions defining a fixpointsystem
    for the American Option Early Exercise Boundary
    """

    def _d_minus(self, tau, X):
        """d_- from Black Scholes formula"""
        return (np.log(X) + (self.r - self.q) * tau - 0.5 * self.sigma ** 2 * tau) / (self.sigma * np.sqrt(tau))

    def _d_plus(self, tau, X):
        """d_+ from Black Scholes formula"""
        return (np.log(X) + (self.r - self.q) * tau + 0.5 * self.sigma ** 2 * tau) / (self.sigma * np.sqrt(tau))

    def _N(self, tau):
        """N term of fixpoint scheme"""
        def integrand(u):
            if self.option_type == 'Put':
                return np.exp(self.r * u) * stats.norm.cdf(self._d_minus(tau - u, self._B(tau) / self._B(u)))
            else:
                return np.exp(self.r * u) * stats.norm.cdf(-self._d_minus(tau - u, self._B(tau) / self._B(u)))
        if self.option_type == 'Put':
            a = stats.norm.cdf(self._d_minus(tau, self._B(tau) / self.K))
        else:
            a = stats.norm.cdf(-self._d_minus(tau, self._B(tau) / self.K))
        return a + self.r * si.fixed_quad(integrand, 0, tau, n=self._integration_base)[0]

    def _D(self, tau):
        """D term of fixpoint scheme"""
        def integrand(u):
            if self.option_type == 'Put':
                return np.exp(self.q * u) * stats.norm.cdf(self._d_plus(tau - u, self._B(tau) / self._B(u)))
            else:
                return np.exp(self.q * u) * stats.norm.cdf(-self._d_plus(tau - u, self._B(tau) / self._B(u)))

        if self.option_type == 'Put':
            a = stats.norm.cdf(self._d_plus(tau, self._B(tau) / self.K))
        else:
            a = stats.norm.cdf(-self._d_plus(tau, self._B(tau) / self.K))
        return a + self.q * si.fixed_quad(integrand, 0, tau, n=self._integration_base)[0]

    def _B_plus(self, tau, eta=0.9):
        """fixpoint scheme for one tau, creating B_plus = F(B)"""
        k = self.K * np.exp(-(self.r - self.q) * tau)
        B_plus = self._B(tau) - eta * (self._B(tau) - k * self._N(tau) / self._D(tau))
        diff = abs(B_plus - self._B(tau))
        if diff > self.max_diff:
            self.max_diff = diff
            #print(diff)
            #print(self.used_iteration_steps)
        return B_plus

    def _B_plus_aggressive_cut_off(self, tau):
        """fixpoint scheme for one tau, creating B_plus = F(B), not finished""" # NOTF
        eta = 1.6
        k = self.K * np.exp(-(self.r - self.q) * tau)
        nabla = (self._B(tau) - k * self._N(tau) / self._D(tau))
        B_plus = self._B(tau) - eta * nabla
        diff = abs(B_plus - self._B(tau))
        if diff > self.max_diff:
            self.max_diff = diff
            eta = 0.9
            B_plus = self._B(tau) - eta * (self._B(tau) - k * self._N(tau) / self._D(tau))
        return B_plus

    def _B_plus_aggressive_eta(self, tau):
        """fixpoint scheme for one tau, creating B_plus = F(B), not properly tested,
        no use for high precision pricing, otherwise when working seems faster than other etas"""
        eta = 1.6
        k = self.K * np.exp(-(self.r - self.q) * tau)
        B_plus = self._B(tau) - eta * (self._B(tau) - k * self._N(tau) / self._D(tau))
        diff = abs(B_plus - self._B(tau))
        if diff > self.max_diff:
            self.max_diff = diff
        return B_plus

    def _B_plus_moving_eta(self, tau):
        """fixpoint scheme for one tau, creating B_plus = F(B),
        not adjusted yet, dont use.""" #NOTF
        if self.used_iteration_steps <= 7:
            if self.max_diff >= 5:
                eta = 0.1
            elif self.max_diff < 5 and self.max_diff > 0.5:
                eta = 0.5 * 1 / self.max_diff
            else:
                eta = 1.3
        else:
            eta = 1
        k = self.K * np.exp(-(self.r - self.q) * tau)
        B_plus = self._B(tau) - eta * (self._B(tau) - k * self._N(tau) / self._D(tau))
        diff = abs(B_plus - self._B(tau))
        if diff > self.max_diff:
            self.max_diff = diff
        return B_plus