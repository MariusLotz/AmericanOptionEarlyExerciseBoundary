import numpy as np
import scipy.stats as stats
import scipy.integrate as si
import Chebyshev
import QD_plus_exercise_boundary as QD_plus
import BS_formulas as B

class Option_Solver():
    def __init__(self, interest_rate, dividend, volatility, strike, maturity, l=113, m=21, n=15):
        self.r = interest_rate
        self.q = dividend
        self.sigma = volatility
        self.K = strike
        self.T = maturity
        self.interpolation_base = m
        self.integration_base = l
        self.iteration_steps = n
        self.B_at_zero = self.K * min(1, self.r / self.q)
        self.__initial()

    def __initial(self):
        """setting up stuff"""
        self.cheby_H = Chebyshev.Interpolation(self.interpolation_base, np.sqrt(self.T), 0)
        self.tau_grid = [x ** 2 for x in self.cheby_H.cheby_points]
        self.QD_plus_B_vec = QD_plus.exercise_boundary(self.K, self.r, self.q, self.sigma, self.tau_grid)
        self.Btau_vec = self.QD_plus_B_vec
        self.cheby_H.fit_by_y_values([self.__H(B) for B in self.Btau_vec])
        self.B = lambda tau : self.__H_inverse(self.cheby_H.value(np.sqrt(tau)))
        self.Btau_vec_new = [0] * len(self.Btau_vec)
        self.B_new = None

    def create_boundary(self):
        """calculate the boundary function of an American Option"""
        for j in range(self.iteration_steps):
            # Fixpoint Iteration:
            for i in range(self.interpolation_base - 1):
                self.Btau_vec_new[i] = self.__B_plus(self.tau_grid[i]) # Iteration per tau
            self.Btau_vec_new[-1] = self.B_at_zero
            self.cheby_H.fit_by_y_values([self.__H(B) for B in self.Btau_vec_new]) # create H-curve
            self.B_new = lambda tau : self.__H_inverse(self.cheby_H.value(np.sqrt(tau))) # create B

            self.B = self.B_new
            self.Btau_vec = self.Btau_vec_new

    def __B_plus(self, tau, eta=0.8):
        """fixpoint scheme for one tau, getting Btau_new"""
        k = self.K * np.exp(-(self.r - self.q) * tau)

        return (self.B(tau) - eta * (self.B(tau) - k * self.__N(tau) / self.__D(tau)))

    def __N(self, tau):
        """N term of fixpoint scheme"""
        def integrand(u):
            return np.exp(self.r * u) * stats.norm.cdf(self.__d_minus(tau - u, self.B(tau) / self.B(u)))

        a = stats.norm.cdf(self.__d_minus(tau, self.B(tau) / self.K))

        return a + self.r * si.fixed_quad(integrand, 0, tau, n=self.integration_base)[0]

    def __D(self, tau):
        """D term of fixpoint scheme"""
        def integrand(u):
            return np.exp(self.q * u) * stats.norm.cdf(self.__d_plus(tau - u, self.B(tau) / self.B(u)))
        a = stats.norm.cdf(self.__d_plus(tau, self.B(tau) / self.K))

        return a + self.q * si.fixed_quad(integrand, 0, tau, n=self.integration_base)[0]

    def __d_minus(self, tau, X):
        """d_- from Black Scholes formula"""
        return (np.log(X) + (self.r - self.q) * tau - 0.5 * self.sigma**2 * tau) / (self.sigma * np.sqrt(tau))

    def __d_plus(self, tau, X):
        """d_+ from Black Scholes formula"""
        return (np.log(X) + (self.r - self.q) * tau + 0.5 * self.sigma**2 * tau) / (self.sigma * np.sqrt(tau))

    def __H(self, B):
        """B -> ln(B/X)^2"""
        return np.log(B / self.B_at_zero)**2

    def __H_inverse(self, H):
        """H(x) -> X * exp(+-sqrt(H) """
        return self.B_at_zero * np.exp(-np.sqrt(H))

    def premium(self, S, tau):
        """American premium, self.B used"""
        def integrand(u):
            z = S / self.B(u)
            a = self.r * self.K * np.exp(-self.r * (tau - u)) * stats.norm.cdf(-self.__d_minus(tau - u, z))
            b = self.q * S * np.exp(-self.q * (tau - u)) * stats.norm.cdf(-self.__d_plus(tau - u, z))
            return a - b
        return si.fixed_quad(integrand, 0, tau, n=self.integration_base)[0]

    def QD_plus_premium(self, S, tau):
        """American premium, QD_plus_B_vec used"""
        cheby_QD_plus = Chebyshev.Interpolation(self.interpolation_base, np.sqrt(self.T), 0)
        cheby_QD_plus.fit_by_y_values([self.__H(B) for B in self.QD_plus_B_vec])  # create H-curve
        B_QD = lambda tau: self.__H_inverse(self.cheby_H.value(np.sqrt(tau)))  # create B
        def integrand(u):
            z = S / B_QD(u)
            a = self.r * self.K * np.exp(-self.r * (tau - u)) * stats.norm.cdf(-self.__d_minus(tau - u, z))
            b = self.q * S * np.exp(-self.q * (tau - u)) * stats.norm.cdf(-self.__d_plus(tau - u, z))
            return a - b
        return si.fixed_quad(integrand, 0, tau, n=self.integration_base)[0]

    def premium_via_boundary_vec(self, S, tau, boundary):
        """American premium for given boundary"""
        cheby_boundary = Chebyshev.Interpolation(self.interpolation_base, np.sqrt(self.T), 0)
        cheby_boundary.fit_by_y_values([self.__H(B) for B in boundary])  # create H-curve
        B_boundary = lambda tau: self.__H_inverse(self.cheby_H.value(np.sqrt(tau)))  # create B
        def integrand(u):
            z = S / B_boundary(u)
            a = self.r * self.K * np.exp(-self.r * (tau - u)) * stats.norm.cdf(-self.__d_minus(tau - u, z))
            b = self.q * S * np.exp(-self.q * (tau - u)) * stats.norm.cdf(-self.__d_plus(tau - u, z))
            return a - b
        return si.fixed_quad(integrand, 0, tau, n=self.integration_base)[0]

    def put_price(self, S, tau):
        if tau>self.T:
            raise ValueError("tau can not be larger than T")
        else:
            return self.premium(S, tau) + B.put_price(S, self.K, self.r, self.q, self.sigma, tau)

