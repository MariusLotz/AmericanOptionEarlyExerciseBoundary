import numpy as np
import scipy.stats as stats
import scipy.integrate as si
from scipy.special import roots_legendre
import Chebyshev
import QD_plus_exercise_boundary as QD_plus
import BS_formulas as B
from FixpointsystemB import FixpointsystemB

H_min = 1e-25  # min-parameter for H-transformed value due to overflow

class Option_Solver(FixpointsystemB):
    def __init__(self, interest_rate, dividend_yield, volatility, strike, maturity,
                 option_type = 'Put', l=55, m=21, n=15, stop_by_diff=1e-6): # set stop_by_diff=None for fixed n
        self.r = interest_rate
        self.q = dividend_yield
        self.sigma = volatility
        self.K = strike
        self.T = maturity
        self._interpolation_base = m  # number of basis points for the Chebyshev polynomial
        self._integration_base = l  # number of basis points for the Gauss quadrature
        self._iteration_steps = n  # number of fixpoint iteration steps
        self.Early_exercise_curve = None
        self.Early_exercise_vec = None
        self._Btau_vec = None
        self._B = None
        self._Btau_vec_new = None
        self._B_new = None
        self.used_iteration_steps = 0
        self.stop_by_diff = stop_by_diff
        self.max_diff_vec = []
        self.max_diff = -1.0
        self.option_type = option_type
        # B(0+) = K * max(1, r/q) (Call) and B(0+) = K * min(1, r/q) (Put) #
        if self.option_type == 'Put' and self.q > self.r:
            self._B_zero_plus = self.K * self.r / self.q
        elif self.option_type == 'Call' and self.q < self.r:
            self._B_zero_plus = self.K * self.r / self.q
        else:
            self._B_zero_plus = self.K
        self.__initial_boundary()
        if self.r < 1e-06 or self.sigma < 1e-06:
            print("""WARNING:
            Extrem small values for the interest rate (r) or volatility (sigma) are problematic! 
            """)

    def premium(self, S, tau, boundary = None): #NOTF
        """American premium for given boundary=exact_boundary of length."""
        if tau > self.T: raise ValueError("tau can not be larger than the maturity!")
        if tau < 0: raise ValueError("tau can not be negativ!")
        if S < 0: raise ValueError("stock price can not be negativ!")
        def integrand(u):
                z = S / boundary(u)
                if self.option_type == 'Put':
                    a = self.r * self.K * np.exp(-self.r * (tau - u)) * stats.norm.cdf(-self._d_minus(tau - u, z))
                    b = self.q * S * np.exp(-self.q * (tau - u)) * stats.norm.cdf(-self._d_plus(tau - u, z))
                else:  # Call:
                    a = self.q * S * np.exp(-self.q * (tau - u)) * stats.norm.cdf(self._d_plus(tau - u, z))
                    b = self.r * self.K * np.exp(-self.r * (tau - u)) * stats.norm.cdf(self._d_minus(tau - u, z))
                return a - b
        if type(boundary) == list:
            print("""
            make sure the boundary values are in order in regard to the '.tau_grid'
            """)
            _cheby_H = Chebyshev.Interpolation(self._interpolation_base, np.sqrt(self.T), 0)
            _cheby_H.fit_by_y_values([self._H(B) for B in boundary])  # create H-curve
            boundary = lambda tau: self._H_inverse(self._cheby_H.value(np.sqrt(tau)))  # define B-curve
        elif self.Early_exercise_curve == None:
            print("""
            QD_plus boundary was used. Therefore outcome will be an approximation only.
            Specify boundary you like to use, or run '.create_boundary()' first to use exact boundary!
            """)
            _cheby_H = Chebyshev.Interpolation(self._interpolation_base, np.sqrt(self.T), 0)
            _cheby_H.fit_by_y_values([self._H(B) for B in self.QD_plus_exercise_vec])  # create H-curve
            boundary = lambda tau: self._H_inverse(_cheby_H.value(np.sqrt(tau)))  # define B-curve
        else:
            boundary = self.Early_exercise_curve
        return max(si.fixed_quad(integrand, 0, tau, n=self._integration_base)[0], 0)

    def European_price(self, S, tau):
        """European optionprice via Black Scholes formula"""
        if tau < 0: raise ValueError("tau can not be negativ!")
        if S < 0: raise ValueError("stock price can not be negativ!")
        if self.option_type == 'Put':
            return B.put_price(S, self.K, self.r, self.q, self.sigma, tau)
        else:
            return B.call_price(S, self.K, self.r, self.q, self.sigma, tau)

    def American_price(self, S, tau, boundary=None):
        """American option price via American premium"""
        return self.premium(S, tau) + self.European_price(S, tau)

    def _loop_end(self, step):
        """check condition for ending main loop"""
        if type(self.stop_by_diff) == float:
            if self.stop_by_diff > self.max_diff:
                return True
            else:
                return False
        else:
            if self._iteration_steps <= step:
                return True
            else:
                return False

    def gaussian_grid_boundary(self, n):
        """returns triple tau, boundary, weigths"""
        x_vec, w_vec = roots_legendre(n)
        x_vec = np.real(x_vec)
        tau_vec = self.T * (x_vec + 1) / 2
        boundary_vec = self.Early_exercise_curve(tau_vec)
        return tau_vec, boundary_vec , w_vec

    def create_boundary(self):
        """calculate the boundary function of an American Option"""
        stop_now = False
        while stop_now == False:
            # Fixpoint Iteration for each tau:
            self.max_diff = 0
            for i in range(self._interpolation_base - 1):
                # might change B_plus if its not working...
                self._Btau_vec_new[i] = self._B_plus_aggressive_eta((self.tau_grid[i]))  # Iteration per tau
            self._Btau_vec_new[-1] = self._B_zero_plus  # B value close to maturity
            self._cheby_H.fit_by_y_values([self._H(B) for B in self._Btau_vec_new])  # create H-curve
            self._B_new = lambda tau : self._H_inverse(self._cheby_H.value(np.sqrt(tau)))  # create B-curve
            self._B = self._B_new
            self._Btau_vec = self._Btau_vec_new
            self.used_iteration_steps += 1
            stop_now = self._loop_end(self.used_iteration_steps)
            self.max_diff_vec.append(self.max_diff)
        self.Early_exercise_vec = self._Btau_vec
        self.Early_exercise_curve = lambda tau: self._B(tau)

    def value_match_condition_for_tau(self, boundary, tau):
        """Checking accuracy of given boundary"""
        if tau < 1e-63:
            tau = 1e-31
        return (self.K - boundary(tau)) - self.American_price(boundary(tau), tau, boundary)

    def value_match_condition_vec(self, boundary = None, tau_grid = None):
        """Accuracy vector of given boundary"""
        if tau_grid == None: tau_grid = self.tau_grid
        if boundary == None: boundary = self.Early_exercise_curve
        vec = [self.value_match_condition_for_tau(boundary, tau) for tau in tau_grid]
        return vec

    def __initial_boundary(self):
        """setting up starting boundary"""
        self._cheby_H = Chebyshev.Interpolation(self._interpolation_base, np.sqrt(self.T), 0)
        self.tau_grid = [x **2 for x in self._cheby_H.cheby_points]
        self._QD_plus_B_vec = QD_plus.exercise_boundary(self.K, self._B_zero_plus, self.r, self.q, self.sigma, self.tau_grid, self.option_type)
        self.QD_plus_exercise_vec = self._QD_plus_B_vec
        self._Btau_vec = self._QD_plus_B_vec  # start with QD_plus_boundary
        self._cheby_H.fit_by_y_values([self._H(B) for B in self._Btau_vec])  # create H-curve
        self._B = lambda tau: self._H_inverse(self._cheby_H.value(np.sqrt(tau)))  # create B-curve
        self.QD_plus_exercise_curve = self._B
        self._Btau_vec_new = [0] * len(self._Btau_vec)

    def _H(self, B):
        """B -> ln(B/X)^2"""
        return np.log(B / self.K)**2

    def _H_inverse(self, H):
        """H(x) -> X * exp(+-sqrt(H) """
        if self.option_type == 'Put':
            if np.any(H < H_min):
                return self.K * np.exp(-np.sqrt(H_min))
            else:
                return self.K * np.exp(-np.sqrt(H))
        else:  # for call:
            if np.any(H < H_min):
                return self.K * np.exp(np.sqrt(H_min))
            else:
                return self.K * np.exp(np.sqrt(H))


