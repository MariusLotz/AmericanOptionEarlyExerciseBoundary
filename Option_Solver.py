import numpy as np
import scipy.stats as stats
import scipy.integrate as si
import Chebyshev
import QD_plus_exercise_boundary as QD_plus
import BS_formulas as B
from FixpointsystemB import FixpointsystemB

H_min = 1e-25

class Option_Solver(FixpointsystemB):
    def __init__(self, interest_rate, dividend, volatility, strike, maturity,
                 option_type = 'Put', l=113, m=21, n=15, print_err_each_step = False):
        self.r = interest_rate
        self.q = dividend
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
        self._B_zero_plus = self.K
        self.option_type = option_type
        self.print_err_each_step = print_err_each_step
        ### Put-Call-Symmetry Trick # using the fact that B_C(tau, r, q) = K^2 / B_P(tau, r, q) ###
        if self.r >= self.q and self.option_type == 'Put':  # Put/Put
            self._internal_option_type = 'Put'
            self._r_internal = self.r
            self._q_internal = self.q
        elif self.r < self.q and self.option_type == 'Put':  # Put/Call
            self._internal_option_type = 'Call'
            self._r_internal = self.q
            self._q_internal = self.r
        elif self.r >= self.q and self.option_type == 'Call':  # Call/Put
            self._internal_option_type = 'Put'
            self._r_internal = self.q
            self._q_internal = self.r
        elif self.r < self.q and self.option_type == 'Call':  # Call/Call
            self._internal_option_type = 'Call'
            self._r_internal = self.r
            self._q_internal = self.q
        else:
            raise ValueError("option_type has to be 'Put' or 'Call'")
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
                else:
                    a = self.q * S * np.exp(-self.q * (tau - u)) * stats.norm.cdf(self._d_plus(tau - u, z))
                    b = self.r * self.K * np.exp(-self.r * (tau - u)) * stats.norm.cdf(self._d_minus(tau - u, z))
                return a - b
        if type(boundary) == list:
            print("""
            make sure the boundary values are in order in regard to the '.tau_grid'
            """)
            _cheby_H = Chebyshev.Interpolation(self._interpolation_base, np.sqrt(self.T), 0)
            _cheby_H.fit_by_y_values([self.__H(B) for B in boundary])  # create H-curve
            boundary = lambda tau: self.__H_inverse(self._cheby_H.value(np.sqrt(tau)))  # define B-curve
        elif self.Early_exercise_curve == None:
            print("""
            QD_plus boundary was used. Therefore outcome will be an approximation only.
            Specify boundary you like to use, or run '.create_boundary()' first to use exact boundary!
            """)
            _cheby_H = Chebyshev.Interpolation(self._interpolation_base, np.sqrt(self.T), 0)
            _cheby_H.fit_by_y_values([self.__H(B) for B in self.QD_plus_exercise_vec])  # create H-curve
            boundary = lambda tau: self.__H_inverse(_cheby_H.value(np.sqrt(tau)))  # define B-curve

        else:
            boundary = self.Early_exercise_curve
        return si.fixed_quad(integrand, 0, tau, n=self._integration_base)[0]

    def European_price(self, S, tau):
        """European optionprice via Black Scholes formula"""
        if tau < 0: raise ValueError("tau can not be negativ!")
        if S < 0: raise ValueError("stock price can not be negativ!")
        if self.option_type == 'Put':
            return B.put_price(S, self.K, self.r, self.q, self.sigma, tau)
        else:
            return B.call_price(S, self.K, self.r, self.q, self.sigma, tau)

    def American_price(self, S, tau):
        """American option price via American premium"""
        return self.premium(S, tau) + self.European_price(S, tau)

    def create_boundary(self):
        """calculate the boundary function of an American Option"""
        self._Btau_vec = self._QD_plus_B_vec  # start with QD_plus_boundary
        self.__cheby_H.fit_by_y_values([self.__H(B) for B in self._Btau_vec])  # create H-curve
        self._B = lambda tau: self.__H_inverse(self.__cheby_H.value(np.sqrt(tau))) # create B-curve
        self._Btau_vec_new = [0] * len(self._Btau_vec)
        for j in range(self._iteration_steps):
            # Fixpoint Iteration for each tau:
            for i in range(self._interpolation_base - 1):
                self._Btau_vec_new[i] = self._B_plus(self.tau_grid[i])  # Iteration per tau
            self._Btau_vec_new[-1] = self._B_zero_plus  # B value close to maturity
            self.__cheby_H.fit_by_y_values([self.__H(B) for B in self._Btau_vec_new])  # create H-curve
            self._B_new = lambda tau : self.__H_inverse(self.__cheby_H.value(np.sqrt(tau)))  # create B-curve
            self._B = self._B_new
            self._Btau_vec = self._Btau_vec_new
            if self.print_err_each_step:
                print(self.errsum_to_fixpoint(self._B))
        # Reverse Put-Call-Symmetry Trick:
        if self.option_type != self._internal_option_type:
            self.Early_exercise_vec = [self._B_zero_plus** 2 / x for x in self._Btau_vec]
            self.Early_exercise_curve = lambda tau: self._B_zero_plus ** 2 / self._B(tau)
        else:
            self.Early_exercise_vec = self._Btau_vec
            self.Early_exercise_curve = lambda tau: self._B(tau)

    def __err_to_fixpoint(self, B, tau):
        """returns the abs. difference between B(tau) and F(B(tau))=B_plus"""
        return abs(self._B_plus(tau, eta=1) - B(tau))  #NOTF

    def __errsum_to_fixpoint(self, B):
        """returns the sum of abs. difference between B(tau) and F(B(tau))=B_plus"""
        sum = 0 #NOTF
        for tau in self.tau_grid:
            sum += self.err_to_fixpoint(B, tau)
        return sum

    def __initial_boundary(self):
        """setting up starting boundary"""
        self.__cheby_H = Chebyshev.Interpolation(self._interpolation_base, np.sqrt(self.T), 0)
        self.tau_grid = [x ** 2 for x in self.__cheby_H.cheby_points]
        self._QD_plus_B_vec = QD_plus.exercise_boundary(self.K, self._r_internal, self._q_internal, self.sigma, self.tau_grid, self._internal_option_type)
        if self._internal_option_type != self.option_type:
            self.QD_plus_exercise_vec = [self._B_zero_plus ** 2 / x for x in self._QD_plus_B_vec]
        else:
            self.QD_plus_exercise_vec = self._QD_plus_B_vec

    def __H(self, B):
        """B -> ln(B/X)^2"""
        return np.log(B / self._B_zero_plus) ** 2

    def __H_inverse(self, H):
        """H(x) -> X * exp(+-sqrt(H) """
        if self._internal_option_type == 'Put':
            if np.any(H < H_min):
                return self._B_zero_plus * np.exp(-np.sqrt(H_min))
            else:
                return self._B_zero_plus * np.exp(-np.sqrt(H_min))
        else:  # for call:
            if np.any(H < H_min):
                return self._B_zero_plus * np.exp(np.sqrt(H_min))
            else:
                return self._B_zero_plus * np.exp(np.sqrt(H_min))


