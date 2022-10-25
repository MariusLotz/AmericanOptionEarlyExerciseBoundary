import numpy as np
import scipy.stats as stats
import scipy.integrate as si
import Chebyshev
import QD_plus_exercise_boundary as QD_plus
import BS_formulas as B
from FixpointsystemB import FixpointsystemB

H_min = 1e-25

class Option_Solver(FixpointsystemB):
    def __init__(self, interest_rate, dividend, volatility, strike, maturity, option_type = 'Put', l=113, m=21, n=15):
        self.r = interest_rate
        self.q = dividend
        self.sigma = volatility
        self.K = strike
        self.T = maturity
        self.interpolation_base = m  # number of basis points for the Chebyshev polynomial
        self.integration_base = l  # number of basis points for the Gauss quadrature
        self.iteration_steps = n  # number of fixpoint iteration steps
        self.Early_exercise_curve = None
        self.Early_exercise_vec = None
        self.Btau_vec = None
        self.B = None
        self.Btau_vec_new = None
        self.B_new = None
        self.B_at_zero = self.K
        self.option_type = option_type
        ### Put-Call-Symmetry Trick ###
        ### using the fact that B_C(tau, r, q) = K^2 / B_P(tau, r, q) ###
        if self.r >= self.q and self.option_type == 'Put':  # Put/Put
            self.internal_option_type = 'Put'
            self.r_internal = self.r
            self.q_internal = self.q
        elif self.r < self.q and self.option_type == 'Put':  # Put/Call
            self.internal_option_type = 'Call'
            self.r_internal = self.q
            self.q_internal = self.r
        elif self.r >= self.q and self.option_type == 'Call':  # Call/Put
            self.internal_option_type = 'Put'
            self.r_internal = self.q
            self.q_internal = self.r
        elif self.r < self.q and self.option_type == 'Call':  # Call/Call
            self.internal_option_type = 'Call'
            self.r_internal = self.r
            self.q_internal = self.q
        else:
            raise ValueError("option_type has to be 'Put' or 'Call'")
        self.__initial_boundary()
        if self.r < 1e-06 or self.sigma < 1e-06:
            print("""WARNING:
            Extrem small values for the interest rate (r) or volatility (sigma) are problematic! 
            """)

    def premium(self, S, tau, boundary = None):
        """American premium for given boundary=exact_boundary.
        boundary_input must be ???..."""
        def integrand(u):
            z = S / boundary(u)
            a = self.r_intern * self.K * np.exp(-self.r_intern * (tau - u)) * stats.norm.cdf(-self.__d_minus(tau - u, z))
            b = self.q_intern * S * np.exp(-self.q_intern * (tau - u)) * stats.norm.cdf(-self.__d_plus(tau - u, z))
            return a - b
        if tau > self.T: raise ValueError("tau can not be larger than the maturity!")
        if tau < 0: raise ValueError("tau can not be negativ!")
        if S < 0: raise ValueError("stock price can not be negativ!")
        if boundary == None and self.B == None:
            print("""QD_plus boundary was used. Therefore outcome will be an approximate premium. 
            Specify boundary you like to use, or run '.create_boundary()' first to use exact boundary!""")
            self.cheby_H.fit_by_y_values([self.__H(B) for B in self.QD_plus_B_vec]) # create H-curve
            boundary = lambda tau: self.__H_inverse(self.cheby_H.value(np.sqrt(tau))) # define B-curve
        if type(boundary) == list:
            boundary = None ##NOTF
        else:
            boundary = self.B
        return si.fixed_quad(integrand, 0, tau, n=self.integration_base)[0]

    def european_put_price(self, S, tau):
        return B.put_price(S, self.K, self.r, self.q, self.sigma, tau)

    def european_call_price(self, S, tau):
        return B.call_price(S, self.K, self.r, self.q, self.sigma, tau)

    def american_put_price(self, S, tau):
        if self.option_type != 'Put':
            raise TypeError("")
            return self.premium(S, tau) + self.european_put_price(self, S, tau)

    def american_call_price(self, S, tau):
        return self.premium(S, tau) + self.european_call_price(self, S, tau)

    def create_boundary(self):
        """calculate the boundary function of an American Option"""
        self.Btau_vec = self.QD_plus_B_vec
        self.cheby_H.fit_by_y_values([self.__H(B) for B in self.Btau_vec])
        self.B = lambda tau: self.__H_inverse(self.cheby_H.value(np.sqrt(tau)))
        self.Btau_vec_new = [0] * len(self.Btau_vec)
        for j in range(self.iteration_steps):
            # Fixpoint Iteration for each tau:
            for i in range(self.interpolation_base - 1):
                self.Btau_vec_new[i] = self.B_plus(self.tau_grid[i])  # Iteration per tau
            self.Btau_vec_new[-1] = self.B_at_zero  # B value close to maturity
            self.cheby_H.fit_by_y_values([self.__H(B) for B in self.Btau_vec_new]) # create H-curve
            self.B_new = lambda tau : self.__H_inverse(self.cheby_H.value(np.sqrt(tau))) # create B
            self.B = self.B_new
            self.Btau_vec = self.Btau_vec_new
        # Reverse Put-Call-Symmetry Trick:
        if self.option_type == 'Put' and self.internal_option_type == 'Call':
            self.Early_exercise_vec = [self.B_at_zero**2 / C for C in self.Btau_vec]
        elif self.option_type == 'Call' and self.internal_option_type == 'Put':
            self.Early_exercise_vec = [self.B_at_zero** 2 / P for P in self.Btau_vec]
        else:
            self.Early_exercise_vec = self.Btau_vec
        self.cheby_H.fit_by_y_values([self.__H(B) for B in self.Early_exercise_vec])  # create H-curve
        self.Early_exercise_curve = lambda tau: self.__H_inverse(self.cheby_H.value(np.sqrt(tau)))  # create B


    def __initial_boundary(self):
        """setting up starting boundary"""
        self.cheby_H = Chebyshev.Interpolation(self.interpolation_base, np.sqrt(self.T), 0)
        self.tau_grid = [x ** 2 for x in self.cheby_H.cheby_points]
        self.QD_plus_B_vec = QD_plus.exercise_boundary(self.K, self.r_internal, self.q_internal, self.sigma, self.tau_grid, self.internal_option_type)


    def __H(self, B):
        """B -> ln(B/X)^2"""
        return np.log(B / self.B_at_zero)**2

    def __H_inverse(self, H):
        """H(x) -> X * exp(+-sqrt(H) """
        if self.internal_option_type == 'Put':
            if np.any(H < H_min):
                return self.B_at_zero * np.exp(-np.sqrt(H_min))
            else:
                return self.B_at_zero * np.exp(-np.sqrt(H_min))
        else:
            if np.any(H < H_min):
                return self.B_at_zero * np.exp(np.sqrt(H_min))
            else:
                return self.B_at_zero * np.exp(np.sqrt(H_min))

if __name__ =="__main__":
    main()

