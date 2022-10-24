import numpy as np
import scipy.stats as stats

import Chebyshev
import QD_plus_exercise_boundary as QD_plus
import BS_formulas as B
from FixpointsystemB import FixpointsystemB

class Option_Solver(FixpointsystemB):
    def __init__(self, interest_rate, dividend, volatility, strike, maturity, option_type = 'Put', l=113, m=21, n=15):
        self.r = interest_rate
        self.q = dividend
        self.sigma = volatility
        self.K = strike
        self.T = maturity
        self.interpolation_base = m # number of basis points for the Chebyshev polynomial
        self.integration_base = l # number of basis points for the Gauss quadrature
        self.iteration_steps = n # number of fixpoint iteration steps
        self.Btau_vec = None
        self.B = None
        self.Btau_vec_new = None
        self.B_new = None
        if option_type == "Put":
            self.put_option = True
            self.r_intern = self.r
            self.q_intern = self.q
        elif option_type == "Call":
            self.put_option = False
            # using the fact that B_C(tau, r, q) = K^2 / B_P(tau, r, q)
            self.r_intern = self.q
            self.q_intern = self.r
        else:
            raise ValueError("option_type has to be 'Put' or 'Call'")
        self.B_at_zero = self.K * min(1, self.r_intern / self.q_intern)
        self.__initial_boundary()

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
            boundary = None ### NOTF
        else:
            boundary = self.B
        return si.fixed_quad(integrand, 0, tau, n=self.integration_base)[0]

    def european_put_price(self, S, tau):
        return B.put_price(S, self.K, self.r_intern, self.q_intern, self.sigma, tau)

    def european_call_price(self, S, tau):
        return B.call_price(S, self.K, self.r_intern, self.q_intern, self.sigma, tau)

    def american_put_price(self, S, tau):
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
                self.Btau_vec[i] = self.__B_plus(self.tau_grid[i])  # Iteration per tau
            self.Btau_vec_new[-1] = self.B_at_zero  # B value close to maturity
            self.cheby_H.fit_by_y_values([self.__H(B) for B in self.Btau_vec_new]) # create H-curve
            self.B_new = lambda tau : self.__H_inverse(self.cheby_H.value(np.sqrt(tau))) # create B
            self.B = self.B_new
            self.Btau_vec = self.Btau_vec_new

    def __initial_boundary(self):
        """setting up starting boundary"""
        self.cheby_H = Chebyshev.Interpolation(self.interpolation_base, np.sqrt(self.T), 0)
        self.tau_grid = [x ** 2 for x in self.cheby_H.cheby_points]
        self.QD_plus_B_vec = QD_plus.exercise_boundary(self.K, self.r_intern, self.q_intern, self.sigma, self.tau_grid)

    def __B_plus(self, tau, eta=0.5):
        """fixpoint scheme for one tau, getting Btau_new"""
        #b = 1e-03 # for f_prime approximation
        #B_plus = lambda tau, B, b : B(tau) + b
        k = self.K * np.exp(-(self.r_intern - self.q_intern) * tau)
        f = lambda tau, B : k * self.N(tau) / self.D(tau)
        #f_prime = lambda tau, B, b: (f(tau, B_plus(tau, B, b)) - f(tau, B)) / b

        #print(f_prime(tau, self.B, b))
        #return self.B(tau) + eta * (self.B(tau) - f(tau, self.B(tau))) / (f_prime(tau, self.B, b) - 1)
        return self.B(tau) + eta * (self.B(tau) - f(tau, self.B(tau))) / ( - 1)


    def __H(self, B):
        """B -> ln(B/X)^2"""
        return np.log(B / self.B_at_zero)**2

    def __H_inverse(self, H):
        """H(x) -> X * exp(+-sqrt(H) """
        return self.B_at_zero * np.exp(-np.sqrt(H))





def main():
    r, q, sigma, K, T, = 0.005, 0.007, 0.35, 100, 1.0
    option = Option_Solver(r, q, sigma, K, T, m=22)
    option.create_boundary()
    print((option.QD_plus_B_vec))
    #print(option.tau_grid)
    print(option.Btau_vec)




if __name__ =="__main__":
    main()

