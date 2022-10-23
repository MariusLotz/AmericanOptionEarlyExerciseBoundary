import numpy as np
import scipy.stats as stats
import scipy.integrate as si
import Chebyshev
import QD_plus_exercise_boundary as QD_plus
import BS_formulas as B

class Option_Solver():
    def __init__(self, interest_rate, dividend, volatility, strike, maturity, l=113, m=21, n=85):
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
        self.fail = False
        self.negativ_sqrt = True

    def __initial(self):
        """setting up stuff"""
        #self.cheby_H = Chebyshev.Interpolation(self.interpolation_base, np.sqrt(self.T), 0)
        self.cheby_B = Chebyshev.Interpolation(self.interpolation_base, self.T, 0)
        #self.tau_grid = [x ** 2 for x in self.cheby_H.cheby_points]
        self.tau_grid = self.cheby_B.cheby_points
        self.QD_plus_B_vec = QD_plus.exercise_boundary(self.K, self.r, self.q, self.sigma, self.tau_grid)
        self.Btau_vec = self.QD_plus_B_vec
        #self.cheby_H.fit_by_y_values([self.__H(B) for B in self.Btau_vec])
        self.cheby_B.fit_by_y_values(self.Btau_vec)
        self.B = self.cheby_B.value
        #self.B = lambda tau : self.__H_inverse(self.cheby_H.value(np.sqrt(tau)))
        self.Btau_vec_new = [0] * len(self.Btau_vec)
        self.B_new = None

    def create_boundary_after_fail(self):
        """calculate the boundary function of an American Option"""
        self.__initial()
        for j in range(self.iteration_steps):
            # Fixpoint Iteration:
            for i in range(self.interpolation_base - 1):
                print(self.Btau_vec_new[i])
                self.Btau_vec_new[i] = self.B_plus_adaptive(self.tau_grid[i])  # Iteration per tau
            self.Btau_vec_new[-1] = self.B_at_zero
            # create cheby instance on B:
            cheby_B = Chebyshev.Interpolation(self.interpolation_base, 0, self.T)
            cheby_B.fit_by_y_values(self.Btau_vec_new)
            self.B_new = cheby_B.value
            self.B = self.B_new
            self.Btau_vec = self.Btau_vec_new

    def create_boundary(self):
        """calculate the boundary function of an American Option"""
        for j in range(self.iteration_steps):
            # Fixpoint Iteration:
            for i in range(self.interpolation_base - 1):
                self.Btau_vec_new[i] = self.B_plus_adaptive(self.tau_grid[i]) # Iteration per tau
            self.Btau_vec_new[-1] = self.B_at_zero
            self.B_diff(self.B, self.Btau_vec_new)
            self.B_new = self.cheby_B.fit_by_y_values(self.Btau_vec_new)
            #self.cheby_H.fit_by_y_values([self.__H(B) for B in self.Btau_vec_new]) # create H-curve
            #self.B_new = lambda tau : self.__H_inverse(self.cheby_H.value(np.sqrt(tau))) # create B
            self.B = self.B_new
            if self.Btau_vec[0] == 0 or np.isnan(self.Btau_vec[0]) or np.isinf(self.Btau_vec[0]):
                self.fail = True
                self.create_boundary_after_fail()
                break
            self.Btau_vec = self.Btau_vec_new

    def B_plus(self, tau, eta=0.8):
        """fixpoint scheme for one tau, getting Btau_new"""
        k = self.K * np.exp(-(self.r - self.q) * tau)
        return (self.B(tau) - eta * (self.B(tau) - k * self.N(tau) / self.D(tau)))

    def B_plus_adaptive(self, tau, allowed_perc = 0.05):
        """fixpoint scheme for one tau, getting Btau_new, adaptive"""
        k = self.K * np.exp(-(self.r - self.q) * tau)
        eta = 1
        ratio = allowed_perc + 1.1
        while ratio > 1 + allowed_perc  or ratio < 1 - allowed_perc:
            #print("eta:", eta)
            eta *= (2 / 3)
            #print(self.N(tau), self.D(tau))
            #print(k * self.N(tau) / self.D(tau))
            Btau_new = (self.B(tau) - eta * (self.B(tau) - k * self.N(tau) / self.D(tau)))
            ratio  = Btau_new / self.B(tau)
            #print("ratio:", ratio)
            print("Btau_new:", Btau_new)
        return Btau_new

    def B_diff(self, Btau, Btau_new):
        sum = 0
        for i in range(len(self.tau_grid)-1):
            #dt = abs(self.tau_grid[i+1] - self.tau_grid[i])
            DB = abs(Btau_new[i] - Btau(self.tau_grid[i]))
            sum += DB
        print(sum)

    def N(self, tau):
        """N term of fixpoint scheme"""
        def integrand(u):
            return np.exp(self.r * u) * stats.norm.cdf(self.__d_minus(tau - u, self.B(tau) / self.B(u)))
        a = stats.norm.cdf(self.__d_minus(tau, self.B(tau) / self.K))
        b = si.fixed_quad(integrand, 0, tau, n=self.integration_base)[0]
        if np.isnan(b) or b==0:
            return a
        else:
            return a + self.r * b

    def D(self, tau):
        """D term of fixpoint scheme"""
        def integrand(u):
            return np.exp(self.q * u) * stats.norm.cdf(self.__d_plus(tau - u, self.B(tau) / self.B(u)))
        a = stats.norm.cdf(self.__d_plus(tau, self.B(tau) / self.K))
        b = si.fixed_quad(integrand, 0, tau, n=self.integration_base)[0]
        if np.isnan(b) or b == 0:
            return a
        else:
            return a + self.q * b

    def __d_minus(self, tau, X):
        """d_- from Black Scholes formula"""
        return (np.log(X) + (self.r - self.q) * tau - 0.5 * self.sigma**2 * tau) / (self.sigma * np.sqrt(tau))

    def __d_plus(self, tau, X):
        """d_+ from Black Scholes formula"""
        return (np.log(X) + (self.r - self.q) * tau + 0.5 * self.sigma**2 * tau) / (self.sigma * np.sqrt(tau))

    def __H(self, B):
        """B -> ln(B/X)^2"""
        a = np.log(B / self.K)**2
        if abs(a) < -1E-12 or np.isnan(0):
            self.negativ_sqrt = True
            return 1E-11
        else:
            if a < 0:
                self.negativ_sqrt = True
                return a
            else:
                self.negativ_sqrt = False
                return a

    def __H_inverse(self, H):
        """H(x) -> X * exp(+-sqrt(H) """
        if self.negativ_sqrt:
            return self.B_at_zero * np.exp(-np.sqrt(H))
        else:
            return self.B_at_zero * np.exp(np.sqrt(H))

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

def main():
    option = Option_Solver(0.005, 0.006, 0.35, 100, 1)
    print(option.QD_plus_B_vec)
    option.create_boundary()
    print(option.Btau_vec)

if __name__=="__main__":
    main()


