import numpy as np


def transform_interval_to_cheby_standard(x, x_min, x_max):
    return (2 * (x - x_min) / (x_max - x_min)) - 1

def transform_cheby_standard_to_intervall(cheby_points, x_min, x_max):
    return x_min + (x_max - x_min) * (cheby_points + 1) * 0.5


class Interpolation:
    """Chebyshev Interpolation Instance"""
    def __init__(self, base_count, x_min, x_max):
        self.base_count = None
        self.n = None
        self.x_min = None
        self.x_max = None
        self.coeff = None
        self.y_vec = None
        self.cheby_points = None
        self.fit_cheby_points(base_count, x_min, x_max)

    def fit_cheby_points(self, base_count, x_min, x_max):
        """ Setting cheby_points"""
        self.base_count = base_count
        self.n = base_count - 1
        self.x_min = x_min
        self.x_max = x_max
        self.cheby_points = []
        if not ((self.x_min == - 1) and (self.x_max == 1)):
            for k in range(self.base_count):
                transformed_cheby_point = transform_cheby_standard_to_intervall(
                    -np.cos(k * np.pi / self.n), self.x_min, self.x_max)
                self.cheby_points.append(transformed_cheby_point)
        else:
            for k in range(self.base_count):
                cheby_point = -np.cos(k * np.pi / self.n)
                self.cheby_points.append(cheby_point)

    def calc_coeff(self):
        """Calculating standard cheby coeff for given y_vec"""
        self.coeff = []
        for k in range(self.base_count):
            res = 0
            for i in range(0, self.base_count):
                partial = self.y_vec[self.n-i] * np.cos(i * k * np.pi / self.n)
                if i == 0 or i == self.n:
                    partial *= 0.5
                res += partial
            res *= (2.0 / self.n)
            self.coeff.append(res)

    def fit_by_y_values(self, y_vec):
        """Setting coefficients for Chebyshev-Polynomial via discrete values """
        if self.n == None:
            print("Use fit_cheby_points to set Interpolation points first!")
        else:
            self.y_vec = y_vec
            self.calc_coeff()

    def fit_by_function(self, f):
        """Setting coefficients for Chebyshev-Polynomial via cheby_point matching condition"""
        y_vec = [f(x) for x in self.cheby_points]
        self.fit_by_y_values(y_vec)

    def value(self, x):
        """returning cheby_y value for a given x value"""
        if self.y_vec == None:
            print("Use fit_by_y_values to calculate coefficients first!")
        else:
            if not ((self.x_min == - 1) and (self.x_max == 1)):
                x = transform_interval_to_cheby_standard(x, self.x_min, self.x_max)
            cheby_y = self.clenshaw_algo(x)
            return cheby_y

    def values(self, x_vec):
        """returning cheby_y values for a given x values"""
        if self.y_vec == None:
            print("Use fit_by_y_values to calculate coefficients first!")
        else:
            try:
                cheby_y_vec = []
                if self.x_max != 1 or self.x_min != -1:
                    for x in x_vec:
                        x = transform_interval_to_cheby_standard(x, self.x_min, self.x_max)
                        cheby_y_vec.append(self.clenshaw_algo(x))
                else:
                    for x in x_vec:
                        cheby_y_vec.append(self.clenshaw_algo(x))
                return cheby_y_vec
            except:
                return self.value(cheby_y_vec)

    def clenshaw_algo(self, x):
        """ returning y-value for given x-value, calculation via clenshaw"""
        d0 = self.coeff[self.n] / 2
        d1, d2  = 0, 0
        for k in range(self.n - 1, -1, -1):
            d1, d2 = d0, d1
            d0 = self.coeff[k] + 2 * x * d1 - d2
        return (d0 - d2) / 2

