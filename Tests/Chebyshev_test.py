import Chebyshev
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.exp(x)

def use_case():
    ERR = 0.0001
    # create instance:
    cheby = Chebyshev.Interpolation()

    # define approximation interval and base count:
    cheby.fit_cheby_points(3, -5, 5)
    assert abs(cheby.cheby_points[0] - (-5)) < ERR
    assert abs(cheby.cheby_points[1] - (0)) < ERR
    assert abs(cheby.cheby_points[2] - (5)) < ERR

    # set cheby coeff regarding the approximation values y_0, y_1, y_2:
    cheby.fit_by_y_values([1, 3, 5])

    # return a single cheby-approximation value from x
    cheby_y = cheby.value(cheby.cheby_points[1])
    assert  abs(cheby_y - 3) < ERR

    # return a whole vector of cheby-approximation values from x_vector
    values = cheby.values(cheby.cheby_points)
    assert abs(values[1] - 3) < ERR

def test_for_convergence():
    max_diff_vec=[]
    sum_diff_vec=[]
    base_counts=[k for k in range(2, 30)]
    for k in base_counts:
        cheby = Chebyshev.Interpolation(k, -5, 5)
        cheby.fit_by_function(f)
        #cheby.fit_by_y_values([f(x) for x in cheby.cheby_points])
        check_points = [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        cheby_values = cheby.values(check_points)
        max_diff = 0
        sum_diff = 0
        for j in range(10):
            diff = abs(cheby_values[j] - f(check_points[j]))
            sum_diff += diff
            if diff > max_diff: max_diff = diff
        max_diff_vec.append(max_diff)
        sum_diff_vec.append(sum_diff)

    plt.plot(base_counts, sum_diff_vec, 'blue', label="sum_diff")
    plt.plot(base_counts, max_diff_vec, 'red', label="max_diff")
    plt.yscale('log', base=10)
    plt.xscale('log', base=2)
    plt.legend(loc='upper right')

def exp_test():
    H_cheby = Chebyshev.Interpolation(5,0,np.sqrt(10))
    B_cheby = Chebyshev.Interpolation(5,0,10)
    print([x**2 for x in H_cheby.cheby_points])
    print(B_cheby.cheby_points)
    #print([np.sqrt(x) for x in B_cheby.cheby_points])

if __name__=="__main__":
    #test_for_convergence()
    #use_case()
    exp_test()
