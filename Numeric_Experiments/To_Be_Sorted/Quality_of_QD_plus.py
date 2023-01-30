import Solver.Option_Solver as OS
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt


def create_random_test_data_within_interval(r_min, r_max, q_min, q_max, sigma_min, sigma_max,
                                             option_type, K=100, T=1, l=81, m=25, n=15, stop_by_diff=None, seed=1):
    """Creating one sample of [QD_plus_boundary_vec, Early_exercise_boundary_vec] for random input"""
    np.random.seed(seed)
    r, q, sigma = np.random.uniform(r_min, r_max), np.random.uniform(q_min, q_max), np.random.uniform(sigma_min, sigma_max)
    option = OS.Option_Solver(r, q, sigma, K, T, option_type, l, m, n, stop_by_diff)
    option.create_boundary()
    b = [option.QD_plus_exercise_vec, option.Early_exercise_vec]
    return [r, q, sigma, b]

def data_creation(file_name, size):
    """Creating txt.file with testdata"""
    file = open(file_name, 'a')
    for i in range(size):
        d = create_random_test_data_within_interval(0.01, 0.1, 0.01, 0.1, 0.1, 0.5, "Put", stop_by_diff=1e-9, seed=i)
        #print(str(d))
        file.write(str(d))
        file.write("\n")
        print(i, " out of ", size)
    file.close()

def load_and_return_trainingsdata(file_path):
    """Loading data from txt. file and return trainingsdata"""
    file = open(file_path)
    lines = file.readlines()
    data = []
    for line in lines:
        try:
            [r, q, sigma, [QD_plus_exercise_vec, Early_exercise_vec]] = eval(line)
            data.append([r, q, sigma, [QD_plus_exercise_vec, Early_exercise_vec]])
        except: break
    return data

def calc_statistics(data):
    sum_avg_diff_per_vector = 0
    sum_max_diff_in_vector = 0
    vec_avg_diff_per_vector = 0
    vec_max_diff_in_vector = 0
    for d in data:
        sum_per_vector = 0
        max_diff_in_vector = 0
        for i in range(len(d[3][0])):
            if np.abs(d[3][0][i] - d[3][1][i]) > max_diff_in_vector:
                max_diff_in_vector = np.abs(d[3][0][i] - d[3][1][i])
            sum_per_vector += np.abs(d[3][0][i] - d[3][1][i])
        avg_diff_per_vector = sum_per_vector / d[3][0]
        sum_avg_diff_per_vector += avg_diff_per_vector
        sum_max_diff_in_vector += max_diff_in_vector
        vec_avg_diff_per_vector.append(avg_diff_per_vector)
        vec_max_diff_in_vector.append(max_diff_in_vector)
    avg_avg_diff_per_vector = sum_avg_diff_per_vector / len(data)
    avg_max_diff_in_vector = sum_max_diff_in_vector / len(data)

def decompose_data(data):
    a_vec= np.empty(shape=(1000, 3))
    b_vec = np.empty(shape=(1000, 25))
    c_vec = np.empty(shape=(1000, 25))
    d_vec = np.empty(shape=(1000, 25))
    for i in range(len(data)):
        [r, q, sigma, [QD_plus_exercise_vec, Early_exercise_vec]] = data[i]
        a_vec[i] = np.array([r, q, sigma])
        b_vec[i] = np.array(Early_exercise_vec)
        c_vec[i] = np.array(QD_plus_exercise_vec)
        d_vec[i] = np.array(QD_plus_exercise_vec) - np.array(Early_exercise_vec)
    return a_vec, b_vec, c_vec, d_vec

def scatter_plot():
    data = load_and_return_trainingsdata("random1000_12_11")
    inp_vec, exact_vec, approx_vec, diff_vec = decompose_data(data)
    abs_diff_vec = np.absolute(diff_vec)
    rel_abs_diff_vec = abs_diff_vec / exact_vec
    mean_rel_abs_diff_vec = np.mean(rel_abs_diff_vec, axis=1)
    max_rel_abs_diff_vec = np.max(rel_abs_diff_vec, axis=1)
    rq_diff = inp_vec[:, 1] - inp_vec[:, 0]
    sigma = inp_vec[:, 2]

    # Mittlere relative Abweichung
    #ax = sns.scatterplot(x=rq_diff, y=mean_rel_abs_diff_vec, hue= sigma,
                         #palette = sns.dark_palette("#69d", reverse=True, as_cmap=True))
    #ax.set(xlabel = "x = q-r")
    #ax.set(ylabel = "y = (1/25) * sum(abs(QD_plus[i] - Exact[i]) / Exact[i])")
    #ax.legend(title="sigma");
    #plt.title('Mittlere relative Abweichung', fontsize=17)
    #plt.show()
    #plt.savefig('scatter_mean', dpi = 333)

    # Maximale relative Abweichung
    bx = sns.scatterplot(x=rq_diff, y=max_rel_abs_diff_vec, hue=sigma,
                         palette=sns.dark_palette("#69d", reverse=True, as_cmap=True))
    bx.set(xlabel="x = q-r")
    bx.set(ylabel="y = max(abs(QD_plus[i] - Exact[i]) / Exact[i])")
    bx.legend(title="sigma");
    plt.title('Maximale relative Abweichung', fontsize=17)
    #plt.show()
    plt.savefig('scatter_max', dpi = 333)


def box_plots():
    data = load_and_return_trainingsdata("random1000_12_11")
    inp_vec, exact_vec, approx_vec, diff_vec = decompose_data(data)
    abs_diff_vec = np.absolute(diff_vec)
    rel_abs_diff_vec = abs_diff_vec / exact_vec
    mean_rel_abs_diff_vec = np.mean(rel_abs_diff_vec, axis=1)
    max_rel_abs_diff_vec = np.max(rel_abs_diff_vec, axis=1)

    # Mittlere relative Abweichung
    """
    ax = sns.boxplot(x=mean_rel_abs_diff_vec)
    ax.set(xlabel="Z = (1/25) * sum(abs(QD_plus[i] - Exact[i]) / Exact[i])")
    plt.title('Mittlere relative Abweichung', fontsize=20)
    #plt.show()
    print("mean =", mean_rel_abs_diff_vec.mean())
    print("median =", np.median(mean_rel_abs_diff_vec))
    print("std = ", np.std(mean_rel_abs_diff_vec))
    print("max = ", np.max(mean_rel_abs_diff_vec), " at ", inp_vec[np.argmax(mean_rel_abs_diff_vec)])
    print("min = ", np.min(mean_rel_abs_diff_vec), " at ", inp_vec[np.argmin(mean_rel_abs_diff_vec)])
    plt.savefig('middle_relative.png', dpi = 333)"""

    print()
    # Maximale relative Abweichung
    ax = sns.boxplot(x=max_rel_abs_diff_vec)
    ax.set(xlabel="Z = max(abs(QD_plus[i] - Exact[i]) / Exact[i])")
    plt.title('Maximale relative Abweichung', fontsize=20)
    #plt.show()
    print("mean = ", max_rel_abs_diff_vec.mean())
    print("median =", np.median(max_rel_abs_diff_vec))
    print("std = ", np.std(max_rel_abs_diff_vec))
    print("max = ", np.max(max_rel_abs_diff_vec), " at ", inp_vec[np.argmax(mean_rel_abs_diff_vec)])
    print("min = ", np.min(max_rel_abs_diff_vec), " at ", inp_vec[np.argmin(mean_rel_abs_diff_vec)])
    plt.savefig('max_relative', dpi = 333)

def main():
    #data_creation("random1000_12_11", 1000)
    box_plots()
    #scatter_plot()


if __name__=="__main__":
    main()




