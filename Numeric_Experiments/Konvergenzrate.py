import Option_Solver as OS
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def create_random_test_data_within_interval(r_min, r_max, q_min, q_max, sigma_min, sigma_max,
                                             option_type, K=100, T=1, l=81, m=25, n=15, stop_by_diff=None, seed=1):
    """Creating one sample of [QD_plus_boundary_vec, Early_exercise_boundary_vec] for random input"""
    np.random.seed(seed)
    r, q, sigma = np.random.uniform(r_min, r_max), np.random.uniform(q_min, q_max), np.random.uniform(sigma_min, sigma_max)
    option = OS.Option_Solver(r, q, sigma, K, T, option_type, l, m, n, stop_by_diff)
    option.create_boundary()
    b = option.used_iteration_steps
    return [r, q, sigma, b]

def data_creation():
    """Creating txt.file with testdata"""
    file_name= "random1000_iter_under_10m6"
    size = 1000
    file = open(file_name, 'a')
    for i in range(size):
        d = create_random_test_data_within_interval(0.01, 0.1, 0.01, 0.1, 0.1, 0.5, "Put", stop_by_diff=1e-6, seed=i)
        #print(str(d))
        file.write(str(d))
        file.write("\n")
        print(i, " out of ", size)
    file.close()

def create_data():
    option_type = "Put"
    K = 100
    T = 1
    file = open("6paths_for_speed", "a")
    r, q, sigma = 0.05, 0.05, 0.1
    option1 = OS.Option_Solver(r, q, sigma, K, T, option_type, l=55, m=25, n=50, stop_by_diff=None)
    option1.create_boundary()
    data = [r, q, sigma, option1.max_diff_vec]
    file.write(str(data))
    file.write("\n")

    r, q, sigma = 0.01, 0.01, 0.5
    option2 = OS.Option_Solver(r, q, sigma, K, T, option_type, l=55, m=25, n=50, stop_by_diff=None)
    option2.create_boundary()
    data = [r, q, sigma, option2.max_diff_vec]
    file.write(str(data))
    file.write("\n")

    r, q, sigma = 0.01, 0.1, 0.5
    option3 = OS.Option_Solver(r, q, sigma, K, T, option_type, l=55, m=25, n=50, stop_by_diff=None)
    option3.create_boundary()
    data = [r, q, sigma, option3.max_diff_vec]
    file.write(str(data))
    file.write("\n")

    r, q, sigma = 0.1, 0.01, 0.5
    option4 = OS.Option_Solver(r, q, sigma, K, T, option_type, l=55, m=25, n=50, stop_by_diff=None)
    option4.create_boundary()
    data = [r, q, sigma, option4.max_diff_vec]
    file.write(str(data))
    file.write("\n")

    r, q, sigma = 0.01, 0.1, 0.1
    option5 = OS.Option_Solver(r, q, sigma, K, T, option_type, l=55, m=25, n=50, stop_by_diff=None)
    option5.create_boundary()
    data = [r, q, sigma, option5.max_diff_vec]
    file.write(str(data))
    file.write("\n")

    r, q, sigma = 0.1, 0.01, 0.1
    option6 = OS.Option_Solver(r, q, sigma, K, T, option_type, l=55, m=25, n=50, stop_by_diff=None)
    option6.create_boundary()
    data = [r, q, sigma, option6.max_diff_vec]
    file.write(str(data))
    file.write("\n")

    file.close()

def load_and_return_trainingsdata(file_path):
    """Loading data from txt. file and return trainingsdata"""
    file = open(file_path)
    lines = file.readlines()
    data = []
    for line in lines:
        try:
            [r, q, sigma, iter_error] = eval(line)
            data.append([r, q, sigma, iter_error])
        except: break
    return data


def plot():
    data = np.array(load_and_return_trainingsdata("6paths_for_speed"))
    # print(data)
    y0 = data[0, 3]
    y1 = data[1, 3]
    y2 = data[2, 3]
    y3 = data[3, 3]
    y4 = data[4, 3]
    y5 = data[5, 3]
    x = [i for i in range(1, 51)]
    ax = sns.lineplot(x=x, y=y0, color="purple")
    ax = sns.lineplot(x=x, y=y1, color="darkgreen")
    ax = sns.lineplot(x=x, y=y2, color="r")
    ax = sns.lineplot(x=x, y=y3, color="b")
    ax = sns.lineplot(x=x, y=y4, color="lime")
    ax = sns.lineplot(x=x, y=y5, color="orange")
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set(xlabel = "number of fixpoint iterations")
    ax.set(ylabel= "max(Boundary[i] - fixpoint_iter(Boundary[i]))")
    plt.title('Konvergenzgeschwindigkeit', fontsize=17)
    ax.text(31, 0.5, "r, q, sigma = " + str(data[0, 0])+ ", " + str(data[0,1]) + ", " + str(data[0,2]),
            color="purple",fontsize=9)
    ax.text(31, 0.125, "r, q, sigma = " + str(data[1, 0]) + ", " + str(data[1, 1]) + ", " + str(data[1, 2]),
            color="darkgreen", fontsize=9)
    ax.text(31, 0.03125, "r, q, sigma = " + str(data[2, 0]) + ", " + str(data[2, 1]) + ", " + str(data[2, 2]),
            color="r", fontsize=9)
    ax.text(31, 0.007812, "r, q, sigma = " + str(data[3, 0]) + ", " + str(data[3, 1]) + ", " + str(data[3, 2]),
            color="b", fontsize=9)
    ax.text(31, 0.00195, "r, q, sigma = " + str(data[4, 0]) + ", " + str(data[4, 1]) + ", " + str(data[4, 2]),
            color="lime", fontsize=9)
    ax.text(31, 0.00048, "r, q, sigma = " + str(data[5, 0]) + ", " + str(data[5, 1]) + ", " + str(data[5, 2]),
            color="orange", fontsize=9)
    #plt.savefig('6path_conv_small', dpi = 333)

def plotsmall():
    data = np.array(load_and_return_trainingsdata("6paths_for_speed"))
    # print(data)
    y0 = data[0, 3][0:15]
    y1 = data[1, 3][0:15]
    y2 = data[2, 3][0:15]
    y3 = data[3, 3][0:15]
    y4 = data[4, 3][0:15]
    y5 = data[5, 3][0:15]
    x = [i for i in range(1, 16)]
    ax = sns.lineplot(x=x, y=y0, color="purple")
    ax = sns.lineplot(x=x, y=y1, color="darkgreen")
    ax = sns.lineplot(x=x, y=y2, color="r")
    ax = sns.lineplot(x=x, y=y3, color="b")
    ax = sns.lineplot(x=x, y=y4, color="lime")
    ax = sns.lineplot(x=x, y=y5, color="orange")
    ax.set_xscale('linear')
    ax.set_yscale('log')
    #ax.set(xlabel = "number of fixpoint iterations")
    #ax.set(ylabel= "max(Boundary[i] - fixpoint_iter(Boundary[i]))")
    #plt.title('Konvergenzgeschwindigkeit', fontsize=17)
    ax.text(31, 0.5, "r, q, sigma = " + str(data[0, 0])+ ", " + str(data[0,1]) + ", " + str(data[0,2]),
            color="purple",fontsize=9)
    ax.text(31, 0.125, "r, q, sigma = " + str(data[1, 0]) + ", " + str(data[1, 1]) + ", " + str(data[1, 2]),
            color="darkgreen", fontsize=9)
    ax.text(31, 0.03125, "r, q, sigma = " + str(data[2, 0]) + ", " + str(data[2, 1]) + ", " + str(data[2, 2]),
            color="r", fontsize=9)
    ax.text(31, 0.007812, "r, q, sigma = " + str(data[3, 0]) + ", " + str(data[3, 1]) + ", " + str(data[3, 2]),
            color="b", fontsize=9)
    ax.text(31, 0.00195, "r, q, sigma = " + str(data[4, 0]) + ", " + str(data[4, 1]) + ", " + str(data[4, 2]),
            color="lime", fontsize=9)
    ax.text(31, 0.00048, "r, q, sigma = " + str(data[5, 0]) + ", " + str(data[5, 1]) + ", " + str(data[5, 2]),
            color="orange", fontsize=9)
    plt.savefig('6path_conv_small2', dpi = 333)
    plt.show()

def box_plots():
    data = np.array(load_and_return_trainingsdata("random1000_iter_under_10m6"))
    ax = sns.boxplot(x=data[:,3])
    ax.set(xlabel="number of iterations")
    plt.title('Laufzeit für einen Fehler<10^-6', fontsize=17)
    #plt.show()
    print("mean = ", data[:,3].mean())
    print("median =", np.median(data[:,3]))
    print("std = ", np.std(data[:,3]))
    print("max = ", np.max(data[:,3]), " at ", data[np.argmax(data[:,3])])
    print("min = ", np.min(data[:,3]), " at ", data[np.argmin(data[:,3])])
    plt.savefig('min_precision', dpi=333)


if __name__=="__main__":
    #create_data()
    #plotsmall()
    #data_creation()
    box_plots()



