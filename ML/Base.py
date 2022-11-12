import numpy as np
import scipy.stats as stats
import Option_Solver as os
import tensorflow as tf

def d_minus(r, q, sigma, tau, X):
    """d_- from Black Scholes formula"""
    return (np.log(X) + (r - q) * tau - 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))

def d_plus(r, q, sigma,  tau, X):
    """d_+ from Black Scholes formula"""
    return (np.log(X) + (r - q) * tau + 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))

def gaussian_premium(r, q, sigma, K, S, tau, tau_vec, boundary_vec, w_vec, T, option_type):
    def integrand(u, Bu):
        z = S / Bu
        if option_type == 'Put':
            a = r * K * np.exp(-r * (tau - u)) * stats.norm.cdf(-d_minus(r, q, sigma, tau - u, z))
            b = q * S * np.exp(-q * (tau - u)) * stats.norm.cdf(-d_plus(r, q, sigma, tau - u, z))
        else:  # Call:
            a = q * S * np.exp(-q * (tau - u)) * stats.norm.cdf(d_plus(r, q, sigma, tau - u, z))
            b = r * K * np.exp(-r * (tau - u)) * stats.norm.cdf(d_minus(r, q, sigma, tau - u, z))
        return a - b
    sum = 0
    for i in range(len(w_vec)):
        sum += integrand(tau_vec[i], boundary_vec[i])* w_vec[i]
    return  T/2.0 * sum

def load_and_return_trainingsdata(file_path):
    """Loading data from txt. file and return trainingsdata"""
    file = open(file_path)
    lines = file.readlines()
    x_train = []
    y_train = []
    for line in lines:
        try:
            [r, q, sigma, boundary, [s, premium]] = eval(line)
            x_train.append([r,q, sigma])
            y_train.append(boundary)
        except: break


    return x_train, y_train




def test_model_on_test_data(model, x_train, S=[80, 120]):
    option_type = 'Call'
    K= 100
    T= 1
    np.random.seed(seed=1)
    big = 0
    for i in range(len(x_train)):
        r, q, sigma = np.random.uniform(0.01, 0.05), np.random.uniform(0.01, 0.05), np.random.uniform(0.1, 0.45)

        option = os.Option_Solver(r, q, sigma, K, T, option_type)
        option.create_boundary()
        tau_vec, boundary, w_vec = option.gaussian_grid_boundary(n=25)

        x = tf.constant([[r, q, sigma]])
        pred_boundary = model(x).numpy()[0] * max(100, 100 * x[0][0] / x[0][1])  # resize
        #print([(pred_boundary[i] - boundary[i]) for i in range(len(pred_boundary))])

        for s in S:
            pred_prem = gaussian_premium(x_train[i][0], x_train[i][1], x_train[i][2], K, s, T, tau_vec, pred_boundary, w_vec, T, option_type)
            prem = gaussian_premium(x_train[i][0], x_train[i][1], x_train[i][2], K, s, T, tau_vec, boundary, w_vec, T, option_type)
            if (pred_prem-prem)/(pred_prem) > 0.2*big:
                big = (pred_prem-prem)/pred_prem
                print("r = ", r , "q =", q, "sigma =", sigma, "(pred - prem) / pred = ", big, "S= ", s)
                print(boundary)
                print(pred_boundary)
            #print("pred_prem= ", pred_prem, "prem= ", prem, "(pred - prem) / pred = ", (pred_prem-prem)/pred_prem)
        #print()
        #print()



def test_model_on_training_data(model, x_train, y_train, S=[100, 110, 120, 130]):
    option_type = 'Call'
    K= 100
    T= 1
    option = os.Option_Solver(0.05, 0.05, 0.35, K, T, option_type)
    option.create_boundary()
    tau_vec, boundary_vec, w_vec = option.gaussian_grid_boundary(n=25)
    big = 0
    sum = 0
    for i in range(len(x_train)):
        x = tf.constant([x_train[i]])
        pred_boundary = model(x).numpy()[0] * max(100, 100 * x[0][0]/ x[0][1])  # resize
        boundary = y_train[i]
        #print([(pred_boundary[i] - boundary[i]) for i in range(len(pred_boundary))])

        for s in S:
            pred_prem = gaussian_premium(x_train[i][0], x_train[i][1], x_train[i][2], K, s, T, tau_vec, pred_boundary, w_vec, T, option_type)
            prem = gaussian_premium(x_train[i][0], x_train[i][1], x_train[i][2], K, s, T, tau_vec, boundary, w_vec, T, option_type)
            sum += (pred_prem - prem) / pred_prem
            if (pred_prem - prem) / pred_prem > 0.5 * big:
                print("r = ", x_train[i][0], "q =", x_train[i][1], "sigma =", x_train[i][2], "(pred - prem) / pred = ",
                      (pred_prem - prem) / pred_prem, "S =", s)
            if (pred_prem-prem)/(pred_prem) > big:
                big = (pred_prem-prem)/pred_prem

            #print("pred_prem= ", pred_prem, "prem= ", prem, "(pred - prem) / pred = ", (pred_prem-prem)/pred_prem)
        #print()
        #print()
    print(sum / len(x_train))

def create_trainings_data():
    T = 1
    K = 100
    S = 100
    SIGMA = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    RQ = [0.02, 0.04, 0.06, 0.08, 0.1]
    file = open('test', 'a')
    counter = 0
    option_type = "Call"
    for sigma in SIGMA:
        for r in RQ:
            for q in RQ:
                counter +=1
                print(counter, " out of ", len(SIGMA) * len(RQ)**2)
                option = os.Option_Solver(r, q, sigma, K, T, option_type)
                option.create_boundary()
                tau_vec, boundary_vec, w_vec = option.gaussian_grid_boundary(n=10)
                premium = gaussian_premium(r, q, sigma, K, S, T, tau_vec, boundary_vec, w_vec, T, option_type='Call')
                entry = [r, q, sigma, list(boundary_vec), [S, premium]]
                file.write(str(entry))
                file.write("\n")
    file.close()

if __name__=="__main__":
    model_path = "/home/user/PycharmProjects/AmericanOptionPricer/ML/5NN_of_Small_Sample_7_11"
    file_path = "/home/user/PycharmProjects/AmericanOptionPricer/ML/Large_Sample_6_11"
    x, y = load_and_return_trainingsdata(file_path)
    test_model_on_test_data(tf.keras.models.load_model(model_path, compile=False), x)
    #test_model_on_training_data(tf.keras.models.load_model(model_path, compile=False), x, y)