import Option_Solver as os
import random as rr


def test():
    r, q, sigma, K, T = 0.02482163710806481, 0.07495768111897567, 0.002004477977022967, 71.49988043587692, 3.0
    option = os.Option_Solver(r, q, sigma, K, T, option_type='Put')
    option.create_boundary()
    print(option.QD_plus_exercise_vec)
    print(option.Early_exercise_vec)
    #print(option.max_diff)
    print(option.tau_grid)
    print(option.value_match_condition_vec(option.QD_plus_exercise_curve))

def masstesting():
    rr.seed(1)  # seed
    Z = 100 # passes
    sum = 0
    max = 0
    for i in range(Z):
        r = rr.uniform(0, 0.1)
        q = rr.uniform(0, 0.1)
        sigma = rr.uniform(0, 0.5)
        K = rr.uniform(50, 99)
        S = rr.uniform(50, 99)
        T = rr.uniform(3, 3)
        tau = rr.uniform(0, T)
        print("r, q, sigma, K, S, T, tau: ", r, q, sigma, K, S, T, tau)
        option = os.Option_Solver(r, q, sigma, K, T, option_type='Put', stop_by_diff=1e-6)
        option.create_boundary()
        print(option.max_diff)
        #print(option.used_iteration_steps)
        sum += option.used_iteration_steps
        if max < option.used_iteration_steps: max = option.used_iteration_steps
            #print("r, q, sigma, K, S, T, tau: ", r, q, sigma, K, S, T, tau)
            #print(option.Early_exercise_vec)
            #print(option.premium(S, tau))
            #print(option.max_diff)
            #print(option.used_iteration_steps)
            #print(option.training_data(10))
            #print(option.Early_exercise_vec)
    print(sum)
    print(max)






if __name__ =="__main__":
    masstesting()




