import Solver.Option_Solver as os

def put_call_symmetry_test():
    ### boundaries ###
    # creating instances:
    r, q, sigma, K, T, option_type = 0.05, 0.1, 0.25, 100, 2, 'Put'
    put_option = os.Option_Solver(r, q, sigma, K, T, option_type)
    r, q, sigma, K, T, option_type = 0.1, 0.05, 0.25, 100, 2, 'Call'
    call_option = os.Option_Solver(r, q, sigma, K, T, option_type)

    # check symmetry relation for QD_plus_boundary
    symmetry_QD_plus_exercise_vec  = [call_option.K**2 / z for z in call_option.QD_plus_exercise_vec]
    max_diff = -1
    for i in range(len(put_option.QD_plus_exercise_vec)):
        if abs(symmetry_QD_plus_exercise_vec[i] - put_option.QD_plus_exercise_vec[i]) > max_diff:
            max_diff = symmetry_QD_plus_exercise_vec[i] - put_option.QD_plus_exercise_vec[i]
    # print(max_diff)
    assert max_diff < 1e-7

    # creating boundaries:
    put_option.create_boundary()
    call_option.create_boundary()

    # check symmetry relation for exact boundary
    symmetry_Early_exercise_vec = [call_option.K ** 2 / z for z in call_option.Early_exercise_vec]
    max_diff = -1
    for i in range(len(put_option.Early_exercise_vec)):
        if abs(symmetry_Early_exercise_vec[i] - put_option.Early_exercise_vec[i]) > max_diff:
            max_diff = symmetry_Early_exercise_vec[i] - put_option.Early_exercise_vec[i]
    #print(max_diff)
    assert max_diff < 1e-6
    ###

    ### Prices ###
    # creating instances:
    r, q, sigma, K, T, option_type = 0.01, 0.02, 0.01, 120, 2, 'Put'
    put_option = os.Option_Solver(r, q, sigma, K, T, option_type)
    r, q, sigma, K, T, option_type = 0.02, 0.01, 0.01, 80, 2, 'Call'
    call_option = os.Option_Solver(r, q, sigma, K, T, option_type)

    # check symmetry relation for American/European and premium for QD_plus_boundary
    for tau in [0.5, 1, 1.5, 2]:
        diff_am = call_option.American_price(120, tau) - put_option.American_price(80, tau)
        diff_eu = call_option.European_price(120, tau) - put_option.European_price(80, tau)
        diff_prem = call_option.premium(120, tau) - put_option.premium(80, tau)
        #print(diff_am, diff_eu, diff_prem)
        assert diff_eu < 1e-8

    # check symmetry relation for American/European and premium for exact boundary
    call_option.create_boundary()
    put_option.create_boundary()
    for tau in [0.5, 1, 1.5, 2]:
        diff_am = call_option.American_price(120, tau) - put_option.American_price(80, tau)
        diff_eu = call_option.European_price(120, tau) - put_option.European_price(80, tau)
        diff_prem = call_option.premium(120, tau) - put_option.premium(80, tau)
        #print(diff_am, diff_eu, diff_prem)
        assert diff_eu < 1e-8


if __name__=="__main__":
    put_call_symmetry_test()