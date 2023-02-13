def test_model_for_boundary(model, x_train, S=[80, 120]):
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