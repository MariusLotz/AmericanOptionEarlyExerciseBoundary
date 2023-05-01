from Solver.Option_Solver import Option_Solver

if __name__=="__main__":
    ### Option-Inputparamter:
    r, q, sigma, K, T, option_type = 0.1, 0.05, 0.25, 100, 2, 'Put'

    ### Further parameters:
    l=99  # number of basis points for the Gauss quadrature
    m=25  # number of basis points for the Chebyshev polynomial
    n=15  # number of fixpoint iteration steps
    stop_by_diff=1e-6  # set stop_by_diff=None for fixed n


    ### Creating instance:
    option = Option_Solver(r, q, sigma, K, T, option_type)

    ### calculate Early Exercise Boundary:
    option.create_boundary()

    ### calculate prices:
    S, tau = 100, 1  # tau has to be in [0, T]
    American_price = option.American_price(S, tau) 
    American_premium = option.premium(S, tau)
    European_price = option.European_price(S, tau)

    ### Output:
    print('American_price:', American_price)
    print('American_premium:', American_premium)
    print('European_price:', European_price)
    




