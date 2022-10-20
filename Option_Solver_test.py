import unittest
import Option_Solver as OS

RE = 10e-6  # relative difference allowed

class Option_Solver_tests(unittest.TestCase):
    """testcases have the form r_q_sigma_tau_K_S"""

    def test_paper_005_005_025_1_100_100(self):
        r, q, sigma, tau, K, S = 0.05, 0.05, 0.25, 1, 100, 100
        option = OS.Option_Solver(r, q, sigma, K, tau)
        option.create_boundary()
        premium = option.premium(S, tau)
        rel_diff = abs(premium - 0.106952702747) / premium
        #print(rel_diff)
        self.assertTrue(rel_diff < RE)

    def test_paper_004_004_02_3_100_80(self):
        r, q, sigma, tau, K, S = 0.04, 0.04, 0.2, 3, 100, 80
        option = OS.Option_Solver(r, q, sigma, K, tau)
        option.create_boundary()
        price = option.put_price(S, tau)
        rel_diff = abs(price - 23.22834) / price
        #print(rel_diff)
        self.assertTrue(rel_diff < RE)

    def test_paper_004_004_02_3_100_100(self):
        r, q, sigma, tau, K, S = 0.04, 0.04, 0.2, 3, 100, 100
        option = OS.Option_Solver(r, q, sigma, K, tau)
        option.create_boundary()
        price = option.put_price(S, tau)
        rel_diff = abs(price - 12.60521) / price
        #print(rel_diff)
        self.assertTrue(rel_diff < RE)

    def test_paper_004_004_02_3_100_120(self):
        r, q, sigma, tau, K, S = 0.04, 0.04, 0.2, 3, 100, 120
        option = OS.Option_Solver(r, q, sigma, K, tau)
        option.create_boundary()
        price = option.put_price(S, tau)
        rel_diff = abs(price - 6.482425) / price
        #print(rel_diff)
        self.assertTrue(rel_diff < RE)


if __name__=="__main__":
    unittest.main()