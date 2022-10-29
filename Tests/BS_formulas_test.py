import unittest
import BS_formulas


class QD_plus_tests(unittest.TestCase):
    def test00(self):
        S, K, r, q, sigma, tau = 100, 100, 0, 0, 0.2, 1
        put_price = BS_formulas.put_price(S, K, r, q, sigma, tau)
        call_price = BS_formulas.call_price(S, K, r, q, sigma, tau)
        self.assertAlmostEqual(put_price, call_price, 10)
        self.assertAlmostEqual(put_price, 7.97, 2 )

    def test02(self):
        S, K, r, q, sigma, tau = 50, 100, 0.1, 0.05, 0.3, 2
        put_price = BS_formulas.put_price(S, K, r, q, sigma, tau)
        call_price = BS_formulas.call_price(S, K, r, q, sigma, tau)
        self.assertAlmostEqual(call_price, 0.94, 2)
        self.assertAlmostEqual(put_price, 37.57, 2)

    def test03(self):
        S, K, r, q, sigma, tau = 130, 100, 0.1, 0.01, 0.5, 0.5
        put_price = BS_formulas.put_price(S, K, r, q, sigma, tau)
        call_price = BS_formulas.call_price(S, K, r, q, sigma, tau)
        self.assertAlmostEqual(call_price, 38.35, 2)
        self.assertAlmostEqual(put_price, 4.13, 2)



if __name__ == '__main__':
    unittest.main()