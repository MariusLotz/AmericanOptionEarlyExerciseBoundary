"""Testing of the QD_plus implementation compared to the results in the pdf
MPRA_paper_15018 starting on the page 18. Allowing a 5% difference, because of
rounding mistakes happening during the calculation"""
import unittest
import QD_plus_exercise_boundary

RE = 0.5  # relative difference allowed

class QD_plus_tests(unittest.TestCase):
    """testcases have the form K_r_q_sigma_tau"""

    def test_100_002_002_02_01(self):
        b = QD_plus_exercise_boundary.exercise_boundary(100, 0.02, 0.02, 0.2, [0.1])[0]
        rel_diff = abs(b - 83.99) / b
        print(rel_diff)
        self.assertTrue(rel_diff < RE)

    def test_100_008_002_06_01(self):
        b = QD_plus_exercise_boundary.exercise_boundary(100, 0.08, 0.02, 0.6, [0.1])[0]
        rel_diff = abs(b - 69.31) / b
        print(rel_diff)
        self.assertTrue(rel_diff < RE)

    def test_100_002_004_02_05(self):
        b = QD_plus_exercise_boundary.exercise_boundary(100, 0.02, 0.04, 0.2, [0.5])[0]
        rel_diff = abs(b - 45.90) / b
        print(rel_diff)
        self.assertTrue(rel_diff < RE)

    def test_100_002_004_04_05(self):
        b = QD_plus_exercise_boundary.exercise_boundary(100, 0.02, 0.04, 0.4, [0.5])[0]
        rel_diff = abs(b - 41.08) / b
        print(rel_diff)
        self.assertTrue(rel_diff < RE)

    def test_100_004_008_04_1(self):
        b = QD_plus_exercise_boundary.exercise_boundary(100, 0.04, 0.08, 0.4, [1])[0]
        rel_diff = abs(b - 37.46) / b
        print(rel_diff)
        self.assertTrue(rel_diff < RE)

    def test_100_004_004_06_3(self):
        b = QD_plus_exercise_boundary.exercise_boundary(100, 0.04, 0.04, 0.2, [3])[0]
        rel_diff = abs(b - 61.52) / b
        print(rel_diff)
        self.assertTrue(rel_diff < RE)

    def test_100_008_0_06_01(self):
        b = QD_plus_exercise_boundary.exercise_boundary(100, 0.08, 0, 0.6, [0.1])[0]
        rel_diff = abs(b - 70.47) / b
        print(rel_diff)
        self.assertTrue(rel_diff < RE)

    def test_35_00488_0_06_00833(self):
        b = QD_plus_exercise_boundary.exercise_boundary(35, 0.0488, 0, 0.2, [0.0833])[0]
        rel_diff = abs(b - 31.80) / b
        print(rel_diff)
        self.assertTrue(rel_diff < RE)

    def test_45_00488_0_06_05833(self):
        b = QD_plus_exercise_boundary.exercise_boundary(45, 0.0488, 0, 0.4, [0.5833])[0]
        rel_diff = abs(b - 28.69) / b
        print(rel_diff)
        self.assertTrue(rel_diff < RE)

if __name__ == '__main__':
    unittest.main()
