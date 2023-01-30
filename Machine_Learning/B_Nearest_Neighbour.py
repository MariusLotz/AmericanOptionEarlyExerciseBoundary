from ast import literal_eval
from math import floor
import numpy as np
import Machine_Learning.Base as Base


class B_Nearest_Neigbour:

    txt_file = open("/home/user/PycharmProjects/AmericanOptionPricer/ML/Small_Sample_31_10")
    SIGMA = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    RQ = [0.02, 0.04, 0.06, 0.08, 0.1]
    mapping_rqs_boundary = {}
    w_vec = [0.06667134, 0.14945135, 0.21908636, 0.26926672, 0.29552422, 0.29552422,
            0.26926672, 0.21908636, 0.14945135, 0.06667134]
    tau_vec = [0.01304674, 0.06746832, 0.16029522, 0.2833023,  0.42556283, 0.57443717,
               0.7166977,  0.83970478 ,0.93253168, 0.98695326]
    K=100
    T=1
    option_type = 'Call'
    def __init__(self, r, q, sigma):
        self.load_and_process()
        self.r = r
        self.q = q
        self.sigma = sigma
        self.neighbouring_boundaries = []
        self.neighbours = self.get_neighbouring_rqs(r, q, sigma)
        self.boundary = self.B_nearest_neighbour_boundary(r, q, sigma, self.neighbours)


    def load_and_process(self):
        """Loading data from txt. file and saving it into a dictionary"""
        file = self.txt_file
        lines = file.readlines()
        for line in lines:
            [r, q, sigma, boundary, [s, premium]] = literal_eval(line)
            self.mapping_rqs_boundary[(r, q, sigma)] = np.array(boundary, dtype=float)

    def get_neighbouring_rqs(self, r_in, q_in, sigma_in):
        """Determine the neighbouring values (low, high) of r, q and sigma"""
        SIGMA_step_width = self.SIGMA[1] - self.SIGMA[0]
        RQ_step_width = self.RQ[1] - self.RQ[0]
        r_low = floor((r_in - self.RQ[0]) / RQ_step_width)
        r_high = floor((r_in - self.RQ[0]) / RQ_step_width) + 1
        q_low = floor((q_in - self.RQ[0]) / RQ_step_width)
        q_high = floor((q_in - self.RQ[0]) / RQ_step_width) + 1
        sigma_low = floor((sigma_in - self.SIGMA[0]) / SIGMA_step_width)
        sigma_high = floor((sigma_in - self.SIGMA[0]) / SIGMA_step_width) + 1
        return [[self.RQ[r_low], self.RQ[r_high]],
                [self.RQ[q_low], self.RQ[q_high]],
                [self.SIGMA[sigma_low], self.SIGMA[sigma_high]]]

    def metric(self, r, q, sigma, r_approx, q_approx, sigma_approx,  a1=0.2, a2=0.2, p1=1, p2=1, p3=1):
        """Distance for two elements of the form (r, q, sigma)"""
        return a1 * abs(r - r_approx)**p1 + a2 * abs(q - q_approx)**p2 + (1-a1-a2) * abs(sigma - sigma_approx)**p3

    def B_nearest_neighbour_boundary(self, r, q, sigma, neighbours):
        R = neighbours[0]
        Q = neighbours[1]
        SIGMA = neighbours[2]
        weighted_boundary = np.zeros(len(self.mapping_rqs_boundary[R[0], Q[0], SIGMA[0]]))
        sum_weight = 0
        for r_approx in R:
            for q_approx in Q:
                for sigma_approx in SIGMA:
                    val = self.mapping_rqs_boundary[(r_approx, q_approx, sigma_approx)]
                    self.neighbouring_boundaries.append(val)
                    weight = self.metric(r, q, sigma, r_approx, q_approx, sigma_approx)
                    sum_weight += weight
                    weighted_boundary += weight * val
        return weighted_boundary * (1 / sum_weight)

    def price(self, S, tau):
        price = Base.gaussian_premium(self.r, self.q, self.sigma, self.K, S, tau, self.tau_vec, self.boundary, self.w_vec, self.T, self.option_type)
        return price


def main():
    None


if __name__=="__main__":
    main()

