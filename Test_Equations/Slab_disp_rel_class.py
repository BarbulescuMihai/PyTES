# -*- coding: utf-8 -*-
"""
Created on Sat Jan 7 18:03:02 2017

@author: Mihai
"""

import scipy as sc

"""
We define

R1 = rho_1/rho_0
R2 = rho_2/rho_0
M_A = U_0/v_A
K = k x_0
W = omega/v_A
"""

class Asymmetric_slab:

    def __init__(self, c0, R1, R2, K, M_A):
        self.c0 = c0
        self.R1 = R1
        self.R2 = R2

        self.c1 = sc.sqrt(1/self.R1 * (self.c0**2 + 5/6))
        self.c2 = sc.sqrt(1/self.R2 * (self.c0**2 + 5/6))
        self.cT = sc.sqrt(self.c0**2 / (1 + self.c0**2))

        self.K = K
        self.M_A = M_A

    def m1(self, W):
        return sc.sqrt(1 - W**2 / self.c1**2)

    def m2(self, W):
        return sc.sqrt(1 - W**2 / self.c2**2)

    def m0(self, W, M_A):
        return sc.sqrt((1 - (W - M_A)**2) * (self.c0**2 - (W - M_A)**2) / ((1 + self.c0**2) * (self.cT**2 - (W - M_A)**2)))

    def disp_rel(self, W, K, M_A):
        return self.m0(W, M_A)**2 * W**4 + 1/self.R1 * 1/self.R2 * self.m1(W) * self.m2(W) * (1 - (W - M_A)**2)**2 - \
	           0.5 * W**2 * self.m0(W, M_A) * (1 - (W - M_A)**2) * (1/self.R1 * self.m1(W) + 1/self.R2 * self.m2(W)) * \
	           (sc.tanh(self.m0(W, M_A) * K) + sc.tanh(self.m0(W, M_A) * K)**(-1))