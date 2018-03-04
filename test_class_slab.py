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
kx0 = k x_0
vph = omega/v_A
"""

class Asymmetric_slab:

    def __init__(self, R1, R2, kx0, c0, M_A):
        self.c0 = c0
        self.R1 = R1
        self.R2 = R2

        self.c1 = sc.sqrt(1/self.R1 * (self.c0**2 + 5/6))
        self.c2 = sc.sqrt(1/self.R2 * (self.c0**2 + 5/6))
        self.cT = sc.sqrt(self.c0**2 / (1 + self.c0**2))

        self.kx0 = kx0
        self.M_A = M_A

    def m1(self, vph):
        return sc.sqrt(1 - vph**2 / self.c1**2)

    def m2(self, vph):
        return sc.sqrt(1 - vph**2 / self.c2**2)

    def m0(self, vph, M_A):
        return sc.sqrt((1 - (vph - M_A)**2) * (self.c0**2 - (vph - M_A)**2) / ((1 + self.c0**2) * (self.cT**2 - (vph - M_A)**2)))

    #function to be solved
    def disp_rel(self, vph, kx0, M_A):
        return self.m0(vph, M_A)**2 * vph**4 + 1/self.R1 * 1/self.R2 * self.m1(vph) * self.m2(vph) * (1 - (vph - M_A)**2)**2 - \
	           0.5 * vph**2 * self.m0(vph, M_A) * (1 - (vph - M_A)**2) * (1/self.R1 * self.m1(vph) + 1/self.R2 * self.m2(vph)) * \
	           (sc.tanh(self.m0(vph, M_A) * kx0) + sc.tanh(self.m0(vph, M_A) * kx0)**(-1))
