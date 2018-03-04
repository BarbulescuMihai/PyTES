# -*- coding: utf-8 -*-
"""
Created on Fri Jul 7 11:18:05 2017

@author: Mihai
"""

import scipy as sc

"""
We define

R1 = rho_1/rho_0
R2 = rho_2/rho_0
M_A = U_0/v_A
K = k x_0
c_ph = omega/v_A
"""

class Asymmetric_magnetic_slab:

    def __init__(self, R1, R2, kx0, c0, v_A1, v_A2, M_A0, M_A1, M_A2):

        self.R1 = R1
        self.R2 = R2

        self.c0 = c0
        self.v_A1 = v_A1
        self.v_A2 = v_A2

        self.c1 = sc.sqrt(self.R1**(-1) * (self.c0**2 + 5/6) - 1/2 * v_A1**2)
        self.c2 = sc.sqrt(self.R2**(-1) * (self.c0**2 + 5/6) - 1/2 * v_A2**2)

        self.c_T0 = sc.sqrt(self.c0**2 /
                           (1 + self.c0**2))
        self.c_T1 = sc.sqrt(self.c1**2 * self.v_A1**2 /
                           (self.v_A1**2 + self.c1**2))
        self.c_T2 = sc.sqrt(self.c2**2 * self.v_A2**2 /
                           (self.v_A2**2 + self.c2**2))

        self.kx0 = kx0

        self.M_A0 = M_A0
        self.M_A1 = M_A1
        self.M_A2 = M_A2

    def m0(self, c_ph, M_A0):
        return sc.sqrt( (1 - (c_ph - M_A0)**2) * (self.c0**2 - (c_ph - M_A0)**2) /
                      ( (self.c0**2 + 1) * (self.c_T0**2 - (c_ph - M_A0)**2)) )

    def m1(self, c_ph, M_A1):
        return sc.sqrt( (self.v_A1**2 - (c_ph - M_A1)**2) * (self.c1**2 - (c_ph - M_A1)**2 ) /
                       ((self.c1**2 + self.v_A1**2) * (self.c_T1**2 - (c_ph - M_A1)**2) ) )

    def m2(self, c_ph, M_A2):
        return sc.sqrt( (self.v_A2**2 - (c_ph - M_A2)**2) * (self.c2**2 - (c_ph - M_A2)**2 ) /
                       ((self.c2**2 + self.v_A2**2) * (self.c_T2**2 - (c_ph - M_A2)**2) ) )

    #function to be solved
    def disp_rel(self, c_ph, kx0, v_A1, v_A2, M_A0, M_A1, M_A2):
        return (1 - (c_ph - M_A0)**2 )**2 + \
            self.m0(c_ph, M_A0)**2 * self.R1 * self.R2 / (self.m1(c_ph, M_A1) * self.m2(c_ph, M_A2)) * \
            (v_A1**2 - (c_ph - M_A1)**2) * (v_A2**2 - (c_ph - M_A2)**2) + \
            1/2 * (1 - (c_ph - M_A0)**2) * \
            (self.m0(c_ph, M_A0) * self.m1(c_ph, M_A1)**(-1) * self.R1 * (v_A1**2 - (c_ph - M_A1)**2) +
            self.m0(c_ph, M_A0) * self.m2(c_ph, M_A2)**(-1) * self.R2 * (v_A2**2 - (c_ph - M_A2)**2)) * \
            ( sc.tanh(self.m0(c_ph, M_A0) * kx0) + sc.tanh(self.m0(c_ph, M_A0) * kx0)**(-1) )
