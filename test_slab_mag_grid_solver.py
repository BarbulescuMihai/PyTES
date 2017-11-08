"""
Created on Wed Nov 8 16:40:12 2017

@author: Mihai
"""

import main as pytes
from Test_Equations.Slab_mag_disp_rel_class import Asymmetric_magnetic_slab

import numpy as np
import matplotlib.pyplot as plt

slab = Asymmetric_magnetic_slab(R1=0.8, R2=0.9, kx0=None,
                                c0=1.5, v_A1=0.2, v_A2=0.2, M_A0=0, M_A1=0, M_A2=0)

x_range = np.linspace(0, 3, 101)
y_range = np.linspace(0, slab.c1, 101)

axes = {'x_axis':'kx0', 'y_axis':'vph'}
args = {'M_A0':slab.M_A0, 'M_A1':slab.M_A1, 'M_A2':slab.M_A2, 'v_A1':slab.v_A1, 'v_A2':slab.v_A2}

root_array = pytes.grid_solver(slab.disp_rel,
                               x_range, y_range,
                               axes, args,
                               method='sp.newton')

plt.figure(num=None, figsize=(8, 11), dpi=80, facecolor='w', edgecolor='k')
ax = plt.subplot()

ax.plot(root_array[:, 0], np.real(root_array[:, 1]), '.', color='b')
ax.plot(root_array[:, 0], np.imag(root_array[:, 1]), '.', color='r')

ax.set_xlim(x_range[0], x_range[-1])
ax.set_ylim(y_range[0], y_range[-1])

ax.set_ylabel(r'$vph$', fontsize=30)
ax.set_xlabel(r'$kx0$', fontsize=30)