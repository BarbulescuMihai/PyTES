"""
Created on Wed Nov 8 16:34:22 2017

@author: Mihai
"""

import main as pytes
from Test_Equations.Slab_disp_rel_class import Asymmetric_slab

import numpy as np
import matplotlib.pyplot as plt

slab = Asymmetric_slab(R1=1.4, R2=1.6, kx0=None, c0=0.6, M_A=0)

x_range = np.linspace(0, 2, 51)
y_range = np.linspace(0, 1, 51)

axes = {'x_axis':'kx0', 'y_axis':'vph'}
args = {'M_A':slab.M_A}

root_array = pytes.grid_solver(slab.disp_rel,
                               x_range, y_range,
                               axes, args,
                               method='sp.brentq')

plt.figure(num=None, figsize=(8, 11), dpi=80, facecolor='w', edgecolor='k')
ax = plt.subplot()

ax.plot(root_array[:, 0], np.real(root_array[:, 1]), '.', color='b')
ax.plot(root_array[:, 0], np.imag(root_array[:, 1]), '.', color='r')

ax.set_xlim(x_range[0], x_range[-1])
ax.set_ylim(y_range[0], y_range[-1])

ax.set_ylabel(r'$vph$', fontsize=30)
ax.set_xlabel(r'$kx0$', fontsize=30)
