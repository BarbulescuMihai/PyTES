"""
Created on Wed Nov 8 16:34:22 2017

@author: Mihai
"""

import main as pytes
from test_class_slab import Asymmetric_slab

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

slab = Asymmetric_slab(R1=1.4, R2=1.4, kx0=1, c0=0.6, M_A=0)

x_range_pts = np.linspace(0, 2, 1000)
y_range_pts = np.linspace(0, 1, 1000)

x_range = np.linspace(0, 2, 51)
y_range = np.linspace(0, 1, 51)

kwargs = {'x-axis':'kx0', 'M_A':slab.M_A}
points = pytes.grid_find_sign_change(slab.disp_rel, x_range_pts, y_range_pts, kwargs)

kwargs = {'x-axis':'kx0', 'M_A':slab.M_A}
root_array = pytes.grid_solver(slab.disp_rel, x_range, y_range, kwargs, method='sp.brentq')

gs = gridspec.GridSpec(1,2)

plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
ax1 = plt.subplot(gs[:,1])

ax1.plot(root_array[:, 0], np.real(root_array[:, 1]), '.', color='b')
ax1.plot(root_array[:, 0], np.imag(root_array[:, 1]), '.', color='r')

ax1.set_xlim(x_range[0], x_range[-1])
ax1.set_ylim(y_range[0], y_range[-1])

ax1.set_ylabel(r'$vph$', fontsize=30)
ax1.set_xlabel(r'$kx_0$', fontsize=30)

ax1.set_title(r'grid_solver', fontsize=25)

ax2 = plt.subplot(gs[:,0])

ax2.plot(points[:, 0], points[:, 1], 'b,')

ax2.set_xlim(x_range_pts[0], x_range_pts[-1])
ax2.set_ylim(y_range_pts[0], y_range_pts[-1])

ax2.set_xlabel(r'$kx_0$', fontsize=30)

ax2.set_title(r'grid_find_sign_change', fontsize=25)

x_range_pts = np.linspace(0, 2, 1000)
y_range_pts = np.linspace(0, 1, 1000)

x_range = np.linspace(0, 2, 101)
y_range = np.linspace(0, 1, 101)

kwargs = {'x-axis':'M_A', 'kx0':slab.kx0}
points = pytes.grid_find_sign_change(slab.disp_rel, x_range_pts, y_range_pts, kwargs)

kwargs = {'x-axis':'M_A', 'kx0':slab.kx0}
root_array = pytes.grid_solver(slab.disp_rel, x_range, y_range, kwargs, method='sp.newton')

plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
ax3 = plt.subplot(gs[:,1])

ax3.plot(root_array[:, 0], np.real(root_array[:, 1]), '.', color='b')
ax3.plot(root_array[:, 0], np.imag(root_array[:, 1]), '.', color='r')

ax3.set_xlim(x_range[0], x_range[-1])
ax3.set_ylim(y_range[0], y_range[-1])

ax3.set_ylabel(r'$vph$', fontsize=30)
ax3.set_xlabel(r'$M_A$', fontsize=30)

ax3.set_title(r'grid_solver', fontsize=25)

ax4 = plt.subplot(gs[:,0])

ax4.plot(points[:, 0], points[:, 1], 'b,')

ax4.set_xlim(x_range_pts[0], x_range_pts[-1])
ax4.set_ylim(y_range_pts[0], y_range_pts[-1])

ax4.set_xlabel(r'$M_A$', fontsize=30)

ax4.set_title(r'grid_find_sign_change', fontsize=25)