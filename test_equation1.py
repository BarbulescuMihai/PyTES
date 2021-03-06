"""
Test equation 1.
"""

import pytes

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#We define a function of two variables.
#The variable to be plotted on the y-axis must always be first.
def equation(y,x):
    return y*np.tan(x) + x*np.tan(y)

x_range = np.linspace(0, 2*np.pi, 200)
y_range = np.linspace(-2*np.pi, 2*np.pi, 200)

x_mesh, y_mesh = np.meshgrid(x_range, y_range)

z=equation(y_mesh, x_mesh)

kwargs = {'x-axis':'x'}
sign_change = pytes.grid_find_sign_change(equation, x_range, y_range, kwargs)

kwargs = {'x-axis':'x'}
root_array = pytes.grid_solver(equation, x_range, y_range, kwargs, method='sp.newton')

plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

gs = gridspec.GridSpec(1, 3)

ax1 = plt.subplot(gs[:,0])

ax1.plot(np.linspace(-2*np.pi, 2*np.pi, 1000),
         equation(1, np.linspace(-2*np.pi, 2*np.pi, 1000)), '.', color = 'b')

ax1.plot([-10,10], [0,0], '--', color='0.5')

ax1.set_xlim(-2*np.pi, 2*np.pi)
ax1.set_ylim(-10, 10)

ax1.set_xlabel(r'$x$', fontsize = 25)
ax1.set_ylabel(r'$func$', fontsize = 25)

ax1.set_title(r'$x \tan(y) + y \tan(x)$', fontsize=25)

ax2 = plt.subplot(gs[:,1])

ax2.plot(sign_change[:,0], np.real(sign_change[:,1]), '.', color = 'b')

ax2.plot([-10,10], [1,1], '--', color='0.5')

ax2.set_xlim(x_range[0], x_range[-1])
ax2.set_ylim(y_range[0], y_range[-1])

ax2.set_xlabel(r'$x$', fontsize = 25)
ax2.set_ylabel(r'$y$', fontsize = 25)

ax2.set_title(r'grid_find_sign_change', fontsize=25)

ax3 = plt.subplot(gs[:,2])

ax3.plot(root_array[:,0], np.real(root_array[:,1]), '.', color = 'b')

ax3.plot([-10,10], [1,1], '--', color='0.5')

ax3.set_xlim(x_range[0], x_range[-1])
ax3.set_ylim(y_range[0], y_range[-1])

ax3.set_xlabel(r'$x$', fontsize = 25)
ax3.set_ylabel(r'$y$', fontsize = 25)

ax3.set_title(r'grid_solver', fontsize=25)
