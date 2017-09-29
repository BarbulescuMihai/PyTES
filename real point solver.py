#import toolbox as tool
from AS1_class import Asym_slab

import numpy as np
#import scipy.optimize as sp
import matplotlib.pyplot as plt

slab = Asym_slab(c0=0.6, R1=1.25, R2=1.25, K=None, M_A=1)

x_range = np.linspace(0, 2, 101)
y_range = np.linspace(-1, 1, 1001)

x_grid, y_grid = np.meshgrid(x_range, y_range)
func_grid = slab.disp_rel(y_grid, x_grid, slab.M_A)

grid_shift_up = func_grid[1:, :]
grid_shift_down = func_grid[:-1, :]
grid_prod = np.real(grid_shift_down) * np.real(grid_shift_up)

root_locs = x_grid[(grid_prod < 0) * (func_grid[:-1,:] < 10)]
roots = y_grid[(grid_prod < 0) * (func_grid[:-1,:] < 10)]

ax = plt.subplot()

ax.plot(root_locs, roots, 'b.')
ax.plot(root_locs, np.imag(roots), 'r.')

ax.set_xlim(x_range[0], x_range[-1])
ax.set_ylim(y_range[0], y_range[-1])