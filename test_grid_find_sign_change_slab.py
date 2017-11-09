from Test_Equations.Slab_disp_rel_class import Asymmetric_slab

import numpy as np
import matplotlib.pyplot as plt
import main as pytes

slab = Asymmetric_slab(R1=1.25, R2=1.25, kx0=None, c0=0.6, M_A=0)

x_range = np.linspace(0, 2, 1000)
y_range = np.linspace(0, 1, 1000)

axes = {'x_axis':'kx0', 'y_axis':'vph'}
args = {'M_A':slab.M_A}

points = pytes.grid_find_sign_change(slab.disp_rel, x_range, y_range, axes, args)

ax = plt.subplot()

ax.plot(points[:, 0], points[:, 1], 'b,')

ax.set_xlim(x_range[0], x_range[-1])
ax.set_ylim(y_range[0], y_range[-1])
