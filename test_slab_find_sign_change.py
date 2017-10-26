from Test_Equations.Slab_disp_rel_class import Asymmetric_slab

import numpy as np
import matplotlib.pyplot as plt
import main as pytes

slab = Asymmetric_slab(c0=0.6, R1=1.25, R2=1.25, K=None, M_A=0)

x_range = np.linspace(0, 2, 101)
y_range = np.linspace(-1, 1, 1001)

root_locs, roots = pytes.find_sign_change(slab.disp_rel, x_range, y_range,
                                          args=slab.M_A)

ax = plt.subplot()

ax.plot(root_locs, roots, 'b.')
ax.plot(root_locs, np.imag(roots), 'r.')

ax.set_xlim(x_range[0], x_range[-1])
ax.set_ylim(y_range[0], y_range[-1])