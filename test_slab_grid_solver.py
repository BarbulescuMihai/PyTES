import main as pytes
from Test_Equations.Slab_disp_rel_class import Asymmetric_slab

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle

def save():
    with open('pickles/wavenumber_c0={}_R1={}_R2={}_K={}_M_A={}.p'.format(
    				slab.c0, slab.R1, slab.R2, slab.K, slab.M_A), 'wb') as f:
    	    pickle.dump(root_array, f)

slab = Asymmetric_slab(c0=0.6, R1=1.4, R2=1.6, K=None, M_A=1)

x_range = np.linspace(0, 2, 51)
y_range = np.linspace(0, 1, 51)

root_array = pytes.grid_solver(slab.disp_rel,
                             x_range, y_range,
                             args=(slab.K, slab.M_A),
                             method='sp.newton')

font = {'size': 15}
matplotlib.rc('font', **font)

plt.figure(num=None, figsize=(8, 11), dpi=80, facecolor='w', edgecolor='k')
ax = plt.subplot()

ax.plot(root_array[:,0], np.real(root_array[:,1]), '.', color = 'b')
ax.plot(root_array[:,0], np.imag(root_array[:,1]), '.', color = 'r')

ax.set_xlim(x_range[0], x_range[-1])
ax.set_ylim(y_range[0], y_range[-1])

ax.set_ylabel(r'$\omega/k v_A$', fontsize = 30)
ax.set_xlabel(r'$k x_0$', fontsize = 30)

ax.plot([x_range[0], x_range[-1]], [slab.c0 + slab.M_A, slab.c0 + slab.M_A],
        [x_range[0], x_range[-1]], [-slab.c0 + slab.M_A, -slab.c0 + slab.M_A],
        [x_range[0], x_range[-1]], [slab.cT + slab.M_A, slab.cT + slab.M_A],
        [x_range[0], x_range[-1]], [-slab.cT + slab.M_A, -slab.cT + slab.M_A],
        [x_range[0], x_range[-1]], [1 + slab.M_A, 1 + slab.M_A],
        [x_range[0], x_range[-1]], [-1 + slab.M_A, -1 + slab.M_A],
        color = '0.5', linestyle=':', linewidth=2)

if slab.cT + slab.M_A < y_range[-1]:
    ax.annotate(r'$c_T+M_A$', xy=(x_range[-1] + 0.03, slab.cT + slab.M_A - 0.01),
                xycoords='data', annotation_clip=False, fontsize=20)
if slab.c0 + slab.M_A < y_range[-1]:
    ax.annotate(r'$c_0+M_A$', xy=(x_range[-1] + 0.03, slab.c0 + slab.M_A - 0.01),
                xycoords='data', annotation_clip=False, fontsize=20)
if 1 + slab.M_A < y_range[-1]:
    ax.annotate(r'$1+M_A$', xy=(x_range[-1] + 0.03, 1+slab.M_A - 0.01),
			xycoords='data', annotation_clip=False, fontsize=20)

ax.annotate(r'$-c_T+M_A$', xy=(x_range[-1] + 0.03, -slab.cT + slab.M_A - 0.01),
			xycoords='data', annotation_clip=False, fontsize=20)
ax.annotate(r'$-c_0+M_A$', xy=(x_range[-1] + 0.03, -slab.c0 + slab.M_A - 0.01),
			xycoords='data', annotation_clip=False, fontsize=20)
ax.annotate(r'$-1+M_A$', xy=(x_range[-1] + 0.03, -1+slab.M_A - 0.01),
			xycoords='data', annotation_clip=False, fontsize=20)

if slab.c1 == slab.c2:
    ax.plot([x_range[0], x_range[-1]], [slab.c1, slab.c1],
             [x_range[0], x_range[-1]], [-slab.c1, -slab.c1],
             color = '0.5', linestyle=':', linewidth=2)
    ax.annotate(r'$c_1, c_2$', xy=(x_range[-1] + 0.03, slab.c1 - 0.01),
                 xycoords='data', annotation_clip=False, fontsize=20)
    ax.annotate(r'$-c_1, -c_2$', xy=(x_range[-1] + 0.03, -slab.c1 - 0.01),
                 xycoords='data', annotation_clip=False, fontsize=20)
else:
    ax.plot([x_range[0], x_range[-1]], [slab.c1, slab.c1],
            [x_range[0], x_range[-1]], [-slab.c1, -slab.c1],
            [x_range[0], x_range[-1]], [slab.c2, slab.c2],
            [x_range[0], x_range[-1]], [-slab.c2, -slab.c2],
            color = '0.5', linestyle=':', linewidth=2)
    ax.annotate(r'$c_1$', xy=(x_range[-1] + 0.03, slab.c1 - 0.01),
                xycoords='data', annotation_clip=False, fontsize=20)
    ax.annotate(r'$-c_1$', xy=(x_range[-1] + 0.03, -slab.c1 - 0.01),
                xycoords='data', annotation_clip=False, fontsize=20)
    ax.annotate(r'$c_2$', xy=(x_range[-1] + 0.03, slab.c2 - 0.01),
                xycoords='data', annotation_clip=False, fontsize=20)
    ax.annotate(r'$-c_2$', xy=(x_range[-1] + 0.03, -slab.c2 - 0.01),
                xycoords='data', annotation_clip=False, fontsize=20)