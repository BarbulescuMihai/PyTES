import solvers as sol
from AS1_class import Asym_slab

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
from pathlib import Path

def save():
    with open('pickles/wavenumber_c0={}_R1={}_R2={}_K={}_M_A={}.p'.format(
    				slab.c0, slab.R1, slab.R2, slab.K, slab.M_A), 'wb') as f:
    	    pickle.dump(root_array, f)

slab = Asym_slab(c0=0.6, R1=1.4, R2=1.4, K=None, M_A=0)

x_range = np.linspace(0, 2, 101)
y_range = np.linspace(0, 1, 101)

if Path('pickles/wavenumber_c0={}_R1={}_R2={}_K={}_M_A={}.p'
        .format(slab.c0, slab.R1, slab.R2, slab.K, slab.M_A)).is_file():
    root_array = pickle.load(open('pickles/wavenumber_c0={}_R1={}_R2={}_K={}_M_A={}.p'.format(
							slab.c0, slab.R1, slab.R2, slab.K, slab.M_A), 'rb'))
    if root_array[-1, 0] != x_range[-1] or root_array[-1, 0] != x_range[-1]:
        root_array = sol.point_finder_sp(slab.disp_rel, x_range, y_range, args=(slab.K, slab.M_A))
        save()
else:
    root_array = sol.point_finder_sp(slab.disp_rel, x_range, y_range, args=(slab.K, slab.M_A))
    save()

use_mp = True

root_array_backup = root_array

root_array = np.delete(root_array, np.where(root_array[:,0]==0), 0)
root_array = np.delete(root_array, np.where(root_array[:,1]==1), 0)

list_of_lines = []
list_of_shapes = []

i = 0

while i <= len(root_array[:,0]) and len(root_array[:,0]) > 0:

    #the code needs to be modified to remove this try statement
    try:
        x, y = root_array[i,0], root_array[i,1]
    except IndexError:
        break

    try:
        line = sol.line_trace_sp(slab.disp_rel, x, y, 0.001, x_range[0], x_range[-1], args=(slab.K, slab.M_A))
    except:
        if use_mp is True:
            try:
                line = sol.line_trace_mp(slab.disp_rel_mp, x, y, 0.01, x_range[0], x_range[-1], args=(slab.K, slab.M_A))
            except:
                root_array = np.delete(root_array, i, 0)
        else:
            root_array = np.delete(root_array, i, 0)

    list_of_lines.append(line)

    del_list = []

    list_of_shapes.append(np.shape(root_array))

    for j, y in enumerate(root_array[:,1]):
        if np.any(np.isclose(y, line, atol=1e-5) == True):
            del_list.append(j)

    root_array = np.delete(root_array, del_list, 0)

    print(np.shape(root_array))

    if len(list_of_shapes) > 1 and list_of_shapes[-1] == list_of_shapes[-2] and len(root_array)>1:
        i += 1
    else:
        i = 0

    print(i)

#for x, y in zip(root_array[:4,0], root_array[:4,1]):
#
#    print(x, y)
#
#    line = sol.line_trace_sp(slab.disp_rel, x, y, 0.001, x_range[0], x_range[-1], args=(slab.K, slab.M_A))
#
#    list_of_lines.append(line)
#
#    del_list = []
#
#    for i, y in enumerate(root_array[:,1]):
#        if np.any(np.isclose(y, line, atol=1e-5) == True):
#            del_list.append(i)
#
#    root_array = np.delete(root_array, del_list, 0)
#
#    print(np.shape(root_array))

####################################################################################################

if True:
    font = {'size': 15}
    matplotlib.rc('font', **font)

    plt.figure(num=None, figsize=(8, 11), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.subplot()

    ax.plot(root_array[:,0], np.real(root_array[:,1]), '.', color = 'b')
    ax.plot(root_array[:,0], np.imag(root_array[:,1]), '.', color = 'r')

    ax.set_xlim(x_range[0], x_range[-1])
    ax.set_ylim(y_range[0], y_range[-1])

    for line in list_of_lines:
        ax.plot(line[:,0], line[:,1], color = 'b', linewidth = 1, linestyle='-')