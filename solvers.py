#import Toolbox_extended as t
#import sys
from functools import partial
import numpy as np
import scipy.optimize as sp
import mpmath as mp
#import time

# Attempts to find the roots of a defined function in a box defined by two
# lists defined by numpy.linspace, using scipy.optimize.newton.
# Arguments:
# func - The function to be solved. This is must be a function of AT LEAST
# two variables, but it may have any number.
# x_range, y_range - (numpy.linspace) Lists defining the box.
# args - The values of the arguments of func. If this is omitted, the point
# finder assumes that func is only dependent on two variables, i.e. that
# it is of the form f(a,b). The point finder will iterate over 'b' and find
# the values of 'a' for which f(a,b) = 0. The point finder will ALWAYS find the
# roots of the FIRST variable in func. If the function depends on more than two
# variables, the variables must be imputed IN ORDER. This means that if you
# have a function f(a, b, c), the point finder will always use its root solver
# on 'a', and depending on how you input 'b' and 'c', it will keep one constant
# and iterate over the other. If, for example, your input is args=(None, 1),
# the point finder will find the values of 'a' for each value of 'b' in x_range
# around the points in y_range, while keeping 'c' constant, with c=1.
# If, on the other hand, your input is args=(1, None), it will find 'a', while
# keeping 'b' constant, with b=1, and iterate over 'c', where 'c' will take the
# values in y_range.

# Example:
# Suppose you have a function f(a, b, c, d) = 0.
# The point finder will attempt to use Newton's method to find values of the
# variable 'a' such that f(a) = 0 for some set values of 'b', 'c', and 'd',
# where one of these is iterated over.
# Say you want b=1, c to be iterated over, and d=3. You would input this as
# y_pts = point_finder_scipy(func, x_range, y_range, args=(1, None, 3))
# Your output will be a numpy array called y_pts.
# You may then use plot(x_range, y_pts).

def point_finder_sp(func, x_range, y_range, args=(None)):
    points = []
    for x_loc in x_range:
        arguments = [x_loc if i is None else i for i in list(args)]
        for y_loc in y_range:
            try:
                root = sp.newton(func, y_loc, args=tuple(arguments), tol=1e-20)
                if (root > y_range[0] and root < y_range[-1]) and not np.isclose(root, points, atol=1e-6).any():
                    if np.imag(root) != 0 and np.imag(root) < 1e-5:
                        root = np.real(root)
                    points.append([x_loc, root])
            except RuntimeError:
                pass
    return np.array(points)

def find_first_imag(func, x_range, y_range, args=(None)):
    for x_loc in x_range:
        arguments = [x_loc if i is None else i for i in list(args)]
        for y_loc in y_range:
            try:
                root = sp.newton(func, y_loc, args=tuple(arguments), tol=1e-20)
                if (root > y_range[0] and root < y_range[-1]) and np.imag(root) > 1e-5:
                    return np.array([x_loc, root])
            except RuntimeError:
                pass

def point_finder_mp(func, x_range, y_range, args=None, solver='muller'):
    points = []
    for x_loc in x_range:
        arguments = [x_loc if i is None else i for i in list(args)]
        flip_func = partial(lambda *args: func(*args[::-1]), *arguments[::-1])
        for y_loc in y_range:
            if solver == 'muller':
                y_loc = [y_loc - 1e-10, y_loc, y_loc + 1e-10]
            try:
                root = mp.findroot(flip_func, y_loc, solver, tol=1e-20)
                root = float(mp.re(root)) + float(mp.im(root)) * 1j
                if not np.isclose(root, points, atol=1e-6).any():
                    points.append([x_loc, root])
            except (RuntimeError, ZeroDivisionError, ValueError):
                pass
    return np.array(points)

# Attempts to find a root that is close to a root inputted into the finder.
# Arguments:
# func - The function. Must depend on a single variable.
# root_list - List of roots of the function found so far. Can be empty.
# x - The x coordinate of the starting point.
# y - The y coordinate of the starting point.
# step_size - The step size to be used in calculating the next point.
# x_values -

def root_find_sp(func, x_loc, y_loc, step_size, arguments, points,
                 wordy=True):

    jump_limit = 0.1
    iterations = 200

    try:
        grad = ((points[-1, 1] - points[-2, 1]) + (points[-1, 1] - \
                2 * points[-2, 1] + points[-3, 1])) * \
                np.abs(step_size/(points[-1, 0] - points[-2, 0]))
        root = sp.newton(func, points[-1, 1] + grad + 1e-20 * 1j,
                         args=arguments, maxiter=iterations)
        if np.abs(root - points[-1, 1]) < jump_limit and\
        np.abs(x_loc - points[-1, 0]) < jump_limit:
            points = np.vstack([points, [x_loc, root]])
        else:
            raise ValueError('Jump of {:.5f} at x = {:.5f}, y = {:.5f}'.format(\
                             np.abs(root-points[-1, 1]), points[-1, 0], points[-1, 1]+grad))
        x_error = None

    except IndexError:
        if points.all() == 0:
            points[0,] = x_loc, sp.newton(func, y_loc, args=arguments, maxiter=iterations)
        elif points.shape == (1, 2):
            root = sp.newton(func, points[-1, 1], args=arguments, maxiter=iterations)
            points = np.vstack([points, [x_loc, root]])
        elif points.shape == (2, 2):
            grad = ((points[-1, 1] - points[-2, 1]) * \
                    np.abs(step_size/(points[-1, 0] - points[-2, 0])))
            root = sp.newton(func, points[-1, 1] + grad + 1e-20 * 1j,
                             args=arguments, maxiter=iterations)
            if np.abs(root - points[-1, 1]) < jump_limit and\
            np.abs(x_loc - points[-1, 0]) < jump_limit:
                points = np.vstack([points, [x_loc, root]])
            else:
                raise ValueError('Jump of {:.5f} at x = {:.5f}, y = {:.5f}'
                                 .format(np.abs(root-points[-1, 1]), points[-1, 0],
                                         points[-1, 1]+grad))
        x_error = None

    return points, np.abs(step_size), x_error, x_loc

def line_trace_sp(func, x_loc, y_loc, step_size, x_end_left, x_end_right,
                  args=None, wordy=True, func_mp=None, solver='halley'):

    points = np.zeros((1, 2))
    step_init = step_size
    x_error_loc = None

    while np.real(x_loc) >= x_end_left:
        arguments = [x_loc if i is None else i for i in list(args)]
        flip_func = partial(lambda *args: func_mp(*args[::-1]), *arguments[::-1])

        try:
            points, step_size, x_error, x_loc = root_find_sp(func, x_loc, y_loc, -step_size,
                                                             tuple(arguments), points, wordy)

        except (RuntimeError, ValueError) as err:
            if step_size >= step_init * 2**(-5):
                if wordy:
                    print('Error when solving for x = {:.5f}, y = {}.\n Error message:{}'
                          .format(x_loc, y_loc, err))
                x_loc -= step_size
                x_error, step_size = x_loc, step_size/2
                if wordy:
                    print('Error when solving for x = {:.5f}, y = {}.\n Error message:{}'
                          .format(x_loc, y_loc, err))
            else:
                points, step_size, x_error, x_loc = root_find_mp(flip_func, x_loc, y_loc, step_size,
                                                                 points, solver, wordy)

        if x_error is None and x_error_loc is None:
            pass
        elif not x_error is None and x_error_loc is None:
            x_error_loc = x_error
        elif x_error is None and not x_error_loc is None and\
            np.abs(x_loc) - np.abs(x_error_loc) >= 10 * step_size:
            step_size = step_init
        else:
            pass

        x_loc -= step_size

    points = points[::-1]
    x_loc = points[-1, 0] + step_size

    while np.real(x_loc) <= x_end_right:
        arguments = [x_loc if i is None else i for i in list(args)]
        flip_func = partial(lambda *args: func_mp(*args[::-1]), *arguments[::-1])

        try:
            points, step_size, x_error, x_loc = root_find_sp(func, x_loc, y_loc, step_size,
                                                             tuple(arguments), points, wordy)
        except (RuntimeError, ValueError) as err:
            if wordy:
                print('Error when solving for x = {:.5f}.\n Error message:{}'
                      .format(x_loc, err))
            x_loc -= step_size
            x_error, step_size = x_loc, step_size/2
            if wordy:
                print('Solving for x = {:.5f}, and step_size = {} instead. \n'
                      .format(x_loc, step_size))

        if x_error is None and x_error_loc is None:
            pass
        elif not x_error is None and x_error_loc is None:
            x_error_loc = x_error
        elif x_error is None and not x_error_loc is None and\
            np.abs(x_loc) - np.abs(x_error_loc) >= 10 * step_size:
            step_size = step_init
        else:
            pass

        x_loc += step_size
    points[:, 0] = np.round(points[:, 0], 4)
    return points

# Attempts to trace a line of roots of a function.
# Arguments:
# func - The function. This can have any number of variables,
# but the first one is always the one that the root finder will be used on.
# x - The x coordinate of the starting point.
# y - The y coordinate of the starting point.
# step_size - The step size to be used in calculating the next point.
# x_end_left, x_end_right - These define the limits of the interval in which
# line is to be traced.

def root_find_mp(func, x_loc, y_loc, step_size, points,
                 solver='halley', wordy=True, tol=1e-15):

    jump_limit = 0.1
    maxsteps = 200

    try:
        grad = ((points[-1, 1] - points[-2, 1]) +
                (points[-1, 1] - 2*points[-2, 1] + points[-3, 1])) * \
                np.abs(step_size/(points[-1, 0] - points[-2, 0]))
        root_mp = mp.findroot(func, points[-1, 1] + grad, solver, tol)
        root = float(mp.re(root_mp)) + 1j * float(mp.im(root_mp))
        if np.abs(root-points[-1, 1]) < 0.1:
            points = np.vstack([points, [x_loc, root]])
        else:
            raise ValueError('Jump of {:.5f} at x = {:.5f}, y = {:.5f}'.format(
                np.abs(root-points[-1, 1]), points[-1, 0], points[-1, 1]+grad))
        x_error = None

    except IndexError:
        if points.all() == 0:
            root_mp = mp.findroot(func, y_loc, solver, tol, maxsteps)
            root = float(mp.re(root_mp)) + 1j*float(mp.im(root_mp))
            points[0,] = x_loc, root
        elif points.shape == (1, 2):
            root_mp = mp.findroot(func, points[-1, 1], solver, tol, maxsteps)
            root = float(mp.re(root_mp)) + 1j*float(mp.im(root_mp))
            points = np.vstack([points, [x_loc, root]])
        elif points.shape == (2, 2):
            grad = (points[-1, 1] - points[-2, 1]) *\
                    np.abs(step_size/(points[-1, 0] - points[-2, 0]))
            root_mp = mp.findroot(func, points[-1, 1] + grad, solver, tol, maxsteps)
            root = float(mp.re(root_mp)) + 1j*float(mp.im(root_mp))
            if np.abs(root - points[-1, 1]) < jump_limit:
                points = np.vstack([points, [x_loc, root]])
            else:
                raise ValueError('Jump of {:.5f} at x = {:.5f}, y = {:.5f}'.format(
                    np.abs(root-points[-1, 1]), points[-1, 0], points[-1, 1]+grad))
        x_error = None

    return points, np.abs(step_size), x_error, x_loc

def line_trace_mp(func, x_loc, y_loc, step_size, x_end_left, x_end_right,
                  wordy=False, args=None, solver='halley'):

    points = np.zeros((1, 2), dtype=np.complex128)
    step_init = step_size
    x_error_loc = None

    while x_loc > x_end_left:
        arguments = [x_loc if i is None else i for i in list(args)]
        flip_func = partial(lambda *args: func(*args[::-1]), *arguments[::-1])

        try:
            points, step_size, x_error, x_loc = root_find_mp(flip_func, x_loc, y_loc, -step_size,
                                                             points, solver, wordy)
        except (RuntimeError, ValueError) as err:
            if wordy:
                print('Error when solving for x = {:.5f}.\n Error message:{}'
                      .format(x_loc, err))
            x_loc -= step_size
            x_error, step_size = x_loc, step_size/2
            if wordy:
                print('Solving for x = {:.5f}, and step_size = {} instead. \n'
                      .format(x_loc, step_size))

        if x_error is None and x_error_loc is None:
            pass
        elif not x_error is None and x_error_loc is None:
            x_error_loc = x_error
        elif x_error is None and not x_error_loc is None and\
            np.abs(x_loc) - np.abs(x_error_loc) >= 10 * step_size:
            step_size = step_init
        else:
            pass

        x_loc -= step_size

    points = points[::-1]
    x_loc = points[-1, 0] + step_size

    while x_loc <= x_end_right:
        arguments = [x_loc if i is None else i for i in list(args)]
        flip_func = partial(lambda *args: func(*args[::-1]), *arguments[::-1])

        try:
            points, step_size, x_error, x_loc = root_find_mp(flip_func, x_loc, y_loc, -step_size,
                                                             points, solver, wordy)
        except (RuntimeError, ValueError) as err:
            if wordy:
                print('Error when solving for x = {:.5f}.\n Error message:{}'
                      .format(x_loc, err))
            x_loc -= step_size
            x_error, step_size = x_loc, step_size/2
            if wordy:
                print('Solving for x = {:.5f}, and step_size = {} instead. \n'
                      .format(x_loc, step_size))

        if x_error is None and x_error_loc is None:
            pass
        elif not x_error is None and x_error_loc is None:
            x_error_loc = x_error
        elif x_error is None and not x_error_loc is None and\
            np.abs(x_loc) - np.abs(x_error_loc) >= 10 * step_size:
            step_size = step_init
        else:
            pass

        x_loc += step_size

    points[:, 0] = np.round(points[:, 0], 4)
    return points