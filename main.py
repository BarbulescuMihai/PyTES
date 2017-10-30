from functools import partial
import numpy as np
import scipy.optimize as sp
import mpmath as mp
import timeit
from numbers import Number

def grid_solver(func,
                x_range, y_range,
                args=(None),
                method='sp.newton',
                debug=True):

    points = []

    if debug:
        start_time = timeit.default_timer()

    if method == 'sp.newton':

        for x_loc in x_range:

            if type(args)==tuple:
                arguments = [x_loc if i is None else i for i in np.array(args)]
            elif isinstance(x_range[2], Number):
                arguments = [x_loc]
            else:
                print('Invalid arguments.\
                      \nWas expecting number or tuple.\
                      \nInstead received {}'.format(type(args)))
                break

            for y_loc in y_range:
                try:
                    root = sp.newton(func, y_loc, args=tuple(arguments), tol=1e-20)
                    if (root > y_range[0] and root < y_range[-1]) \
                    and not np.isclose(root, points, atol=1e-6).any():
                        if np.imag(root) != 0 and np.imag(root) < 1e-5:
                            root = np.real(root)
                        points.append([x_loc, root])
                except RuntimeError:
                    pass

    if method == 'sp.brentq':

        """Half the stepsize used in y_range."""
        step_size = np.abs(y_range[0]-y_range[-1])/(2*len(y_range))

        for x_loc in x_range:

            if type(args)==tuple:
                arguments = [x_loc if i is None else i for i in np.array(args)]
            elif isinstance(x_range[2], Number):
                arguments = [x_loc]
            else:
                print('Invalid arguments.\
                      \nWas expecting number or tuple.\
                      \nInstead received {}'.format(type(args)))
                break

            for y_loc in y_range:

                y_range_local = np.linspace(y_loc-step_size, y_loc+step_size, 10000)

                flip_func = partial(lambda *args: func(*args[::-1]), *arguments[::-1])

                func_grid = flip_func(y_range_local)

                grid_shift_up = func_grid[1:]
                grid_shift_down = func_grid[:-1]
                grid_prod = np.real(grid_shift_down) * np.real(grid_shift_up)
                root_locs = (grid_prod < 0) * (np.abs(func_grid[:-1]) < 1)

                for i in np.where(root_locs == True)[0]:
                    brent_start = y_range_local[i-2]
                    brent_end = y_range_local[i+2]

                    try:
                        root = sp.brentq(func, brent_start, brent_end, args=tuple(arguments),
                                         xtol=1e-5, rtol=1e-5, maxiter=200)
                        if (root > y_range[0] and root < y_range[-1]) and not np.isclose(root, points, atol=1e-6).any():
                            if np.imag(root) != 0 and np.imag(root) < 1e-5:
                                root = np.real(root)
                            points.append([x_loc, root])

                    except (RuntimeError, ValueError):
                        pass

    if method == 'sp.brenth':

        step_size = np.abs(y_range[0]-y_range[-1])/(2*len(y_range))

        for x_loc in x_range:

            if type(args)==tuple:
                arguments = [x_loc if i is None else i for i in np.array(args)]
            elif isinstance(x_range[2], Number):
                arguments = [x_loc]
            else:
                print('Invalid arguments.\
                      \nWas expecting number or tuple.\
                      \nInstead received {}'.format(type(args)))
                break

            for y_loc in y_range:

                y_range_local = np.linspace(y_loc-step_size, y_loc+step_size, 10000)

                flip_func = partial(lambda *args: func(*args[::-1]), *arguments[::-1])

                func_grid = flip_func(y_range_local)

                grid_shift_up = func_grid[1:]
                grid_shift_down = func_grid[:-1]
                grid_prod = np.real(grid_shift_down) * np.real(grid_shift_up)
                root_locs = (grid_prod < 0) * (np.abs(func_grid[:-1]) < 1)

                for i in np.where(root_locs == True)[0]:
                    brent_start = y_range_local[i-2]
                    brent_end = y_range_local[i+2]

                    try:
                        root = sp.brenth(func, brent_start, brent_end, args=tuple(arguments),
                                         xtol=1e-5, rtol=1e-5, maxiter=200)
                        if (root > y_range[0] and root < y_range[-1]) and not np.isclose(root, points, atol=1e-6).any():
                            if np.imag(root) != 0 and np.imag(root) < 1e-5:
                                root = np.real(root)
                            points.append([x_loc, root])

                    except (RuntimeError, ValueError):
                        pass

    points = np.array(points)

    if debug:

        end_time = timeit.default_timer()
        running_time = end_time - start_time

        print('Grid solver finished running.\
              \nUsing the {} method, the total running time was {:.6f}s.\
              \nA total of {} real roots, and {} complex roots were found in the {}x{} grid.'
              .format(method,
                      running_time,
                      np.count_nonzero(np.isreal(points[:,1])),
                      np.count_nonzero(np.isreal(points[:,1])==False),
                      len(x_range), len(y_range)
                      )
              )

    return points

def find_sign_change(func, x_range, y_range, args=(None)):

    """"Not really functional. Needs argument system implemented."""

    x_grid, y_grid = np.meshgrid(x_range, y_range)
    func_grid = func(y_grid, x_grid, args)

    grid_shift_up = func_grid[1:, :]
    grid_shift_down = func_grid[:-1, :]
    grid_prod = np.real(grid_shift_down) * np.real(grid_shift_up)

    root_locs = x_grid[(grid_prod < 0) * (func_grid[:-1,:] < 10)]
    roots = y_grid[(grid_prod < 0) * (func_grid[:-1,:] < 10)]

    return root_locs, roots

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

def next_root(func, x_loc, y_loc, step_size, arguments, points,
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
            points, step_size, x_error, x_loc = next_root(func, x_loc, y_loc, -step_size,
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
                points, step_size, x_error, x_loc = next_root_mp(flip_func, x_loc, y_loc, step_size,
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
            points, step_size, x_error, x_loc = next_root(func, x_loc, y_loc, step_size,
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

def next_root_mp(func, x_loc, y_loc, step_size, points,
                 solver='halley', wordy=True, tol=1e-15):

    jump_limit = 0.1
    maxsteps = 200

    try:
        grad = (
                (points[-1, 1] - points[-2, 1]) +
                (points[-1, 1] - 2*points[-2, 1] + points[-3, 1])) * \
                np.abs(step_size/(points[-1, 0] - points[-2, 0])
                )
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