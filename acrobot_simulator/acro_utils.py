import sys
import os
import numpy as np
import numbers

""" This set of utility functions are adapted from rlpy team."""

# Original attribution information:
__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann"

def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.

    *y0*
        initial state vector

    *t*
        sample times

    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``

    *args*
        additional arguments passed to the derivative function

    *kwargs*
        additional keyword arguments passed to the derivative function

    Example 1 ::

        ## 2D system

        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)

    Example 2::

        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)

        y0 = 1
        yout = rk4(derivs, y0, t)


    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0
    i = 0

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout

def wrap(x, m, M):
    """
    :param x: a scalar
    :param m: minimum possible value in range
    :param M: maximum possible value in range

    Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x

def bound(x, m, M=None):
    """
    :param x: scalar

    Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)

def fromAtoB(x1, y1, x2, y2, color='k', connectionstyle="arc3,rad=-0.4",
             shrinkA=10, shrinkB=10, arrowstyle="fancy", ax=None):
    """
    Draws an arrow from point A=(x1,y1) to point B=(x2,y2) on the (optional)
    axis ``ax``.

    .. note::

        See matplotlib documentation.

    """
    if ax is None:
        return pl.annotate("",
                           xy=(x2, y2), xycoords='data',
                           xytext=(x1, y1), textcoords='data',
                           arrowprops=dict(
                               arrowstyle=arrowstyle,  # linestyle="dashed",
                               color=color,
                               shrinkA=shrinkA, shrinkB=shrinkB,
                               patchA=None,
                               patchB=None,
                               connectionstyle=connectionstyle),
                           )
    else:
        return ax.annotate("",
                           xy=(x2, y2), xycoords='data',
                           xytext=(x1, y1), textcoords='data',
                           arrowprops=dict(
                               arrowstyle=arrowstyle,  # linestyle="dashed",
                               color=color,
                               shrinkA=shrinkA, shrinkB=shrinkB,
                               patchA=None,
                               patchB=None,
                               connectionstyle=connectionstyle),
                           )