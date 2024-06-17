'''
Flash, T., & Hogan, N. (1985).
The coordination of arm movements: an experimentally confirmed mathematical model.
Journal of neuroscience, 5(7), 1688-1703.
https://www.jneurosci.org/content/jneuro/5/7/1688.full.pdf
'''


import numpy as np
import matplotlib.pyplot as plt
from findiff import FinDiff


def f_tau(tau):
    return 15 * tau ** 4 - 6 * tau ** 5 - 10 * tau ** 3


def x_st(xi, xf, tau):
    return xi + (xi - xf) * f_tau(tau)


def y_st(yi, yf, tau):
    return yi + (yi - yf) * f_tau(tau)


if __name__ == "__main__":

    tf_ = 1.0
    step_size = 0.01
    time = np.arange(0, tf_, step_size)

    xi_ = 0
    yi_ = 0

    xf_ = 1.1
    yf_ = 0.9

    x_hand = []
    y_hand = []

    for t in time:
        tau_ = t / tf_
        x_hand = np.append(x_hand, x_st(xi_, xf_, tau_))
        y_hand = np.append(y_hand, y_st(yi_, yf_, tau_))

    # Velocity
    d_dt = FinDiff(0, step_size, 1, acc=6)
    dx_dt = d_dt(x_hand)
    dy_dt = d_dt(y_hand)

    # Acceleration
    d2_dt2 = FinDiff(0, step_size, 2, acc=6)
    d2x_dt2 = d2_dt2(x_hand)
    d2y_dt2 = d2_dt2(y_hand)

    # Jerk
    d3_dt3 = FinDiff(0, step_size, 3, acc=6)
    d3x_dt3 = d3_dt3(x_hand)
    d3y_dt3 = d3_dt3(y_hand)

    fig = plt.figure(0)
    gs = fig.add_gridspec(4, 2)

    fig.add_subplot(gs[:, 0])
    plt.grid(True)
    plt.plot(x_hand, y_hand, 'b.')
    plt.plot(xi_, yi_, 'ro')
    plt.plot([xi_, xf_], [yi_, yf_], 'b--', alpha=0.25)
    plt.plot([xi_, -xf_], [yi_, yf_], 'b--', alpha=0.25)
    plt.xlim(-1.1, 1.1)

    fig.add_subplot(gs[0, 1])
    plt.grid(True)
    plt.plot(time, x_hand, '.')
    plt.plot(time, y_hand, '.')

    fig.add_subplot(gs[1, 1])
    plt.grid(True)
    plt.plot(time, dx_dt, '.')
    plt.plot(time, dy_dt, '.')

    fig.add_subplot(gs[2, 1])
    plt.grid(True)
    plt.plot(time, d2x_dt2, '.')
    plt.plot(time, d2y_dt2, '.')

    fig.add_subplot(gs[3, 1])
    plt.grid(True)
    plt.plot(time, d3x_dt3, '.')
    plt.plot(time, d3y_dt3, '.')

    plt.show()