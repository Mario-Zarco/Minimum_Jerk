'''
Flash, T., & Hogan, N. (1985).
The coordination of arm movements: an experimentally confirmed mathematical model.
Journal of neuroscience, 5(7), 1688-1703.
c1, pi1, c2, pi2 can be found in Appendix C
https://www.jneurosci.org/content/jneuro/5/7/1688.full.pdf
'''

import numpy as np
import matplotlib.pyplot as plt
from findiff import FinDiff


def _c1(x0, x1, xf, tf, tau1):
    c1_1 = (xf - x0) * (300 * tau1**5 - 1200 * tau1**4 + 1600 * tau1**3)
    c1_2 = tau1**2 * (-720 * xf + 120 * x1 + 600 * x0)
    c1_3 = (x0 - x1) * (300 * tau1 - 200)
    f = 1 / (tf**5 * tau1**2 * (1 - tau1)**5)
    c1 = f * (c1_1 + c1_2 + c1_3)
    return c1


def _pi1(x0, x1, xf, tf, tau1):
    pi1_1 = (xf - x0) * (120 * tau1**5 - 300 * tau1**4 + 200 * tau1**3)
    pi1_2 = -20 * (x1 - x0)
    f = 1 / (tf**5 * tau1**5 * (1 - tau1)**5)
    pi1 = f * (pi1_1 + pi1_2)
    return pi1


def _c2(y0, y1, yf, tf, tau1):
    c2_1 = (yf - y0) * (300 * tau1**5 - 1200 * tau1**4 + 1600 * tau1**3)
    c2_2 = tau1**2 * (-720 * yf + 120 * y1 + 600 * y0)
    c2_3 = (y0 - y1) * (300 * tau1 - 200)
    f = 1 / (tf**5 * tau1**2 * (1 - tau1)**5)
    c2 = f * (c2_1 + c2_2 + c2_3)
    return c2


def _pi2(y0, y1, yf, tf, tau1):
    pi2_1 = (yf - y0) * (120 * tau1**5 - 300 * tau1**4 + 200 * tau1**3)
    pi2_2 = -20 * (y1 - y0)
    f = 1 / (tf**5 * tau1**5 * (1 - tau1)**5)
    pi2 = f * (pi2_1 + pi2_2)
    return pi2


'''
Curved point-to-point movements (p. 1691)
x-(tau) 
'''
def x1_ct(x0, tf, tau1, c1, pi1, tau):
    x1_1 = tau1**4 * (15 * tau**4 - 30 * tau**3)
    x1_2 = tau1**3 * (80 * tau**3 - 30 * tau**4)
    x1_3 = -60 * tau**3 * tau1**2 + 30 * tau**4 * tau1 - 6 * tau**5
    x1_4 = 15 * tau**4 - 10 * tau**3 - 6 * tau**5
    f = tf**5 / 720
    x1_tau = f * (pi1 * (x1_1 + x1_2 + x1_3) + c1 * x1_4) + x0
    return x1_tau


'''
Curved point-to-point movements (p. 1691)
x+(tau) 
x2_tau = x1(x0, tf, tau1, c1, pi1, tau) + pi1 * f
'''
def x2_ct(xf, tf, tau1, c1, pi1, tau):
    x2_1 = tau1**4 * (15 * tau**4 - 30 * tau**3 + 30 * tau - 15)
    x2_2 = tau1**3 * (-30 * tau**4 + 80 * tau**3 - 60 * tau**2 + 10)
    x2_3 = -6 * tau**5 + 15 * tau**4 - 10 * tau**3 + 1
    f = tf**5 / 720
    x2_tau = f * (pi1 * (x2_1 + x2_2) + c1 * x2_3) + xf
    return x2_tau


def x_ct(x0, x1, t1, xf, tf, tau1, t):
    c1 = _c1(x0, x1, xf, tf, tau1)
    pi1 = _pi1(x0, x1, xf, tf, tau1)
    tau = t / tf
    if t <= t1:
        return x1_ct(x0, tf, tau1, c1, pi1, tau)
    elif t > t1:
        return x2_ct(xf, tf, tau1, c1, pi1, tau)


def y1_ct(y0, tf, tau1, c2, pi2, tau):
    y1_1 = tau1 ** 4 * (15 * tau ** 4 - 30 * tau ** 3)
    y1_2 = tau1 ** 3 * (80 * tau ** 3 - 30 * tau ** 4)
    y1_3 = -60 * tau ** 3 * tau1 ** 2 + 30 * tau ** 4 * tau1 - 6 * tau ** 5
    y1_4 = 15 * tau ** 4 - 10 * tau ** 3 - 6 * tau ** 5
    f = tf ** 5 / 720
    y1_tau = f * (pi2 * (y1_1 + y1_2 + y1_3) + c2 * y1_4) + y0
    return y1_tau


def y2_ct(yf, tf, tau1, c2, pi2, tau):
    y2_1 = tau1 ** 4 * (15 * tau ** 4 - 30 * tau ** 3 + 30 * tau - 15)
    y2_2 = tau1 ** 3 * (-30 * tau ** 4 + 80 * tau ** 3 - 60 * tau ** 2 + 10)
    y2_3 = -6 * tau ** 5 + 15 * tau ** 4 - 10 * tau ** 3 + 1
    f = tf ** 5 / 720
    y2_tau = f * (pi2 * (y2_1 + y2_2) + c2 * y2_3) + yf
    return y2_tau


def y_ct(y0, y1, t1, yf, tf, tau1, t):
    c1 = _c2(y0, y1, yf, tf, tau1)
    pi1 = _pi2(y0, y1, yf, tf, tau1)
    tau = t / tf
    if t <= t1:
        return y1_ct(y0, tf, tau1, c1, pi1, tau)
    elif t > t1:
        return y2_ct(yf, tf, tau1, c1, pi1, tau)


if __name__ == "__main__":

    tf_ = 1.0
    step_size = 0.01
    time = np.arange(0, tf_, step_size)

    x0_ = 0
    y0_ = 0

    x1_ = -0.1
    y1_ = 0.4
    t1_ = tf_ / 2

    xf_ = 1
    yf_ = 1

    tau1_ = t1_ / tf_
    time1 = np.arange(0, tf_/2 , step_size)

    c1_ = _c1(x0_, x1_, xf_, tf_, tau1_)
    pi1_ = _pi1(x0_, x1_, xf_, tf_, tau1_)

    c2_ = _c2(y0_, y1_, yf_, tf_, tau1_)
    pi2_ = _pi2(y0_, y1_, yf_, tf_, tau1_)

    x_hand = []
    y_hand = []

    for t in time:
        # tau_ = t / tf_
        # xa = x1_ct(x0_, tf_, tau1_, c1_, pi1_, tau_)
        # print(xa)
        x_hand = np.append(x_hand, x_ct(x0_, x1_, t1_, xf_, tf_, tau1_, t))
        # print(x_hand)
        y_hand = np.append(y_hand, y_ct(y0_, y1_, t1_, yf_, tf_, tau1_, t))
        # print(y_hand)

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
    plt.plot(x1_, y1_, 'ro')
    plt.plot([x0_, xf_], [y0_, yf_], 'b--', alpha=0.25)
    plt.plot([x0_, -xf_], [y0_, yf_], 'b--', alpha=0.25)
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



