'''
Fligge, N., McIntyre, J., & van der Smagt, P. (2012, June).
Minimum jerk for human catching movements in 3D.
In 2012 4th IEEE RAS & EMBS International Conference on Biomedical Robotics and Biomechatronics (BioRob)
(pp. 581-586). IEEE.
https://mediatum.ub.tum.de/doc/1285786/document.pdf
'''
import numpy as np


def minimum_jerk_3d(x_ic: np.array, y_ic: np.array, z_ic: np.array,
                    x_fc: np.array, y_fc: np.array, z_fc: np.array,
                    d: float, t: np.array) -> np.array:
    x = _a0(x_ic) + _a1(x_ic) * t + _a2(x_ic) * t**2 + _a3(x_ic, x_fc, d) * t**3 \
        + _a4(x_ic, x_fc, d) * t**4 + _a5(x_ic, x_fc, d) * t**5
    y = _a0(y_ic) + _a1(y_ic) * t + _a2(y_ic) * t**2 + _a3(y_ic, y_fc, d) * t**3 \
        + _a4(y_ic, y_fc, d) * t**4 + _a5(y_ic, y_fc, d) * t**5
    z = _a0(z_ic) + _a1(z_ic) * t + _a2(z_ic) * t**2 + _a3(z_ic, z_fc, d) * t**3 \
        + _a4(z_ic, z_fc, d) * t**4 + _a5(z_ic, z_fc, d) * t**5
    return x, y, z


def _a0(x_ic):
    return x_ic[0]


def _a1(x_ic):
    return x_ic[1]


def _a2(x_ic):
    return 0.5 * x_ic[2]


def _a3(x_ic, x_fc, d):
    return (-10/d**3) * x_ic[0] + (-6/d**2) * x_ic[1] + (-3/(2*d)) * x_ic[2] \
           + (10/d**3) * x_fc[0] + (-4/d**2) * x_fc[1] + (1/(2*d)) * x_fc[2]


def _a4(x_ic, x_fc, d):
    return (15/d**4) * x_ic[0] + (8/d**3) * x_ic[1] + (3/(2*d**2)) * x_ic[2] \
           + (-15/d**4) * x_fc[0] + (7/d**3) * x_fc[1] + (-1/d**2) * x_fc[2]


def _a5(x_ic, x_fc, d):
    return (-6/d**5) * x_ic[0] + (-3/d**4) * x_ic[1] + (-1/(2*d**3)) * x_ic[2] \
           + (6/d**5) * x_fc[0] + (-3/d**4) * x_fc[1] + (1/(2*d**3)) * x_fc[2]


if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from findiff import FinDiff

    step = 1 / 90
    duration = np.random.uniform(1, 1.5)  # Random duration of the movement phase
    t_mp = np.arange(0, 1 + step, step)  # Time of the movement phase

    x_init = np.array([0, 0, 0])
    y_init = np.array([0, 0, 0])
    z_init = np.array([0, 0, 0])

    # Assuming that final velocities are not zero to create a curved trajectory

    x_fin = np.array([1, -0.5, 0])
    y_fin = np.array([1, 2, 0])
    z_fin = np.array([1, -0.25, 0])

    # Movement phase
    x_tr, y_tr, z_tr = minimum_jerk_3d(x_init, y_init, z_init, x_fin, y_fin, z_fin, duration, t_mp)

    # Add Static phase
    to_real = np.random.uniform(0.3, 0.9)  # Random movement onset time
    t_tr = np.arange(0, 1 + to_real + step, step)
    x_tr = np.append(np.full(t_tr.size - x_tr.size, x_init[0]), x_tr)
    y_tr = np.append(np.full(t_tr.size - y_tr.size, y_init[0]), y_tr)
    z_tr = np.append(np.full(t_tr.size - z_tr.size, z_init[0]), z_tr)

    # print(t_tr.size, x_tr.size)

    # Findiff can be used to calculate velocities because synthetic data is not noisy
    d_dt = FinDiff(0, step, 1, acc=6)
    vx = d_dt(x_tr)
    vy = d_dt(y_tr)
    vz = d_dt(z_tr)

    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 3)
    ax = fig.add_subplot(gs[0, 0], projection='3d')
    ax.plot(x_tr, y_tr, z_tr, '.')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax = fig.add_subplot(gs[0, 1])
    ax.grid(True)
    ax.plot(t_tr, x_tr, '.', label='x')
    ax.plot(t_tr, y_tr, '.', label='y')
    ax.plot(t_tr, z_tr, '.', label='z')
    ax.axvline(to_real, ls='--', color='g', label='to')
    ax.legend()
    ax.set_xlabel("t")

    ax = fig.add_subplot(gs[0, 2])
    ax.grid(True)
    ax.plot(t_tr, vx, '.', label='vx')
    ax.plot(t_tr, vy, '.', label='vy')
    ax.plot(t_tr, vz, '.', label='vz')
    ax.axvline(to_real, ls='--', color='g', label='to')
    ax.legend()
    ax.set_xlabel("t")

    plt.tight_layout()
    plt.show()