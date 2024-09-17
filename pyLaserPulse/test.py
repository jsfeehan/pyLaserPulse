import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.constants as const
from scipy.interpolate import approximate_taylor_polynomial



# x = np.linspace(-10, 10, 1024)
# y = np.sin(x)

# tc = approximate_taylor_polynomial(np.sin, 0, 13, 1, order=15)

# print(tc)

# # tc = tc[::-1]
# # _y = np.zeros_like(x)
# # for i, v in enumerate(tc):
# #     _y += v**(i) * x / math.factorial(i)

# _y = tc(x)


# plt.plot(x, y)
# plt.plot(x, _y, ls='--')
# plt.show()



# Taylors from "Supercontinuum generation in photonic crystal fiber"
# [ps**n / km] -- for 800 nm
# tc = [0, 0, -11.83, 8.1038e-2, -9.5205e-5, 2.0737e-7, -5.3943e-10,
#       1.3486e-12, -2.5495e-15, 3.0524e-18, -1.714e-21]
# tc = [1e-3*coeff*1e-12**(i) for i, coeff in enumerate(tc)]


import pyLaserPulse.grid as grid
import pyLaserPulse.pulse as pulse
import pyLaserPulse.catalogue_components as cc
import pyLaserPulse.base_components as bc
import pyLaserPulse.optical_assemblies as oa
import pyLaserPulse.single_plot_window as spw
import pyLaserPulse.data.paths as paths

if __name__ == "__main__":
    g = grid.grid(2**14, 1040e-9, 2000e-9)
    # p = pulse.pulse(100e-15, [200000, 0], 'Gauss', 40e6, g)

    pcf = cc.passive_fibres.NKT_NL_1050_NEG_1(g, 0.25, 1e-5, -1e-3)

    polyFitCo = np.polyfit(g.omega, pcf.beta_2, 11)
    print(polyFitCo[::-1])

    polyFitCo = polyFitCo[::-1]
    tc = np.zeros((len(polyFitCo)+2))
    tc[2::] = polyFitCo

    foo = np.zeros((g.points), dtype=np.complex128)
    for i, b in enumerate(tc):
        print(i, b)
        foo += 1j * b * g.omega**i / math.factorial(i)

    foo = np.gradient(foo, g.omega, edge_order=2)
    foo = np.gradient(foo, g.omega, edge_order=2)
    D = -2e6*np.pi*const.c * foo.imag / g.lambda_window**2

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(g.lambda_window, D)  # foo.imag)
    ax.plot(g.lambda_window, 1e6*pcf.D)
    plt.show()

