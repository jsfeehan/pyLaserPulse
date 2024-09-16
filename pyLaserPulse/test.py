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



import pyLaserPulse.grid as grid
import pyLaserPulse.pulse as pulse
import pyLaserPulse.catalogue_components as cc
import pyLaserPulse.optical_assemblies as oa
import pyLaserPulse.single_plot_window as spw

if __name__ == "__main__":
    g = grid.grid(2**14, 1040e-9, 2000e-9)
    p = pulse.pulse(100e-15, [200000, 0], 'Gauss', 40e6, g)

    pcf = cc.passive_fibres.NKT_NL_1050_NEG_1(g, 0.25, 1e-5, -1e-3)

    # REQUIRES FUNCTION THAT RETURNS THE DISPERSION CURVE.

    polyFitCo = np.polyfit(g.omega, pcf.beta_2, 11)
    print(polyFitCo[::-1])

    polyFitCo = polyFitCo[::-1]
    tc = np.zeros((len(polyFitCo)+2))
    tc[2::] = polyFitCo

    foo = np.zeros((g.points), dtype=np.complex128)
    for i, b in enumerate(polyFitCo):  # tc):
        print(i, b)
        foo += 1j * b * g.omega**(i) / math.factorial(i)

    D = -2*np.pi*const.c * foo.imag / g.lambda_window**2

    ## This dispersion value looks good, now to add birefringence.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(g.lambda_window, D*1e6) 
    plt.show()

