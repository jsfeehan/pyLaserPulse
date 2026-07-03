"""
Testing script for new Taylor calculation method

Found a method based on real parts of FFT. Faster and more stable than other
methods that I have found.

The purpose of the Taylor coefficients is NOT to replace beta_2 in the
propagation calculations, but instead for doing secondary calculations based
on system dispersion, such as determining a cFBG dispersion profile which gives
a high degree of compressibility in CPA, for example.

James Feehan, 2/7/2026
"""


import pyLaserPulse.grid as grid
import pyLaserPulse.pulse as pulse
import pyLaserPulse.base_components as bc
import pyLaserPulse.catalogue_components.passive_fibres as pf
import pyLaserPulse.data as data
import pyLaserPulse.utils as ut


import matplotlib.pyplot as plt
import numpy as np
from math import factorial
from scipy.signal import savgol_filter


def FT_grad(axis, arr):  # , order=1):
    # return ut.fft((-1j * ut.fftshift(axis))**order * ut.ifft(arr)).real
    return ut.fft(-1j * ut.fftshift(axis) * ut.ifft(arr)).real


# g = grid.grid(800e-9, (500e-9, 1500e-9), 10e-12)
g = grid.grid(1040e-9, (1000e-9, 1300e-9), 10e-12)
p = pulse.pulse(100e-15, [1, 0], 'Gauss', 40e6, g)


smf = pf.PM980_XP(g, 1, 1e-5)
betas = []
betas.append(smf.beta_2[g.midpoint])
# b = FT_grad(g.time_window, smf.beta_2)
b = savgol_filter(smf.beta_2, window_length=110, polyorder=2, deriv=1, delta=g.dOmega)
# b = np.gradient(smf.beta_2, g.dOmega, edge_order=2)
for i in range(10):
    betas.append(b[int(len(b)/2)])
    b = savgol_filter(b, window_length=110, polyorder=2, deriv=1, delta=g.dOmega)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(b)
    plt.show()
    print(b[int(len(b)/2)])
    # betas.append(FT_grad(g.time_window, smf.beta_2, order=i)[g.midpoint])
    # betas.append(FT_grad(g.time_window, smf.beta_2, order=i)[g.midpoint])




# betas.append(smf.beta_2[g.midpoint])
# # print(betas)
# b3 = np.gradient(smf.beta_2, g.dOmega)[g.midpoint]
# # print(b3)
# betas.append(b3)
# b4 = np.gradient(np.gradient(smf.beta_2, g.dOmega), g.dOmega)[g.midpoint]
# betas.append(b4)

# betas = []
# betas.append(smf.beta_2[g.midpoint])  # beta 2 added OUT of loop
# b = FT_grad(g.time_window, smf.beta_2)
# for i in range(1):
#     betas.append(np.abs(b[g.midpoint]))
#     b = FT_grad(g.time_window, b)



print(betas)


beta_2_reconstruction = np.zeros((g.points))
for i, TC in enumerate(betas):
    print(i, TC)
    beta_2_reconstruction += TC * g.omega**(i) / factorial(i)

# beta_2_reconstruction = ut.Taylor_expansion(betas, g.omega)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(g.lambda_window*1e9, smf.beta_2, c='k', lw=2)
ax.plot(g.lambda_window*1e9, beta_2_reconstruction, c='darkorange', ls='--')
ax.set_ylim([smf.beta_2.min(), smf.beta_2.max()])
plt.show()

# # smf.get_Taylors(5)

# Taylors = [-11.83 * 1e-12**2 / 1e3,
#            8.1038e-2 * 1e-12**3 / 1e3,
#            -9.5205e-5 * 1e-12**4 / 1e3,
#            2.0737e-7 * 1e-12**5 / 1e3,
#            -5.3943e-10 * 1e-12**6 / 1e3,
#            1.3486e-12 * 1e-12**7 / 1e3,
#            -2.5494e-15 * 1e-12**8 / 1e3,
#            3.0524e-18 * 1e-12**9 / 1e3,
#            -1.714e-21 * 1e-12**10 / 1e3]

# beta_2 = ut.Taylor_expansion(Taylors, g.omega_crop)
# import scipy.constants as const
# D = -2 * np.pi * const.c  * beta_2 / g.lambda_window_crop**2


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(g.lambda_window_crop, D* 1e6)  # beta_2 / (1e-24 / 1e3))
# plt.show()


# COME BACK TO THE COMPRESSOR LATER -- START WITH FIBERS INITIALLY

# ##############
# # Compressor #
# ##############
# loss = 0.04           # percent loss per grating reflection
# transmission = 20e-9  # transmission bandwidth
# coating = data.paths.materials.reflectivities.gold
# epsilon = 1e-1         # Jones parameter for polarization mixing and phase
# theta = 0              # Jones parameter for angle subtended by x-axis
# beamsplitting = 0      # Useful for output couplers, etc.
# crosstalk = 1e-5       # polarization mixing -- very low
# l_mm = 1200            # grating lines per mm
# sep_initial = 90e-2    # initial guess for grating separation
# angle_initial = 0.7    # initial guess for incidence angle, rad
# gc = bc.grating_compressor(
#     loss, transmission, coating, g.lambda_c, epsilon, theta, beamsplitting,
#     crosstalk, sep_initial, angle_initial, l_mm, g, order=5, optimize=False)


# gc.get_Taylors(2)

# # approx_disp = ut.Taylor_expansion(gc.Taylor_coefficients, g.omega_crop)
# approx_disp = np.zeros_like((g.omega_crop))

# for i, tc in enumerate(gc.Taylor_coefficients):
#     approx_disp += tc * (g.omega_crop + g.omega_c) / factorial(i)

# print(gc.beta_2.shape, g.lambda_window_crop.shape)

# fig = plt.figure()
# ax=  fig.add_subplot(111)
# ax.plot(g.lambda_window_crop, gc.beta_2, c='k')
# ax.plot(g.lambda_window_crop, approx_disp)
# # ax.set_ylim([np.amin(gc.beta_2), np.amax(gc.beta_2)])
# plt.show()