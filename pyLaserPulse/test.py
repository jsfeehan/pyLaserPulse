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
from scipy.signal import savgol_filter
import scipy.constants as const


def FT_grad(axis, arr):  # , order=1):
    return ut.fft(-1j * ut.fftshift(axis) * ut.ifft(arr)).real


g = grid.grid(1040e-9, (300e-9, 2550e-9), 50e-12)
p = pulse.pulse(100e-15, [1, 0], 'Gauss', 40e6, g)

# GIVE USERS THE CHOICE OVER POLYORDER -- SOMETIMES 4 IS BETTER THAN 2, etc.
# ALSO, CARE SHOULD BE TAKEN -- VERY BROAD WAVELENGTH GRIDS ALWAYS PRODUCE
# POOR RESULTS. PERHAPS BEST TO CALCULATE THE COEFFICIENTS FOR NARROW GRIDS
# FIRST.

# smf = pf.PM980_XP(g, 1, 1e-5)
smf = pf.NKT_NL_1050_NEG_1(g, 1, 1e-5, 1e-2)
betas = []
betas.append(smf.beta_2[g.midpoint])
b = savgol_filter(
        smf.beta_2[g.sim_idx], window_length=int(g.points/8), polyorder=2,
        deriv=1, delta=g.dOmega)
for i in range(12):
    betas.append(b[g.sim_idx_midpoint])
    b = savgol_filter(b, window_length=int(g.points/8), polyorder=2, deriv=1, delta=g.dOmega)

print(betas)

beta_2_reconstruction = ut.Taylor_expansion(betas, g.omega)

D_reconstruction = -2 * np.pi * const.c * beta_2_reconstruction / g.lambda_window**2


err = np.mean(np.abs(D_reconstruction[g.sim_idx] - smf.D) / smf.D)
print("ERROR: ", err)

fig = plt.figure()
ax = fig.add_subplot(111)
# ax.plot(g.lambda_window_crop*1e9, smf.beta_2[g.sim_idx], c='k', lw=2)
ax.plot(g.lambda_window_crop*1e9, 1e6 * smf.D, c='k', lw=2)
# ax.plot(g.lambda_window_crop*1e9, beta_2_reconstruction[g.sim_idx], c='darkorange', ls='--')
ax.plot(g.lambda_window_crop*1e9, 1e6 * D_reconstruction[g.sim_idx], c='darkorange', ls='--')
# ax.set_ylim([smf.beta_2.min(), smf.beta_2.max()])
ax.set_ylim([smf.D.min()*1e6, smf.D.max()*1e6])
plt.show()
