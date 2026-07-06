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
import pyLaserPulse.catalogue_components.passive_fibres as pf
import pyLaserPulse.utils as ut


import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const


g = grid.grid(1030e-9, (700e-9, 1550e-9), 20e-12)

# smf = pf.PM980_XP(g, 1, 1e-5)
smf = pf.NKT_NL_1050_NEG_1(g, 1, 1e-5, 1e-2)
# smf = pf.NKT_SC_5_1040_PM(g, 1, 1e-5)

print(g.omega_crop.min(), g.omega_crop.max())
betas = ut.Maclaurin_coefficients(
    smf.beta_2, g.omega, g.dOmega, g, 15,
    (g.omega_crop.min(), g.omega_crop.max()), int(g.points / 8), 2) 

print(betas[0], betas[1])
betas = betas[0]

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
