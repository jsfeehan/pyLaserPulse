#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrieve the Taylor coefficients for optical fibres and compressors.

A number of examples are shown in this script, including how different choices
of parameters for the simulation grid and for the Taylor calculation grid can
impact the outcome of the Taylor coefficient calculations and expansions, as
as examples for how to retrieve Taylor coefficients for bulk components (where
applicable) such as grating compressors.

Care must be taken when Taylor coefficients are calculated. An inappropriate
choice of domain range, or even altering grid parameters can change the
accuracy of the calculation (this is not avoidable, as far as I know).
Parameters which produce good Taylor coefficients for one component may
produce less good coefficients for a different component. It is recommended
that the Taylor coefficients are always checked against the input data before
being trusted.

Taylor coefficients are NOT used for dispersion calculations in the code
unless input by the user through the component class (or others) for, e.g.,
cFBG definitions (see examples\\fibre_amplifiers\\Yb_fibre_CPA_system.py).
Taylor coefficient retrieval has been added to pyLaserPulse because it can be
a useful thing to have when specifying components such as chirped mirrors, or
getting an idea of the magnitude of dispersion in, for example, a parabolic
amplifier.

The differentiation required to obtain the coefficients is done using a
Savitzky-Golay filter, embedded in the function utils.Maclaurin_coefficients
(Taylor expansion is always done from y=y_0 and still applies to any relevant
grid without error if this is taken into account externally). This filtering
method means that high-order gradients can be calculated without amplifying
noise and without additional constraints (e.g., function tending to zero at
window edges, or function periodicity, both of which need to be considered for
Fourier differentiation, for example). Additional parameters controlling the
Savitzky-Golay window size, the domain limits, and the polynomial order must be
set for this method, and this is best done by trial and error to minimize the
fit error (qualitative assesssment is usually good enough here, which can be
done by overlaying plots of the expanded dispersion curves and the calculated
ones).
"""

from pyLaserPulse import grid
import pyLaserPulse.catalogue_components.passive_fibres as pf
import pyLaserPulse.utils as ut

from decimal import Decimal
import matplotlib.pyplot as plt
import scipy.constants as const
import numpy as np


def plot_comparison(x, y, y_expansion, x_label, y_label, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, c='k', lw=2)
    ax.plot(x, y_expansion, c='darkorange', ls='--')
    ax.legend(['y', 'y_expansion'])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)


# Time-frequency grid parameters and instantiation
central_wl = 1030e-9         # Central wavelength, m
wl_lims = [700e-9, 1500e-9]  # Wavelength limits, m (for plots)
time_window_size = 5e-12     # Minimum width of the time window, s

g = grid.grid(central_wl, wl_lims, time_window_size)

# Fibre parameters and instantiation
L = 1
tol = 1e-5

smf = pf.PM980_XP(g, L, tol)

###############################################################################
#     Example of a good fit with lots of coefficients over a broad domain     #
###############################################################################

# Taylor coefficient calculations
N_coeffs = 17                      # return [beta_2, ..., beta_N_coeffs]
omega_lims = (g.omega_crop.min(),
              g.omega_crop.max())  # Expansion domain
win_length = int(g.points / 8)     # Savitzky-Golay filter size
polyorder = 2                      # Savitzky-Golay filter order
betas = ut.Maclaurin_coefficients(
    smf.beta_2, g.omega, g.dOmega, g, N_coeffs, omega_lims, win_length,
    polyorder)

beta_2_expansion = ut.Taylor_expansion(betas, g.omega_crop).real

stat_str = ("\n\nbeta_2 to beta_%d for PM980 XP when the coefficient "
            "calculation domain is set to g.omega_crop:" % (N_coeffs + 2))
underliner = '-'*len(stat_str)
print(stat_str)
print(underliner)

for i, beta in enumerate(betas):
    format_str = '%.3E' % Decimal(beta * 1e12**(i+2))
    print("beta %d = " % (i+2) + format_str + " ps^(%d) / m" % (i+2))

D_expansion = -2*np.pi*const.c * beta_2_expansion / g.lambda_window_crop**2
plot_comparison(1e9*g.lambda_window_crop, 1e6*smf.D, 1e6*D_expansion,
                'Wavelength, nm', 'D, ps/(nm km)',
                'Good fit, lots of coeffs, large domain')
plt.show()


###############################################################################
#     Example of a worse fit with fewer coefficients over the same domain     #
###############################################################################

# Taylor coefficient calculations
N_coeffs = 11                      # return [beta_2, ..., beta_N_coeffs]
omega_lims = (g.omega_crop.min(),
              g.omega_crop.max())  # Expansion domain
win_length = int(g.points / 8)     # Savitzky-Golay filter size
polyorder = 2                      # Savitzky-Golay filter order
betas = ut.Maclaurin_coefficients(
    smf.beta_2, g.omega, g.dOmega, g, N_coeffs, omega_lims, win_length,
    polyorder)

beta_2_expansion = ut.Taylor_expansion(betas, g.omega_crop).real

stat_str = ("\n\nbeta_2 to beta_%d for PM980 XP when the coefficient "
            "calculation domain is set to g.omega_crop:" % (N_coeffs + 2))
underliner = '-'*len(stat_str)
print(stat_str)
print(underliner)

for i, beta in enumerate(betas):
    format_str = '%.3E' % Decimal(beta * 1e12**(i+2))
    print("beta %d = " % (i+2) + format_str + " ps^(%d) / m" % (i+2))

D_expansion = -2*np.pi*const.c * beta_2_expansion / g.lambda_window_crop**2
plot_comparison(1e9*g.lambda_window_crop, 1e6*smf.D, 1e6*D_expansion,
                'Wavelength, nm', 'D, ps/(nm km)',
                'Worse fit, fewer coeffs, large domain')
plt.show()


###############################################################################
#     Example of a good fit with fewer coefficients over a smaller domain     #
###############################################################################

# Taylor coefficient calculations
N_coeffs = 11                            # return [beta_2, ..., beta_N_coeffs]
omega_lims = (0.5 * g.omega_crop.min(),
              0.5 * g.omega_crop.max())  # Expansion domain
win_length = int(g.points / 8)           # Savitzky-Golay filter size
polyorder = 2                            # Savitzky-Golay filter order
betas = ut.Maclaurin_coefficients(
    smf.beta_2, g.omega, g.dOmega, g, N_coeffs, omega_lims, win_length,
    polyorder)

# Get the correct domain for the expansion:
min_idx, _ = ut.find_nearest(omega_lims[0], g.omega_crop)
max_idx, _ = ut.find_nearest(omega_lims[1], g.omega_crop)
beta_2_expansion = ut.Taylor_expansion(
        betas, g.omega_crop[min_idx:max_idx]).real

stat_str = ("\n\nbeta_2 to beta_%d for PM980 XP when the coefficient "
            "calculation domain is made much smaller:" % (N_coeffs + 2))
underliner = '-'*len(stat_str)
print(stat_str)
print(underliner)

for i, beta in enumerate(betas):
    format_str = '%.3E' % Decimal(beta * 1e12**(i+2))
    print("beta %d = " % (i+2) + format_str + " ps^(%d) / m" % (i+2))

D_expansion = -2*np.pi*const.c * beta_2_expansion \
        / g.lambda_window_crop[min_idx:max_idx]**2
plot_comparison(1e9*g.lambda_window_crop[min_idx:max_idx],
                1e6*smf.D[min_idx:max_idx], 1e6*D_expansion,
                'Wavelength, nm', 'D, ps/(nm km)',
                'Ok fit, fewer coeffs, smaller domain')
plt.show()
