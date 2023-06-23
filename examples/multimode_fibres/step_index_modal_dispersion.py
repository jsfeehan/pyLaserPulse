#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pyLaserPulse import grid
from pyLaserPulse import bessel_mode_solver as bms
from pyLaserPulse import utils as utils
from pyLaserPulse.data import paths


if __name__ == "__main__":

    ##########################################
    # Define the step-index fibre parameters #
    ##########################################
    core_rad = 50e-6 / 2
    cladding_rad = 125e-6 / 2
    NA = 0.22

    ###################################################################
    # Choose the wavelength range and maximum number of modes to find #
    ###################################################################
    g = grid.grid(2**7, 1025e-9, 1055e-9)
    max_modes = 50

    beta = []  # propagation constants

    for i, wavelength in enumerate(g.lambda_window):

        ######################################################################
        # Use the Sellmeier equation for the fibre core and cladding indices #
        ######################################################################
        n_cladding = utils.Sellmeier(
            wavelength, paths.materials.Sellmeier_coefficients.silica)
        n_core = (NA**2 + n_cladding**2)**(0.5)

        ##########################
        # Instantiate the solver #
        ##########################
        solver = bms.bessel_mode_solver(
            core_rad, cladding_rad, n_core, n_cladding, wavelength)

        ##################
        # Find the modes #
        ##################
        solver.solve(max_modes=max_modes)

        ###############################################################
        # Sort the mode propagation constants in descending order and #
        # calculate the phase velocity of each mode                   #
        ###############################################################
        sort_idx = np.argsort(solver.beta_arr)[::-1]
        beta.append(solver.beta_arr[sort_idx])

    ################################################
    # Ensure each array in beta is the same length #
    ################################################
    max_l = np.inf
    for i in range(len(beta)):  # Get length of smallest array
        l = len(beta[i])
        if l < max_l:
            max_l = l
    for i in range(len(beta)):  # Restrict lengths of all arrays
        beta[i] = beta[i][0:max_l:1]

    ################################################
    # Calculate the group velocity and group index #
    ################################################
    v_group = 1 / np.gradient(beta, g.omega_window, axis=0)
    n_group = const.c / v_group

    fig1, ax1 = plt.subplots()
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im1 = ax1.pcolormesh(
        np.linspace(0, max_l - 1, max_l), g.lambda_window * 1e9, n_group)
    ax1.set_xlabel('Mode number')
    ax1.set_ylabel('Wavelength, nm')
    cbar1 = fig1.colorbar(im1, cax=cax, orientation='vertical')
    cbar1.set_label('Group index')
    fig1.tight_layout()

    fig2, ax2 = plt.subplots()
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im2 = ax2.pcolormesh(
        np.linspace(
            0,
            max_l - 1,
            max_l),
        g.lambda_window * 1e9,
        1e-6 * v_group)
    ax2.set_xlabel('Mode number')
    ax2.set_ylabel('Wavelength, nm')
    cbar2 = fig1.colorbar(im2, cax=cax, orientation='vertical')
    cbar2.set_label('Group velocity, Mm/s')
    fig2.tight_layout()

    plt.show()
