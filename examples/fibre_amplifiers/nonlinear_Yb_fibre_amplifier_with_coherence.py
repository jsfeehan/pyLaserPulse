#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This example shows how the python multiprocessing library can be used to
run  pyLaserPulse Yb-fibre amplifier simulations in parallel (on multiple cores
of a CPU) with different quantum noise seeds to that that complex first-order
degree of coherence can be calculated.

The simulation is wrapped in a function which can then be passed to a
multiprocessing worker pool. The optical_assemblies module works well for this,
but the plot option must be turned off. The function can be replaced by a class
method if you prefer.

It is recommended that you become familiar with some of the other examples
before reading this one.

James Feehan <pylaserpulse.outlook.com>
"""


import os
# Limit the number of threads spawned by any process to 1.
os.environ["OMP_NUM_THREADS"] = "1"  # for openBLAS or similar
# os.environ["MKL_NUM_THREADS"] = "1"  # For Intel math kernel library

from pyLaserPulse import grid
from pyLaserPulse import pulse
from pyLaserPulse import optical_assemblies
import pyLaserPulse.catalogue_components.active_fibres as af
import pyLaserPulse.catalogue_components.fibre_components as fc

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import psutil


def ydfa_sim(_g):
    """
    Yb-doped fibre amplifier simulation using pyLaserPulse

    Parameters
    ----------
    _g : pyLaserPulse.grid object

    Notes
    -----
    The pulse object is instantiated inside this function so that the quantum
    noise seed is different to all others in the ensemble.
    """
    # Laser pulse parameters
    tau = 150e-15         # Pulse duration, s
    P_peak = [150, .15]   # [P_x, P_y], W
    f_rep = 40e6          # Repetition frequency, Hz
    shape = 'sech'        # Can also take 'Gauss'
    p = pulse.pulse(tau, P_peak, shape, f_rep, _g)

    # isolator-WDM parameters
    L_in = 0.2       # input fibre length, m
    L_out = 0.2      # output fibre length, m

    # Yb-fibre parameters
    L = 1                                  # length, m
    ase_points = 2**8                      # no. of points in pump & ASE grid
    ase_wl_lims = [900e-9, _g.lambda_max]  # wavelength limits for ASE grid
    bounds = {'co_pump_power': 1,            # co-pump power, W
              'co_pump_wavelength': 916e-9,  # co-pump wavelength, m
              'co_pump_bandwidth': 1e-9,     # co-pump bandwidth, m
              'counter_pump_power': 0}       # counter-pump power, W
    iso_wdm = fc.Opneti_PM_isolator_WDM_hybrid(_g, L_in, L_out, _g.lambda_c)
    ydf = af.Nufern_PM_YSF_HI_HP(
        _g, L, p.repetition_rate, ase_points, ase_wl_lims, bounds,
        time_domain_gain=True)
    component_list = [iso_wdm, ydf]
    amp = optical_assemblies.sm_fibre_amplifier(
        _g, component_list, plot=False, name='amp 1', verbose=False)
    p = amp.simulate(p)
    return p


if __name__ == "__main__":
    # Time-frequency grid parameters
    points = 2**9         # Number of grid points
    central_wl = 1030e-9  # Central wavelength, m
    max_wl = 1200e-9      # Maximum wavelength, m
    g = grid.grid(points, central_wl, max_wl)

    num_processes = psutil.cpu_count(logical=False)  # only use physical cores
    num_simulations = 4*num_processes  # no. of simulations in CFODC ensemble
    gridlist = [[g]] * num_simulations  # iterable of func arguments
    pool = mp.get_context("spawn").Pool(
            processes=num_processes, maxtasksperchild=1)
    output_pulses = pool.starmap(ydfa_sim, gridlist)
    pool.close()
    pool.join()

    cfodc, lw = pulse.complex_first_order_degree_of_coherence(g, output_pulses)

    # Convert output_pulses[n].power_spectral_density into an array for easier
    # plotting
    PSDs = np.zeros((num_simulations, 2, g.points))
    for i, op in enumerate(output_pulses):
        PSDs[i, :, :] = op.power_spectral_density

    colors = ['seagreen', 'lightcoral']
    dark_colors = ['darkgreen', 'indianred']
    legend1 = []
    fig = plt.figure(num=1, figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for p in range(2):  # iterate over polarization basis vectors
        l, = ax1.semilogy(
            g.lambda_window*1e9, np.average(PSDs[:, p, :], axis=0),
            c=dark_colors[p])
        for i in range(num_simulations):
            ax1.semilogy(
                g.lambda_window*1e9, PSDs[i, p, :], c=colors[p], alpha=0.1)
        ax2.plot(
            lw*1e9, cfodc[p, :], c=['k', 'grey'][p], ls=['-', '--'][p])
        legend1.append(l)
    ax1.set_ylabel('Power spectral density, mW/nm')
    ax1.set_xlabel('Wavelength, nm')
    ax2.set_ylabel('Complex first-order degree of coherence')
    ax2.set_xlabel('Wavelength, nm')
    ax1.set_xlim([1e9*g.lambda_min, 1e9*g.lambda_max])
    ax2.set_xlim([1e9*g.lambda_min, 1e9*g.lambda_max])
    ax1.legend(legend1, ['$S_{x}(\lambda)$', '$S_{y}(\lambda)$'])
    ax2.legend(['$g_{x}(\lambda)$', '$g_{y}(\lambda)$'])
    fig.tight_layout()
    plt.show()
