#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This example shows how the python multiprocessing library can be used to
run  pyLaserPulse supercontinuum simulations in parallel (on multiple cores of
a CPU) with different quantum noise seeds so that that complex first-order
degree of coherence can be calculated.

The supercontinuum simulation is wrapped in a function which can then be
passed to a multiprocessing worker pool. The optical_assemblies module works
well for this, but the plot option must be turned off. The function can be
replaced by a class method if you prefer.

It is recommended that you become familiar with some of the other examples
before reading this one.

James Feehan <pylaserpulse.outlook.com>
"""


# Limit the number of threads spawned by any process to 1.
import os
os.environ["OMP_NUM_THREADS"] = "1"  # for openBLAS or similar
# os.environ["MKL_NUM_THREADS"] = "1"  # For Intel math kernel library

import psutil
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pyLaserPulse.catalogue_components.passive_fibres as pf
import pyLaserPulse.optical_assemblies as oa
import pyLaserPulse.pulse as pulse
import pyLaserPulse.grid as grid


def scg_sim(_g):
    """
    Supercontinuum simulation using pyLaserPulse

    Parameters
    ----------
    _g : pyLaserPulse.grid object

    Notes
    -----
    The pulse object is instantiated inside this function so that the quantum
    noise seed is different to all others in the ensemble.
    """
    # pulse
    duration = 200e-15  # pulse duration, s
    Pp = [15e3, 15]     # Peak power [slow axis, fast axis]
    shape = 'sech'      # can accept 'Gauss'
    rr = 50e6           # repetition rate
    delay = 0           # Initial delay from T = 0 s.
    p = pulse.pulse(duration, Pp, shape, rr, _g, initial_delay=delay)

    # Photonic crystal fibre (NKT SC-5.0-1040-PM) from catalogue_components
    length = .2  # Fibre length in m
    err = 1e-11  # integration error (CQEM)
    pcf = pf.NKT_NL_1050_NEG_1(_g, length, err, 1e-2)

    # supercontinuum optical assembly
    scg = oa.passive_assembly(_g, [pcf], 'scg', plot=False, verbose=False)

    p = scg.simulate(p)
    return p


if __name__ == "__main__":
    # grid
    points = 2**13    # Time-frequency grid points
    wl = 1040e-9      # Grid & pulse central wavelength
    max_wl = 4000e-9  # Max grid wavelength
    g = grid.grid(points, wl, max_wl)

    num_processes = psutil.cpu_count(logical=False)  # only use physical cores
    num_simulations = 4 * num_processes  # no. of simulations in CFODC ensemble
    gridlist = [[g]] * num_simulations  # iterable of func arguments
    pool = mp.get_context("spawn").Pool(
        processes=num_processes, maxtasksperchild=1)
    output_pulses = pool.starmap(scg_sim, gridlist)
    pool.close()
    pool.join()

    cfodc, lw = pulse.complex_first_order_degree_of_coherence(g, output_pulses)

    # Convert output_pulses[n].power_spectral_density into an array for easier
    # plotting
    PSDs = np.zeros((num_simulations, 2, g.points))
    for i, op in enumerate(output_pulses):
        PSDs[i, :, :] = op.power_spectral_density

    colors = ['seagreen', 'lightcoral']
    dark_colors = ['darkgreen', 'darkred']
    legend1 = []
    fig = plt.figure(num=1, figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for p in range(2):  # iterate over polarization basis vectors
        for i in range(num_simulations):
            ax1.semilogy(
                g.lambda_window * 1e9, PSDs[i, p, :], c=colors[p], alpha=0.1)
        l, = ax1.semilogy(
            g.lambda_window * 1e9, np.average(PSDs[:, p, :], axis=0),
            c=dark_colors[p])
        ax2.plot(
            lw * 1e9, cfodc[p, :], c=['k', 'grey'][p], ls=['-', '--'][p])
        legend1.append(l)
    ax1.set_ylabel('Power spectral density, mW/nm')
    ax1.set_xlabel('Wavelength, nm')
    ax2.set_ylabel('Complex first-order degree of coherence')
    ax2.set_xlabel('Wavelength, nm')
    ax1.set_xlim([1e9 * g.lambda_min, 1e9 * g.lambda_max])
    ax2.set_xlim([1e9 * g.lambda_min, 1e9 * g.lambda_max])
    ax1.legend(legend1, ['$S_{x}(\\lambda)$', '$S_{y}(\\lambda)$'])
    ax2.legend(['$g_{x}(\\lambda)$', '$g_{y}(\\lambda)$'])
    fig.tight_layout()
    plt.show()
