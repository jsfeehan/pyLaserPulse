#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This example demonstrates how to load simulation results. This is done very
easily using the grid_from_pyLaserPulse_simulation and
pulse_from_pyLaserPulse_simulation classes, and all that is required is a
valid parent directory string which is passed to these classes at
instantiation.

Amplifiers can accept co-propagating ASE from preceding amplifiers using the
co_ASE keyword argument in the optical_assemblies.sm_fibre_amplifier class
__init__ method. The co-propagating ASE can be loaded from
/amp 1/optical_assembly.npz.

James Feehan <pylaserpulse@outlook.com>
25/6/2023.
"""


import os
import numpy as np

from pyLaserPulse import grid
from pyLaserPulse import pulse
from pyLaserPulse import optical_assemblies
from pyLaserPulse import single_plot_window
import pyLaserPulse.catalogue_components.active_fibres as af
import pyLaserPulse.catalogue_components.fibre_components as fc
import pyLaserPulse.catalogue_components.passive_fibres as pf
import pyLaserPulse.base_components as bc


############################################################################
# Enter the directory where the data for the preceding amplifier was saved #
############################################################################
directory = os.path.dirname(os.path.abspath(__file__)) + '/amp 1/'

##############################################################
# Instantiate the time-frequency grid, pulse, and components #
##############################################################

# Time-frequency grid defined using the grid module
g = grid.grid_from_pyLaserPulse_simulation(directory)

# pulse defined using the pulse module
p = pulse.pulse_from_pyLaserPulse_simulation(g, directory) 

# amp 1 data
amp_1 = np.load(directory + 'optical_assembly.npz')

# isolator-WDM parameters
L_in = 0.2       # input fibre length, m
L_out = 0.2      # output fibre length, m

# Yb-fibre parameters
L = 3                                # length, m
ase_points = amp_1['pump_points']    # number of points in pump & ASE grid
ase_wl_lims = amp_1['pump_wl_lims']           # wavelength limits for ASE grid
bounds = {'co_pump_power': 0,                 # co-pump power, W
          'co_pump_wavelength': 976e-9,       # co-pump wavelength, m
          'co_pump_bandwidth': 1e-9,          # co-pump bandwidth, m
          'counter_pump_power': 4,            # counter-pump power, W
          'counter_pump_wavelength': 976e-9,  # counter-pump wavelength, m
          'counter_pump_bandwidth': 1e-9}     # counter-pump bandwidth, m

# isolator
iso = fc.Opneti_high_power_PM_isolator(g, L_in, L_out, g.lambda_c)

# pump combiner
in_fibre = pf.PM980_XP(g, L_in, 1e-5)
out_fibre = pf.Nufern_PLMA_GDF_25_250(g, 0.5, 1e-5)
combiner = bc.fibre_component(
    g, in_fibre, out_fibre, 0.2, 200e-9, g.lambda_c, 1, 0, 0, 1e-5, order=5)

# YDF
ydf = af.Nufern_PLMA_YDF_25_250(
    g, L, p.repetition_rate, ase_points, ase_wl_lims, bounds,
    time_domain_gain=True, cladding_pumping=True)

################################################################
# Use the optical_assemblies module for automatic inclusion of #
# coupling loss between components and for generating plots.   #
################################################################
component_list = [iso, combiner, ydf]
amp = optical_assemblies.sm_fibre_amplifier(
    g, component_list, plot=True, name='amp 2', high_res_sampling=100,
    verbose=True, co_ASE=amp_1['co_core_ASE_ESD_output'])

######################
# Run the simulation #
######################
p = amp.simulate(p)

##########################################################
# Use the matplotlib_gallery module to display the plots #
##########################################################
if amp.plot:
    plot_dicts = [amp.plot_dict]
    single_plot_window.matplotlib_gallery.launch_plot(plot_dicts=plot_dicts)
