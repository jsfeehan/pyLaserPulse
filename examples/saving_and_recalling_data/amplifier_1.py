#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This example demonstrates how to save simulation results. This is done very
easily using the optical_assemblies module, and all that is required is a
valid parent directory string which is passed to the data_directory keyword
argument when instantiating an optical_assemblies class.

Data is saved independently for each optical assembly. A new directory is
created in data_directory which has the same name as the optical assembly.
Inside this directory, the following files are created:
    grid.npz
    pulse.npz.
    optical_assembly.npz

These files can be loaded as a dictionary using the numpy.load method. The
files have the following dictionary keys:
    grid.npz -- Contains all data required to recreate the pyLaserPulse grid:
        points
        lambda_c
        lambda_max
    pulse.npz -- The sample fields are not present if sampling is off:
        field
        output
        power_spectral_density
        energy_spectral_density
        repetition_rate
        sample_points
        field_samples
        rep_rate_samples
        B_integral_samples
    optical_assembly.npz:
        sample_points
        net_co_PSD_samples
        net_counter_PSD_samples
        boundary_value_solver_ESD_optimization_loss
        boundary_value_solver_field_optimization_loss
        inversion_vs_distance
        net_co_PSD
        net_counter_PSD
        co_core_ASE_ESD_output -- co_ASE passed to subsequent amplifiers.
        pump_points
        pump_wl_lims

Loading data, shown in script amplifier_2.py in the same directory, is easy
to do using the pyLaserPulse.grid.grid_from_pyLaserPulse_simulation and
pyLaserPulse.pulse.pulse_grom_pyLaserPulse_simulation classes.

James Feehan <pylaserpulse@outlook.com>
25/6/2023.
"""

import os

from pyLaserPulse import grid
from pyLaserPulse import pulse
from pyLaserPulse import optical_assemblies
from pyLaserPulse import single_plot_window
import pyLaserPulse.catalogue_components.active_fibres as af
import pyLaserPulse.catalogue_components.fibre_components as fc


#############################################
# Choose a directory for saving the data.   #
# Leave as None if no data should be saved. #
#############################################
directory = os.path.dirname(os.path.abspath(__file__)) + '/'

############################################################
# Set time-frequency grid, pulse, and component parameters #
############################################################

# Time-frequency grid parameters
points = 2**11        # Number of grid points
central_wl = 1030e-9  # Central wavelength, m
max_wl = 1200e-9      # Maximum wavelength, m

# Laser pulse parameters
tau = 3e-12              # Pulse duration, s
P_peak = [.15, 1.5e-4]   # [P_x, P_y], W
f_rep = 40e6             # Repetition frequency, Hz
shape = 'Gauss'          # Can also take 'sech'

# isolator-WDM parameters
L_in = 0.2       # input fibre length, m
L_out = 0.2      # output fibre length, m

# Yb-fibre parameters
L = 1                                # length, m
ase_points = 2**8                    # number of points in pump & ASE grid
ase_wl_lims = [900e-9, max_wl]       # wavelength limits for ASE grid
bounds = {'co_pump_power': .15,          # co-pump power, W
          'co_pump_wavelength': 976e-9,  # co-pump wavelength, m
          'co_pump_bandwidth': 1e-9,     # co-pump bandwidth, m
          'counter_pump_power': 0}       # counter-pump power, W

##############################################################
# Instantiate the time-frequency grid, pulse, and components #
##############################################################

# Time-frequency grid defined using the grid module
g = grid.grid(points, central_wl, max_wl)

# pulse defined using the pulse module
p = pulse.pulse(tau, P_peak, shape, f_rep, g)

# Opneti isolator/WDM hybrid component from the catalogue_components module.
iso_wdm = fc.Opneti_PM_isolator_WDM_hybrid(g, L_in, L_out, g.lambda_c)

# Nufern PM-YSF-HI-HP defined using the catalogue_components module
ydf = af.Nufern_PM_YSF_HI_HP(g, L, p.repetition_rate, ase_points, ase_wl_lims,
                             bounds, time_domain_gain=True)

################################################################
# Use the optical_assemblies module for automatic inclusion of #
# coupling loss between components and for generating plots.   #
################################################################
component_list = [iso_wdm, ydf]
amp = optical_assemblies.sm_fibre_amplifier(
    g, component_list, plot=True, name='amp 1', high_res_sampling=100,
    data_directory=directory, verbose=True)

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
