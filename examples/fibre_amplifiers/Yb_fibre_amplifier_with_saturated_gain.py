#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyLaserPulse import grid
from pyLaserPulse import pulse
from pyLaserPulse import optical_assemblies
from pyLaserPulse import single_plot_window
import pyLaserPulse.base_components as bc
import pyLaserPulse.catalogue_components.fibre_components as fc
import pyLaserPulse.data.paths as paths


#############################################
# Choose a directory for saving the data.   #
# Leave as None if no data should be saved. #
#############################################
directory = None

############################################################
# Set time-frequency grid, pulse, and component parameters #
############################################################

# Time-frequency grid parameters
points = 2**18         # Number of grid points
central_wl = 1030e-9  # Central wavelength, m
max_wl = 1090e-9      # Maximum wavelength, m

# Laser pulse parameters
tau = 1.2e-9          # Pulse duration, s
P_peak = [9400, 0]    # [P_x, P_y], W
f_rep = 40e3          # Repetition frequency, Hz
shape = 'Gauss'
chirp = 16000         # Chirp factor

# isolator-WDM parameters
L_in = 0.2       # input fibre length, m
L_out = 0.2      # output fibre length, m

# Yb-fibre parameters
L = .8                                # length, m
ase_points = 2**9                    # number of points in pump & ASE grid
ase_wl_lims = [900e-9, max_wl]       # wavelength limits for ASE grid
bounds = {'co_pump_power': 15,            # co-pump power, W
          'co_pump_wavelength': 976e-9,  # co-pump wavelength, m
          'co_pump_bandwidth': 1e-9,     # co-pump bandwidth, m
          'counter_pump_power': 0}       # counter-pump power, W
tol = 1e-5
core_diam = 20e-6
NA = 0.045
beat_length = 1e-2
n2 = 2.19e-20
fR = 0.18
doping_concentration = 5.2e25
cladding_pumping = {
    'pump_core_diam': 125e-6, 'pump_delta_n': 1.4467 - 1.375,
    'pump_cladding_n': 1.375}

##############################################################
# Instantiate the time-frequency grid, pulse, and components #
##############################################################

# Time-frequency grid defined using the grid module
g = grid.grid(points, central_wl, max_wl)

# pulse defined using the pulse module
p = pulse.pulse(tau, P_peak, shape, f_rep, g, chirp=chirp)

# p.get_chirp(g, p.field)
# import matplotlib.pyplot as plt
# import numpy as np
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax2 = ax1.twinx()
# ax1.semilogy(g.lambda_window, p.power_spectral_density.T)
# # ax1.plot(g.time_window*1e9, np.abs(p.field[0, :])**2)
# # ax2.plot(g.time_window*1e9, 1e9 * p.chirp[0, :])
# plt.show()

# Define custom YDF to match example in paper
ydf = bc.step_index_active_fibre(
        g, L, paths.materials.loss_spectra.silica,
        paths.materials.Raman_profiles.silica, core_diam, NA, beat_length, n2,
        fR, tol, doping_concentration, paths.fibres.cross_sections.Yb_Al_silica,
        p.repetition_rate, ase_points, ase_wl_lims,
        paths.materials.Sellmeier_coefficients.silica, bounds, lifetime=1e-3,
        cladding_pumping=cladding_pumping, time_domain_gain=True, verbose=True)

################################################################
# Use the optical_assemblies module for automatic inclusion of #
# coupling loss between components and for generating plots.   #
################################################################
component_list = [ydf]
amp = optical_assemblies.sm_fibre_amplifier(
    g, component_list, plot=True, name='amp 1', high_res_sampling=200,
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
