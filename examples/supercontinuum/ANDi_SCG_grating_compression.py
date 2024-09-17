#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pyLaserPulse import grid
from pyLaserPulse import pulse
from pyLaserPulse import base_components
from pyLaserPulse import data
from pyLaserPulse import optical_assemblies
from pyLaserPulse import single_plot_window
import pyLaserPulse.catalogue_components.passive_fibres as pf


#############################################
# Choose a directory for saving the data.   #
# Leave as None if no data should be saved. #
#############################################
directory = None

############################################################
# Set time-frequency grid, pulse, and component parameters #
############################################################

# Time-frequency grid parameters
points = 2**14        # Number of grid points
central_wl = 1050e-9  # Central wavelength, m
max_wl = 8000e-9      # Maximum wavelength, m

# Laser pulse parameters
tau = 100e-15         # Pulse duration, s
P_peak = [25000, 250]   # [P_x, P_y], W
f_rep = 40e6          # Repetition frequency, Hz
shape = 'Gauss'       # Can also take 'sech'

# ANDi photonic crystal fibre parameters
L_beat = 1e-2  # polarization beat length (m)
L = .25          # length, m

# grating compressor parameters
loss = 0.04            # percent loss per grating reflection
transmission = 700e-9  # transmission bandwidth
coating = data.paths.materials.reflectivities.gold
epsilon = 1e-1         # Jones parameter for polarization mixing and phase
theta = 0              # Jones parameter for angle subtended by x-axis
crosstalk = 1e-3       # polarization crosstalk
beamsplitting = 0      # Useful for output couplers, etc.
l_mm = 600             # grating lines per mm
sep_initial = 1e-2     # initial guess for grating separation
angle_initial = 0.31   # initial guess for incidence angle, rad

##############################################################
# Instantiate the time-frequency grid, pulse, and components #
##############################################################

# Time-frequency grid defined using the grid module
g = grid.grid(points, central_wl, max_wl)

# pulse defined using the pulse module
p = pulse.pulse(tau, P_peak, shape, f_rep, g)

# isolator
iso = base_components.component(
    0.2, 250e-9, g.lambda_c, epsilon, theta, 0, g, crosstalk, order=5)

# ANDi photonic crystal fibre - NKT NL-1050-NEG-1 - from catalogue_components
pcf = pf.NKT_NL_1050_NEG_1(g, L, 1e-6, L_beat)

# grating compressor defined using the base_components module
gc = base_components.grating_compressor(
    loss, transmission, coating, g.lambda_c, epsilon, theta, beamsplitting,
    crosstalk, sep_initial, angle_initial, l_mm, g, order=5, optimize=True)

################################################################
# Use the optical_assemblies module for automatic inclusion of #
# coupling loss between components and for generating plots.   #
################################################################

scg_components = [iso, pcf]
scg = optical_assemblies.passive_assembly(
    g, scg_components, 'scg', high_res_sampling=100,
    plot=True, data_directory=directory, verbose=True)

compressor_components = [gc]
compression = optical_assemblies.passive_assembly(
    g, compressor_components, 'compressor', plot=True,
    data_directory=directory, verbose=True)

######################
# Run the simulation #
######################
p = scg.simulate(p)
p = compression.simulate(p)

##########################################################
# Use the matplotlib_gallery module to display the plots #
##########################################################
if scg.plot or compression.plot:
    plot_dicts = [scg.plot_dict, compression.plot_dict]
    single_plot_window.matplotlib_gallery.launch_plot(plot_dicts=plot_dicts)
