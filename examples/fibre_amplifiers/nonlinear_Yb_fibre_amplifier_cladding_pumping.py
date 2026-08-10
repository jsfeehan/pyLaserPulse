#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyLaserPulse import grid
from pyLaserPulse import pulse
from pyLaserPulse import optical_assemblies
from pyLaserPulse import single_plot_window
import pyLaserPulse.catalogue_components.active_fibres as af
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
central_wl = 1055e-9         # Grid central wavelength, m
wl_lims = [900e-9, 1300e-9]  # Wavelength limits used for plots, m
t_span = 25e-12              # Minimum time window duration, s

# Laser pulse parameters
tau = 300e-15         # Pulse duration, s
P_peak = [1500, 7.5]  # [P_x, P_y], W
f_rep = 80e6          # Repetition frequency, Hz
shape = 'sech'        # Can also take 'Gauss'

# SMF pigtail
L_pmf = .3            # length, m

# Yb-fibre parameters
L_ydf = 8                   # length, m
ase_points = 2**10          # number of points in pump & ASE grid
ase_wl_lims = wl_lims       # wavelength limits for ASE grid
bounds = {'counter_pump_power': 15,           # counter-pump power, W
          'counter_pump_wavelength': 916e-9,  # counter-pump wavelength, m
          'counter_pump_bandwidth': 1e-9}     # counter-pump bandwidth, m

##############################################################
# Instantiate the time-frequency grid, pulse, and components #
##############################################################

# Time-frequency grid defined using the grid module
g = grid.grid(central_wl, wl_lims, t_span)

# pulse defined using the pulse module
p = pulse.pulse(tau, P_peak, shape, f_rep, g, high_res_sampling=True)

# PM980 'pigtail', defined using the catalogue_components module
pmf = pf.PM980_XP(g, L_pmf, 1e-5)

# Nufern PM-YSF-HI-HP defined using the catalogue_components module
ydf = af.Nufern_PLMA_YDF_25_250(
    g, L_ydf, p.repetition_rate, ase_points, ase_wl_lims, bounds,
    time_domain_gain=True, cladding_pumping=True)

################################################################
# Use the optical_assemblies module for automatic inclusion of #
# coupling loss between components and for generating plots.   #
################################################################
component_list = [pmf, ydf]
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
