#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
directory = None

############################################################
# Set time-frequency grid, pulse, and component parameters #
############################################################

# Time-frequency grid parameters
central_wl = 1030e-9         # Central wavelength, m
wl_lims = [900e-9, 1300e-9]  # Wavelength limits, m (for plots)
time_window_size = 5e-12     # Minimum width of the time window, s


# Laser pulse parameters
tau = 150e-15         # Pulse duration, s
P_peak = [150, .15]   # [P_x, P_y], W
f_rep = 40e6          # Repetition frequency, Hz
shape = 'sech'        # Can also take 'Gauss'

# isolator-WDM parameters
L_in = 0.2       # input fibre length, m
L_out = 0.2      # output fibre length, m

# Yb-fibre parameters
L = 1                                 # length, m
ase_points = 2**10                     # number of points in pump & ASE grid
ase_wl_lims = [800e-9, max(wl_lims)]  # wavelength limits for ASE grid
bounds = {'co_pump_power': 1,            # co-pump power, W
          'co_pump_wavelength': 916e-9,  # co-pump wavelength, m
          'co_pump_bandwidth': 1e-9,     # co-pump bandwidth, m
          'counter_pump_power': 0}       # counter-pump power, W

##############################################################
# Instantiate the time-frequency grid, pulse, and components #
##############################################################

# Time-frequency grid defined using the grid module
# g = grid.grid(points, central_wl, max_wl)
g = grid.grid(central_wl, wl_lims, time_window_size, verbose=True)

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
