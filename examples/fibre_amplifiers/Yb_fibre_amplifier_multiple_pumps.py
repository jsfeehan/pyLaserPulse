#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This example shows, among other things, how to add a pump source which shares a
propagation direction with the pump that is defined when a doped fibre is
instantiated. At instantiation, it is possible to define two pumps with opposite
propagation directions, but it is not possible to define two pumps which share
a propagation direction. To do this, the fibre is first defined in the normal
way, and then the add_pump method is used before the propagation is simulated.
"""

from pyLaserPulse import grid
from pyLaserPulse import pulse
from pyLaserPulse import base_components
from pyLaserPulse.data import paths
from pyLaserPulse import optical_assemblies
from pyLaserPulse import single_plot_window


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
central_wl = 1080e-9  # Central wavelength, m
max_wl = 1160e-9      # Maximum wavelength, m

# Laser pulse parameters
tau = 100e-12         # Pulse duration, s
P_peak = [1500, 1.5]  # [P_x, P_y], W
f_rep = 20e6          # Repetition frequency, Hz
shape = 'Gauss'
order = 8             # 8th order supergaussian pulse shape

# Yb-fibre parameters
L_ydf = 4                            # length, m
doping = 8e25                        # ion number density, m-3
core_diam = 44e-6                    # core diameter in m
NA = 0.028                           # core numerical aperture
beat_length = 1e-2                   # polarization beat length
n2 = 2.19e-20                        # nonlinear index in m/W
fR = 0.18                            # Raman contribution to nonlinear response
tol = 1e-5                           # Integration error tolerance
ase_points = 2**8                    # number of points in pump & ASE grid
ase_wl_lims = [900e-9, max_wl]       # wavelength limits for ASE grid
bounds = {'counter_pump_power': 150,          # counter-pump power, W
          'counter_pump_wavelength': 976e-9,  # counter-pump wavelength, m
          'counter_pump_bandwidth': 1e-9}     # counter-pump bandwidth, m
cladding_pumping = {'pump_core_diam': 400e-6,      # pump core diameter, m
                    'pump_delta_n': 1.45 - 1.375,  # pump core ref. index step
                    'pump_cladding_n': 1.375}      # pump cladding ref. index

##############################################################
# Instantiate the time-frequency grid, pulse, and components #
##############################################################

# Time-frequency grid defined using the grid module
g = grid.grid(points, central_wl, max_wl)

# pulse defined using the pulse module
p = pulse.pulse(
    tau, P_peak, shape, f_rep, g, order=order, high_res_sampling=True)

# Define a custom double-clad, large-mode-area Yb-doped fibre and use the
# add_pump method to add another pump source.
ydf = base_components.step_index_active_fibre(
    g, L_ydf, paths.materials.loss_spectra.silica,
    paths.materials.Raman_profiles.silica, core_diam, NA, beat_length, n2,
    fR, tol, doping, paths.fibres.cross_sections.Yb_Al_silica,
    p.repetition_rate, ase_points, ase_wl_lims,
    paths.materials.Sellmeier_coefficients.silica, bounds, lifetime=1.5e-3,
    time_domain_gain=True, cladding_pumping=cladding_pumping)
ydf.add_pump(950e-9, 1e-9, 150, p.repetition_rate, 'counter')

################################################################
# Use the optical_assemblies module for automatic inclusion of #
# coupling loss between components and for generating plots.   #
################################################################
component_list = [ydf]
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
