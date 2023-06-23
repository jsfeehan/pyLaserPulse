#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pyLaserPulse.grid as grid
import pyLaserPulse.pulse as pulse
import pyLaserPulse.optical_assemblies as oa
import pyLaserPulse.single_plot_window as spw
import pyLaserPulse.catalogue_components.passive_fibres as pf


#############################################
# Choose a directory for saving the data.   #
# Leave as None if no data should be saved. #
#############################################
directory = None


##############################################################
# Instantiate the time-frequency grid, pulse, and components #
##############################################################
# Time-frequency grid defined using the grid module
points = 2**13    # Time-frequency grid points
wl = 1040e-9      # Grid & pulse central wavelength
max_wl = 2000e-9  # Max grid wavelength
g = grid.grid(points, wl, max_wl)

# pulse defined using the pulse module
duration = 80e-15  # pulse duration, s
Pp = [47e3, 47]    # Peak power [slow axis, fast axis]
shape = 'sech'     # can accept 'Gauss'
rr = 50e6          # repetition rate
delay = -12e-12    # Initial delay from T = 0 s.
p = pulse.pulse(duration, Pp, shape, rr, g, initial_delay=delay)

# Photonic crystal fibre (NKT SC-5.0-1040-PM) from catalogue_components
lenght = 1  # Fibre length in m
err = 1e-8  # integration error (CQEM)
pcf = pf.NKT_SC_5_1040_PM(g, lenght, err)


################################################################
# Use the optical_assemblies module for automatic inclusion of #
# coupling loss between components and for generating plots.   #
################################################################
scg = oa.passive_assembly(
    g,
    [pcf],
    'scg',
    high_res_sampling=200,
    plot=True,
    data_directory=directory)


######################
# Run the simulation #
######################
p = scg.simulate(p)


##########################################################
# Use the matplotlib_gallery module to display the plots #
##########################################################
plot_dicts = [scg.plot_dict]
spw.matplotlib_gallery.launch_plot(plot_dicts=plot_dicts)
