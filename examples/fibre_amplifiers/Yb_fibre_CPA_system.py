#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate a 3-amplifier Yb-doped CPA system using pyLaserPulse.

The seed pulses have a central wavelength of 1083 nm, a repetition rate of 40
MHz, and are chirped to 3 ps (transform limit of 124 fs). The first component
is a circulator and CFBG setup which stretches the pulses to ~275 ps FWHM.
Three amplifiers follow, with an AOM to reduce the repetition rate to 1 MHz
after the first. Each amplifier is cladding pumped to give the quasi-four-level
dynamics required for high gain at the seed wavelength. The CFBG dispersion
Taylor coefficients are calculated from the initial compressor dispersion and
the net fibre dispersion (note that they are unlikely to be optimal for this
CPA system).

This example also shows how co-propagating ASE can be passed from one amplifier
to the next using the optical_assemblies module and the co_ASE keyword
argument.

This simulation takes some time to run. This is because both a broad time and
frequency grid are required for strongly-chirped femtosecond pulses, and this
requires a lot of grid points. Additionally, cladding-pumped amplifiers take a
bit longer to simulate than core-pumped amplifiers because the overlap of the
pump light with the signal core is calculated using a Bessel mode solver and the
doped fibres tend to be longer than in core-pumped amplifiers.

James Feehan, 19/6/2023
"""

import numpy as np

from pyLaserPulse import grid
from pyLaserPulse import pulse
from pyLaserPulse import base_components
from pyLaserPulse import optical_assemblies
from pyLaserPulse import single_plot_window
from pyLaserPulse import utils
from pyLaserPulse import data
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
points = 2**15        # Number of grid points
central_wl = 1083e-9  # Central wavelength, m
max_wl = 1150e-9      # Maximum wavelength, m

# Laser pulse parameters
tau = 3e-12             # Pulse duration, s
chirp = 20              # 3 ps pulse with bandwidth, strong positive chirp
P_peak = [300, 0.3]     # [P_x, P_y], W
f_rep = 40e6            # Repetition frequency, Hz
shape = 'Gauss'


##############################################################
#        Instantiate the time-frequency grid and pulse       #
##############################################################
# Time-frequency grid defined using the grid module
g = grid.grid(points, central_wl, max_wl)

# pulse defined using the pulse module.
# Print some useful data to the terminal
p = pulse.pulse(
        tau, P_peak, shape, f_rep, g, chirp=chirp)
p.get_ESD_and_PSD(g, p.field)
p.get_transform_limit(p.field)
T_FWHM = utils.get_width(g.time_window*1e12, np.abs(p.field[0, :])**2)
S_FWHM = utils.get_width(
        g.lambda_window*1e9, p.power_spectral_density[0, :])
TL_FWHM = utils.get_width(g.time_window*1e15, p.transform_limit)
print("Starting FWHM duration: %.3f ps"
      "\nStarting FWHM spectral width: %.3f nm"
      "\nStarting transform limited FWHM duration: %.3f fs"
      "\nStarting average power: %.3f mW"
      "\nStarting pulse energy: %.3f nJ"
      % (T_FWHM, S_FWHM, TL_FWHM, p.average_power*1e3, p.pulse_energy*1e9))

# Pump and ASE grid
pump_points = 2**9
pump_wl_lims = [900e-9, g.lambda_max]

# Useful constants
ps = 1e-12


##########################################
# Repeat components and generic settings #
##########################################
tol = 1e-4        # integration tolerance for CQEM
crosstalk = 1e-5  # General crosstalk value for standard components.
pm_pigtail = pf.PM980_XP(g, 0.3, tol)
dc_pm_pigtail = pf.Nufern_PM_GDF_5_130(g, 0.3, tol)
dc_10_125_pm_pigtail = pf.Nufern_PLMA_GDF_10_125_M(g, 0.3, tol)
dc_25_250_pm_pigtail = pf.Nufern_PLMA_GDF_25_250(g, 0.3, tol)
num_samples = 10  # num field samples per component


######################################################################
# chirped fibre Bragg grating stretcher, amp 1, and AOM pulse picker #
######################################################################
circulator_1_to_2 = base_components.fibre_component(
    g, pm_pigtail, pm_pigtail, 0.2, 100e-9, g.lambda_c, 0.1, 0, 0, crosstalk)
# CFBG dispersion calculated by adding the compressor dispersion and total
# fibre dispersion and then multiplying by -1.
beta_2 = -1 * ps**2 * (
    0.045 + 0.058 + 0.028 + 0.0043 + 0.12 + 0.047 + 0.072 - 13.53)
beta_3 = -1 * ps**3 * (
    6.26e-5 + 7.02e-5 + 7.8e-5 + 1.43e-5 + 1.46e-4 + 1.3e-4 + 2.39e-4
    + 0.0582)
beta_4 = -1 * ps**4 * (
    -1.48e-5 - 1.338e-5 + 0.995e-6 - 1.526e-6 - 2.788e-5 + 1.659e-6 - 2.54e-5
    - 3.53e-4)
beta_5 = -1 * ps**5 * (
    1.283e-8 + 1.16e-8 - 7.8e-10 + 1.33e-9 + 2.419e-8 - 1.3e-9 + 2.217e-8
    + 3.04e-6)
cfbg = base_components.fibre_component(
    g, pm_pigtail, pm_pigtail, 0.4, 50e-9, g.lambda_c, 1, 0, 0, crosstalk,
    order=5, beta_list=[beta_2, beta_3, beta_4, beta_5])
circulator_2_to_3 = base_components.fibre_component(
    g, pm_pigtail, pm_pigtail, 0.2, 100e-9, g.lambda_c, 0.1, 0, 0, crosstalk)
combiner1 = base_components.fibre_component(
    g, pm_pigtail, dc_pm_pigtail, 0.2, 60e-9, g.lambda_c, 1, 0, 0, crosstalk)
bounds_1 = {'co_pump_wavelength': 976e-9,
            'co_pump_power': 0.5,
            'co_pump_bandwidth': 1e-9,
            'counter_pump_power': 0}
ydf1 = af.Nufern_PM_YDF_5_130_VIII(
    g, 5, p.repetition_rate, pump_points, pump_wl_lims,
    boundary_conditions=bounds_1, time_domain_gain=True, cladding_pumping=True)
bp1 = base_components.fibre_component(
    g, dc_pm_pigtail, dc_pm_pigtail, 0.2, 40e-9, g.lambda_c, 1, 0, 0,
    crosstalk, order=4)
time_gate = 20e-9
new_rep_rate = 1e6
reduction = int(p.repetition_rate / new_rep_rate)
aom_loss = 10**-0.35  # 3.5 dB insertion loss
aom_bw = 60e-9  # transmission bandwidth
per = 0.1
theta = 0
beamsplitting = 0
aom = base_components.fibre_pulse_picker(
    g, dc_pm_pigtail, dc_pm_pigtail, aom_loss, aom_bw, g.lambda_c, per, theta,
    beamsplitting, crosstalk, time_gate, reduction, p.repetition_rate, order=6)
components1 = [
        circulator_1_to_2, cfbg, circulator_2_to_3, combiner1, ydf1, bp1, aom]
amp_1 = optical_assemblies.sm_fibre_amplifier(
    g, components1, high_res_sampling=num_samples, plot=True,
    data_directory=directory, name='amp 1', verbose=True)
p = amp_1.simulate(p)


#########
# Amp 2 #
#########
iso1 = base_components.fibre_component(
    g, dc_pm_pigtail, dc_pm_pigtail, 0.2, 100e-9, g.lambda_c, 0.05, 0, 0,
    crosstalk, order=5)
combiner2 = base_components.fibre_component(
    g, dc_pm_pigtail, dc_10_125_pm_pigtail, 0.2, 60e-9, g.lambda_c, 1, 0, 0,
    crosstalk)
bounds_2 = {'co_pump_wavelength': 976e-9,
            'co_pump_power': 1,
            'co_pump_bandwidth': 1e-9,
            'counter_pump_power': 0}
ydf2 = af.Nufern_PLMA_YDF_10_125_M(
    g, 3, p.repetition_rate, pump_points, pump_wl_lims,
    boundary_conditions=bounds_2, time_domain_gain=True, cladding_pumping=True)
bp2 = base_components.fibre_component(
    g, dc_10_125_pm_pigtail, dc_10_125_pm_pigtail, 0.2, 40e-9, g.lambda_c, 1,
    0, 0, crosstalk, order=4)
components2 = [iso1, combiner2, ydf2, bp2]  # , aom]
amp_2 = optical_assemblies.sm_fibre_amplifier(
    g, components2, high_res_sampling=num_samples, plot=True,
    data_directory=directory, name='amp 2',
    co_ASE=amp_1.co_core_ASE_ESD_output, verbose=True)
p = amp_2.simulate(p)

#########
# Amp 3 #
#########
iso2 = base_components.fibre_component(
    g, dc_10_125_pm_pigtail, dc_10_125_pm_pigtail, 0.2, 100e-9, g.lambda_c,
    0.05, 0, 0, crosstalk, order=5)
combiner3 = base_components.fibre_component(
    g, dc_10_125_pm_pigtail, dc_25_250_pm_pigtail, 0.2, 60e-9, g.lambda_c, 1,
    0, 0, crosstalk)
bounds_3 = {'co_pump_wavelength': 976e-9,
            'co_pump_power': 0,
            'co_pump_bandwidth': 1e-9,
            'counter_pump_power': 5,
            'counter_pump_wavelength': 976e-9,
            'counter_pump_bandwidth': 1e-9}
ydf3 = af.Nufern_PLMA_YDF_25_250(
    g, 4, p.repetition_rate, pump_points, pump_wl_lims,
    boundary_conditions=bounds_3, time_domain_gain=True, cladding_pumping=True)
components3 = [iso2, combiner3, ydf3]
amp_3 = optical_assemblies.sm_fibre_amplifier(
    g, components3, high_res_sampling=num_samples, plot=True,
    data_directory=directory, name='amp 3',
    co_ASE=amp_2.co_core_ASE_ESD_output, verbose=True)
p = amp_3.simulate(p)

##############
# Compressor #
##############
loss = 0.04           # percent loss per grating reflection
transmission = 20e-9  # transmission bandwidth
coating = data.paths.materials.reflectivities.gold
epsilon = 1e-1         # Jones parameter for polarization mixing and phase
theta = 0              # Jones parameter for angle subtended by x-axis
beamsplitting = 0      # Useful for output couplers, etc.
l_mm = 1200            # grating lines per mm
sep_initial = 90e-2    # initial guess for grating separation
angle_initial = 0.7   # initial guess for incidence angle, rad
gc = base_components.grating_compressor(
    loss, transmission, coating, g.lambda_c, epsilon, theta, beamsplitting,
    crosstalk, sep_initial, angle_initial, l_mm, g, order=5, optimize=True)
compressor = optical_assemblies.passive_assembly(
        g, [gc], 'compressor', plot=True, data_directory=directory,
        verbose=True)
p = compressor.simulate(p)

############
# Plotting #
############
plot_dicts = [
    amp_1.plot_dict, amp_2.plot_dict, amp_3.plot_dict, compressor.plot_dict]
single_plot_window.matplotlib_gallery.launch_plot(plot_dicts=plot_dicts)
