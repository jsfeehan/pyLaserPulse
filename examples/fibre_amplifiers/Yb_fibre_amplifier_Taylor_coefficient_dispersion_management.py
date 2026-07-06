#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Show how dispersion Taylor coefficients can be calculated for different fibre
types and how this information can be used to balance dispersion in a CPA
system.

This method is not optimal, but works.

The idea is to define all fibre types used up front, then calculate their
Taylor coefficients using pyLaserPulse.utils.Maclaurin_coefficients (and then
checking that the returned coefficients are representative -- see the Taylor
coefficient calculation example). Summing these for each fibre, scaled by the
fibre lengths, subtracting them from the corresponding coefficients for the
compressor and multiplying by -1 will give cFBG parameters which, neglecting
SPM and other nonlinear chirps, will give a good pulse shape after compression.

The Taylor coefficients for each fibre type have to be calculated and then
manipulated separately from the fibre objects themselves. This is because the
Taylor coefficient calculations _can_, under some circumstances, require some
iterative adjustment before they are representative of the actual dispersion
curve. Doing this each time a fibre object is instantiated would be a pain, so
this roundabout method ends up being faster. This is especially true when
considering that pyLaserPulse doesn't depend on Taylor coefficients, and that
they have been included in this module as a bonus for convenience/aid in some
system design tasks.
"""


from pyLaserPulse import grid
from pyLaserPulse import pulse
import pyLaserPulse.optical_assemblies as oa
import pyLaserPulse.single_plot_window as spw
import pyLaserPulse.base_components as bc
import pyLaserPulse.utils as ut
import pyLaserPulse.catalogue_components.active_fibres as af
import pyLaserPulse.catalogue_components.passive_fibres as pf


###############################################################################
#                         Grid and pulse instantiation                        #
###############################################################################
g = grid.grid(1040e-9, (900e-9, 1300e-9), 200e-12)
p = pulse.pulse(100e-15, [1000, .1], 'Gauss', 20e6, g)


###############################################################################
#                            Fibre instantiation                              #
###############################################################################
tol = 1e-5
crosstalk = 1e-5
pm_pigtail = pf.PM980_XP(g, 1, tol)
pump_points = 2**9
pump_wl_lims = [g.lambda_window_crop.min(), g.lambda_window_crop.max()]
bounds = {'co_pump_wavelength': 976e-9,
          'co_pump_power': 1,
          'co_pump_bandwidth': 1e-9,
          'counter_pump_power': 0}
ydf = af.Nufern_PM_YDF_5_130_VIII(
    g, 0.5, p.repetition_rate, pump_points, pump_wl_lims,
    boundary_conditions=bounds, time_domain_gain=True, cladding_pumping=False)


###############################################################################
#        Taylor coefficients for the different fibre types in s^N / m         #
###############################################################################
N = 11  # number of Taylor coefficients counting from beta_2
pm_beta = ut.Maclaurin_coefficients(
    pm_pigtail.beta_2, g.omega, g.dOmega, g, N,
    (g.omega_crop.min(), g.omega_crop.max()), int(g.points / 8))
ydf_beta = ut.Maclaurin_coefficients(
    ydf.beta_2, g.omega, g.dOmega, g, N,
    (g.omega_crop.min(), g.omega_crop.max()), int(g.points / 8))


###############################################################################
#                           Passive components                                #
###############################################################################
circulator_1_to_2 = bc.fibre_component(
    g, pm_pigtail, pm_pigtail, 0.2, 100e-9, g.lambda_c, 0.1, 0, 0, crosstalk)
circulator_2_to_3 = bc.fibre_component(
    g, pm_pigtail, pm_pigtail, 0.2, 100e-9, g.lambda_c, 0.1, 0, 0, crosstalk)
wdm = bc.fibre_component(
    g, pm_pigtail, pm_pigtail, 0.2, 100e-9, g.lambda_c, 0.1, 0, 0, crosstalk)

# Need to define the cFBG length elsewhere because it can't be instantiated
# before the Taylor coefficients have first been calculated.
cfbg_L = 0.5

###############################################################################
#                 Net Taylor coefficients for the fibres                      #
###############################################################################
fibre_betas = [circulator_1_to_2.input_fibre.L * b for b in pm_beta]
fibre_betas = [
        _b + circulator_1_to_2.output_fibre.L * b
        for (_b, b) in zip(fibre_betas, pm_beta)]
fibre_betas = [
        _b + circulator_2_to_3.input_fibre.L * b
        for _b, b in zip(fibre_betas, pm_beta)]
fibre_betas = [
        _b + circulator_2_to_3.output_fibre.L * b
        for _b, b in zip(fibre_betas, pm_beta)]
fibre_betas = [
        _b + wdm.input_fibre.L * b for _b, b in zip(fibre_betas, pm_beta)]
fibre_betas = [
        _b + wdm.output_fibre.L * b for _b, b in zip(fibre_betas, pm_beta)]
fibre_betas = [
        _b + ydf.L * b for _b, b in zip(fibre_betas, ydf_beta)]
fibre_betas = [
        _b + cfbg_L * b for _b, b in zip(fibre_betas, pm_beta)]

# Define compressor dispersion coefficients. I have no idea if these are
# representative, but that doesn't change the method here, which is good.
# Usually, these would be calculated using analytic expressions for the
# compressor design, or raytracing in, for example, Zemax, for more complex
# compressor designs (e.g., where prisming and geometric effects from
# transmission grating substrates needs to be taken into account).
compressor_betas = [b * -11.56 for b in fibre_betas]

cfbg_betas = [
        (-1*comp_b - fibre_b)
        for comp_b, fibre_b in zip(compressor_betas, fibre_betas)]


###############################################################################
#                   Instantiate the compressor and cFBG                       #
###############################################################################
compressor = bc.component(0.25, 200e-9, g.lambda_c, 1, 0, 0, g, crosstalk,
                          order=5, beta_list=compressor_betas)
cfbg = bc.fibre_component(
    g, pm_pigtail, pm_pigtail, cfbg_L, 50e-9, g.lambda_c, 1, 0, 0, crosstalk,
    order=5, beta_list=cfbg_betas)


###############################################################################
#                 Instantiate the amplifier assembly                          #
###############################################################################
components = [circulator_1_to_2, cfbg, circulator_2_to_3, wdm, ydf, compressor]
amp = oa.sm_fibre_amplifier(
        g, components, high_res_sampling=20, plot=True, name='amp',
        verbose=True)

p = amp.simulate(p)
spw.matplotlib_gallery.launch_plot(plot_dicts=[amp.plot_dict])
