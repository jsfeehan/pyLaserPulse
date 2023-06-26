#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 09:23:08 2023

@author: james feehan

Module of classes for branded fibre-coupled components.
"""

import numpy as np

from pyLaserPulse.data import paths
import pyLaserPulse.base_components as bc
import pyLaserPulse.catalogue_components.passive_fibres as pf
import pyLaserPulse.utils as utils


class JDSU_fused_976_1030_WDM(bc.fibre_component):
    """
    Fused fibre-coupled WDM with measured loss from a JDSU(?) component.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length_in : float
        Length of the input fibre in m.
    length_out : float
        Length of the output fibre in m.
    beat_length : float
        Polarization beat length in m.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    The component transmission spectrum formed part of the data set for:

    J. Feehan, et al., "Simulations and experiments showing the origin of
    multiwavelength mode locking in femtosecond, Yb-fiber lasers", JOSA B
    33(8), pp 1668-1676 (2016)

    Experimental work that led to this publication, including the measured loss
    profile used here, was carried out in lab 4103 of building 46, University
    of Southampton, circa 2014 (D. Richardson's group, J. Price's lab,
    optoelectronics research centre).

    Measured loss data file is component_loss_profiles/976_1030_fused_WDM.txt
    Pigtails assumed to be HI1060.
    """
    def __init__(
            self, grid, length_in, length_out, beat_length, verbose=False):
        tol = 1e-5
        loss = 0  # interpolated from data file
        trans_bw = np.inf  # given in loss data file
        epsilon = 1
        angle = 0
        beamsplitting = 0
        crosstalk = 1e-3
        input_fibre = pf.Corning_HI1060(grid, length_in, beat_length, tol)
        output_fibre = pf.Corning_HI1060(grid, length_out, beat_length, tol)
        super().__init__(
            grid, input_fibre, output_fibre, loss, trans_bw,
            grid.lambda_c, epsilon, angle, beamsplitting, crosstalk,
            verbose=verbose)

        wdm_data = utils.interpolate_data_from_file(
               paths.components.loss_spectra.fused_WDM_976_1030,
               grid.lambda_window, 1e-9, 1, interp_kind='linear')
        wdm_data[wdm_data > 1] = 1
        wdm_data[wdm_data < 1e-3] = 1e-3
        self.component.transmission_spectrum = utils.fftshift(wdm_data)


class Thorlabs_IO_J_1030(bc.fibre_component):
    """
    Thorlabs PM optical isolator with peak isolation and transmission at
    1030 nm. The fibre pigtails are PM980-XP.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length_in : float
        Input fibre length in m.
    length_out : float
        Output fibre length in m.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    Transmission spectrum digitized from Thorlabs data sheet for this
    component. Loss data was only available between 1010 nm and 1050 nm, so
    the loss profile beyond this region will be extrapolated linearly.

    Max operating power of 3 W.
    """
    def __init__(self, grid, length_in, length_out, verbose=False):
        tol = 1e-5
        loss = 0  # Loss profile interpolated from file
        trans_bw = np.inf  # Included in loss profile
        epsilon = 0.126
        angle = 0
        beamsplitting = 0
        crosstalk = 0.03
        input_fibre = pf.PM980_XP(grid, length_in, tol)
        output_fibre = pf.PM980_XP(grid, length_out, tol)
        super().__init__(
            grid, input_fibre, output_fibre, loss, trans_bw, grid.lambda_c,
            epsilon, angle, beamsplitting, crosstalk, verbose=verbose)
        iso_data = utils.interpolate_data_from_file(
            paths.components.loss_spectra.Thorlabs_IO_J_1030,
            grid.lambda_window, 1e-9, 1, interp_kind='linear')
        iso_data[iso_data > 1] = 1
        iso_data[iso_data < 0] = 0
        self.component.transmission_spectrum = utils.fftshift(iso_data)


class Opneti_1x2_PM_filter_coupler_500mW(bc.fibre_component):
    """
    Opneti 1x2 PM filter coupler. 500 mW max average power handling.
    PMFC-type. Fibre pigtails are PM980-XP.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length_in : float
        Input fibre length in m
    length_out : float
        Output fibre length in m
    lambda_c : float
        Central operating wavelength in m
    split_fraction : float
        Ratio of the beam splitting as a fraction. E.g., if 1% is tapped from
        the beam, split_fraction = 0.99, if 5% is tapped from the beam,
        split_fraction = 0.95, etc.
    output_coupler : bool
        If True, this component behaves like an output coupler or tap.
    verbose : bool
        Print information to terminal if True
    """
    def __init__(self, grid, length_in, length_out, lambda_c, split_fraction,
                 output_coupler=True, verbose=False):
        tol = 1e-5
        loss = 0.17
        trans_bw = 40e-9
        epsilon = 1
        angle = 0
        crosstalk = 0.01
        input_fibre = pf.PM980_XP(grid, length_in, tol)
        output_fibre = pf.PM980_XP(grid, length_out, tol)

        super().__init__(
            grid, input_fibre, output_fibre, loss, trans_bw, lambda_c, epsilon,
            angle, split_fraction, crosstalk, order=5,
            output_coupler=output_coupler, verbose=verbose)


class Opneti_1x2_PM_95_5_filter_coupler_500mW_fast_axis_blocked(
        bc.fibre_component):
    """
    Opneti 1x2 PM filter coupler. 500 mW max average power handling.
    Splitting ratio is nominally 95/5, measured unit has 94.45/5.55
    PMFC-type. Fibre pigtails are PM980-XP.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length_in : float
        Input fibre length in m
    length_out : float
        Output fibre length in m
    lambda_c : float
        Central operating wavelength in m
    split_fraction : float
        Ratio of the beam splitting as a fraction. E.g., if 1% is tapped from
        the beam, split_fraction = 0.99, if 5% is tapped from the beam,
        split_fraction = 0.95, etc.
    output_coupler : bool
        If True, this component behaves like an output coupler or tap.
    verbose : bool
        Print information to terminal if True
    """
    def __init__(self, grid, length_in, length_out, lambda_c,
                 output_coupler=True, verbose=False):
        tol = 1e-5
        loss = 0.17
        trans_bw = 40e-9
        epsilon = 0.08  # specified 22 dB fast-axis extinction
        angle = 0
        crosstalk = 0.0
        input_fibre = pf.PM980_XP(grid, length_in, tol)
        output_fibre = pf.PM980_XP(grid, length_out, tol)
        split_fraction = 0.9445

        super().__init__(
            grid, input_fibre, output_fibre, loss, trans_bw, lambda_c, epsilon,
            angle, split_fraction, crosstalk, order=5,
            output_coupler=output_coupler, verbose=verbose)

        tap_data = utils.interpolate_data_from_file(
            paths.components.loss_spectra.Opneti_95_5_PM_fibre_tap_fast_axis_blocked,
            grid.lambda_window, 1e-9, 1, interp_kind='linear')
        tap_data[tap_data > 1] = 1
        tap_data[tap_data < 1e-6] = 1e-6
        self.component.transmission_spectrum = utils.fftshift(tap_data)
        self.component.transmission_spectrum = \
            self.component.transmission_spectrum[None, :].repeat(2, axis=0)

class Opneti_PM_isolator_WDM_hybrid(bc.fibre_component):
    """
    Opneti PMIWDM-S-XXXXT/XXXXR-F/B-F-250-5-0.8-NE-2W
    Fibre pigtails are PM980-XP.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length_in : float
        Input fibre length in m
    length_out : float
        Output fibre length in m
    lambda_c : float
        Central operating wavelength in m
    verbose : bool
        Print information to terminal if True
    """
    def __init__(self, grid, length_in, length_out, lambda_c, verbose=False):
        tol = 1e-5
        loss = 0.52
        trans_bw = 40e-9
        epsilon = 0.126
        angle = 0
        beamsplitting = 0
        crosstalk = 0.001

        input_fibre = pf.PM980_XP(grid, length_in, tol)
        output_fibre = pf.PM980_XP(grid, length_out, tol)
        super().__init__(
            grid, input_fibre, output_fibre, loss, trans_bw, lambda_c,
            epsilon, angle, beamsplitting, crosstalk, order=5,
            verbose=verbose)


class Opneti_high_power_PM_isolator(bc.fibre_component):
    """
    Opneti high-power isolator (e.g., 1, 2, 5, 10, 20 W) with PM980 pigtails.
    Fast-axis blocking. Fibre pigtails are PM980-XP.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length_in : float
        Input fibre length in m
    length_out : float
        Output fibre length in m
    lambda_c : float
        Central operating wavelength in m
    verbose : bool
        Print information to terminal if True
    """
    def __init__(
            self, grid, length_in, length_out, lambda_c, verbose=False):
        tol = 1e-5
        loss = 0.2
        trans_bw = 150e-9
        epsilon = 0.056
        angle = 0
        beamsplitting = 0
        crosstalk = 0.01

        input_fibre = pf.PM980_XP(grid, length_in, tol)
        output_fibre = pf.PM980_XP(grid, length_out, tol)
        super().__init__(
            grid, input_fibre, output_fibre, loss, trans_bw, lambda_c, epsilon,
            angle, beamsplitting, crosstalk, order=5, verbose=verbose)

        iso_data = utils.interpolate_data_from_file(
            paths.components.loss_spectra.Opneti_high_power_isolator_HPMIS_1030_1_250_5,
            grid.lambda_window, 1e-9, 1, interp_kind='linear')
        iso_data[iso_data > 1] = 1
        iso_data[iso_data < 1e-6] = 1e-6
        self.component.transmission_spectrum = utils.fftshift(iso_data)
        self.component.transmission_spectrum = \
            self.component.transmission_spectrum[None, :].repeat(2, axis=0)


class Opneti_high_power_filter_WDM_1020_1080(bc.fibre_component):
    """
    Opneti high-power filter-based WDM with a transmission window of
    1060 +/- 40 nm. Fibre pigtails are PM980-XP.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length_in : float
        Input fibre length in m
    length_out : float
        Output fibre length in m
    verbose : bool
        Print information to terminal if True
    """
    def __init__(self, grid, length_in, length_out, verbose=False):
        tol = 1e-5
        loss = 0  # 0.17
        trans_bw = np.inf  # 60e-9
        epsilon = 1
        angle = 0
        beamsplitting = 0
        crosstalk = 0.01

        input_fibre = pf.PM980_XP(grid, length_in, tol)
        output_fibre = pf.PM980_XP(grid, length_out, tol)
        super().__init__(
            grid, input_fibre, output_fibre, loss, trans_bw, grid.lambda_c,
            epsilon, angle, beamsplitting, crosstalk, order=5, verbose=verbose)

        wdm_data = utils.interpolate_data_from_file(
            paths.components.loss_spectra.Opneti_microoptic_976_1030_wdm,
            grid.lambda_window, 1e-9, 1, interp_kind='linear',
            fill_value=(1.3e-03, 5.946e-01))
        wdm_data[wdm_data > 1] = 1
        wdm_data[wdm_data < 1e-6] = 1e-6
        self.component.transmission_spectrum = utils.fftshift(wdm_data)
        self.component.transmission_spectrum = \
            self.component.transmission_spectrum[None, :].repeat(2, axis=0)


class AFR_fast_axis_blocking_isolator_PMI_03_1_P_N_B_F(bc.fibre_component):
    """
    AFR low-power (50 mW average power max) fast-axis blocking isolator.
    Single stage. Fibre pigtails are PM980-XP.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length_in : float
        Input fibre length in m
    length_out : float
        Output fibre length in m
    lambda_c : float
        Central operating wavelength in m
    verbose : bool
        Print information to terminal if True
    """
    def __init__(
            self, grid, length_in, length_out, lambda_c, verbose=False):
        tol = 1e-5
        loss = 0.5
        trans_bw = 20e-9
        epsilon = 0.051
        angle = 0
        beamsplitting = 0
        crosstalk = 0.01

        input_fibre = pf.PM980_XP(grid, length_in, tol)
        output_fibre = pf.PM980_XP(grid, length_out, tol)
        super().__init__(
            grid, input_fibre, output_fibre, loss, trans_bw, lambda_c, epsilon,
            angle, beamsplitting, crosstalk, order=5, verbose=verbose)


class Opneti_fast_axis_blocking_isolator_PMIS_S_P_1030(bc.fibre_component):
    """
    Opneti low-power (50 mW average power max) fast-axis blocking isolator.
    Single stage. Fibre pigtails are PM980-XP

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length_in : float
        Input fibre length in m
    length_out : float
        Output fibre length in m
    lambda_c : float
        Central operating wavelength in m
    verbose : bool
        Print information to terminal if True
    """
    def __init__(
            self, grid, length_in, length_out, lambda_c, verbose=False):
        tol = 1e-5
        loss = 0.5
        trans_bw = 20e-9
        epsilon = 0.051
        angle = 0
        beamsplitting = 0
        crosstalk = 0.01

        input_fibre = pf.PM980_XP(grid, length_in, tol)
        output_fibre = pf.PM980_XP(grid, length_out, tol)
        super().__init__(
            grid, input_fibre, output_fibre, loss, trans_bw, lambda_c, epsilon,
            angle, beamsplitting, crosstalk, order=5, verbose=verbose)

        iso_data = utils.interpolate_data_from_file(
            paths.components.loss_spectra.Opneti_microoptic_isolator_PMIS_S_P_1030_F,
            grid.lambda_window, 1e-9, 1, interp_kind='linear')
        iso_data[iso_data > 1] = 1
        iso_data[iso_data < 1e-6] = 1e-6
        self.component.transmission_spectrum = utils.fftshift(iso_data)
        self.component.transmission_spectrum = \
            self.component.transmission_spectrum[None, :].repeat(2, axis=0)
