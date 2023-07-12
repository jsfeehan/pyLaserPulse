#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 09:22:08 2023

@author: james feehan

Module of classes for branded bulk components.
"""

import numpy as np

from pyLaserPulse.data import paths
import pyLaserPulse.base_components as bc
import pyLaserPulse.utils as utils


class half_wave_plate(bc.component):
    """
    Zero-order half wave plate. Assumes perfect half-wave retardation for all
    wavelengths.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    lambda_c : float
        Central wavelength of the transmission window in m
    angle : float
        Angle with respect to the x-axis.
    """
    def __init__(self, grid, lambda_c, angle):
        loss = 0.01
        trans_bw = 150e-9
        epsilon = -1
        beamsplitting = 0
        crosstalk = 0
        super().__init__(
            loss, trans_bw, lambda_c, epsilon, angle, beamsplitting, grid,
            crosstalk, output_coupler=False)


class quarter_wave_plate(bc.component):
    """
    Zero-order quarter wave plate. Assumes perfect quarter-wave retardation for
    all wavelengths.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    lambda_c : float
        Central wavelength of the transmission window in m
    angle : float
        Angle with respect to the x-axis.
    """
    def __init__(self, grid, lambda_c, angle):
        loss = 0.01
        trans_bw = 150e-9
        epsilon = 0 + 1j
        beamsplitting = 0
        crosstalk = 0
        super().__init__(
            loss, trans_bw, lambda_c, epsilon, angle, beamsplitting, grid,
            crosstalk, output_coupler=False)


class Thorlabs_broadband_PBS(bc.component):
    """
    PBS with parameters roughly based on the Thorlabs broadband PBS range.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    lambda_c : float
        Central wavelength of the transmission window in m
    output_coupler : bool
        If True, this component behaves like an output coupler or tap.
    """
    def __init__(self, grid, lambda_c, output_coupler):
        loss = 0.075
        trans_bw = 150e-9
        epsilon = 2e-3
        angle = 0
        beamsplitting = 1
        crosstalk = 1e-3
        super().__init__(
            loss, trans_bw, lambda_c, epsilon, angle, beamsplitting, grid,
            crosstalk, output_coupler=output_coupler,
            coupler_type="polarization")


class Thorlabs_Laserline_bandpass(bc.component):
    """
    Bandpass filter with parameters roughly based on the Thorlabs NIR/Laserline
    bandpass range.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    transmission_bandwidth : float
        Transmission window FWHM in m.
    lambda_c : float
        Central wavelength of the transmission window in m
    output_coupler : bool
        If True, this component behaves like an output coupler or tap.
    """
    def __init__(self, grid, transmission_bandwidth, lambda_c):
        loss = 0.25
        epsilon = 1
        angle = 0
        beamsplitting = 0
        crosstalk = 1e-3
        super().__init__(
            loss, transmission_bandwidth, lambda_c, epsilon, angle,
            beamsplitting, grid, crosstalk, output_coupler=False)


class Andover_155FS10_25_bandpass(bc.component):
    """
    Bandpass filter with parameters based on the Andover 155FS10-25 filer.
    Transmission bandwidth is 10 nm.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    peak_transmission : float
        Peak of transmission window. The measured data from the Andover website
        (component_loss_profiles/Andover_155FS10_25_bandpass) is usually
        used, but some papers suggest much higher peak transmission is
        possible. The manufacturer's specification is 0.324244.
    """
    def __init__(self, grid, peak_transmission=0.324244):
        loss = 0  # measured profile interpolated from file
        epsilon = 1
        angle = 0
        beamsplitting = 0
        crosstalk = 1e-3
        trans_bw = 1e-9
        super().__init__(
            loss, trans_bw, grid.lambda_c, epsilon, angle, beamsplitting, grid,
            crosstalk, output_coupler=False)
        loss_data = utils.interpolate_data_from_file(
            paths.components.loss_spectra.Andover_155FS10_25_bandpass,
            grid.lambda_window, 1e-9, 1, interp_kind='linear')
        loss_data[loss_data < 0] = 0
        loss_data = peak_transmission * loss_data / np.amax(loss_data)
        self.transmission_spectrum = utils.fftshift(loss_data)


class AA_Optoelectronic_AOM_MT200_A02_980_1100(bc.pulse_picker):
    """
    AA Optoelectronic AOM MT200-A02-xx, 200 MHz.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    time_open : float
        Time in s that the pulse picker is 'open', i.e., lets light through
        Cannot be less than 20 ns (specified rise time of 10 ns).
    rate_reduction_factor : int
        Ratio of the input and output repetition rates of the pulse picker.
    input_rep_rate : float
        Repetition rate of the seed laser before the pulse picker.

    Notes
    -----
    The diffraction efficiency of free-space AOMs is dependent on the wavelength
    and beam diameter. The minimum specified diffraction efficiency of 80% is
    used in this model.
    """
    def __init__(self, grid, time_open, rate_reduction_factor, input_rep_rate):
        if time_open >= 20e-9:
            loss = 0.98 * 0.8  # 98 % transmission, 80 % diffraction efficiency
            lambda_c = 1040e-9
            trans_bw = 120e-9  # specified for 980 - 1100 nm
            epsilon = 1
            super().__init__(
                loss, trans_bw, lambda_c, epsilon, 0, 0, grid, 1e-5, time_open,
                rate_reduction_factor, input_rep_rate)
        else:
            raise ValueError(
                "The AA Optoelectronic MT200-A0.2-xx 200 MHz AOM has a "
                "specified rise time of 10 ns. Parameter time_open cannot be "
                "less than 20 ns.")
