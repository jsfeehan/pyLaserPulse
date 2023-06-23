#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 22:28:09 2020

@author: james feehan

Class for pump light
"""

import numpy as np
import scipy.constants as const
import pyLaserPulse.utils as utils


class pump:
    """
    Class for pump light.
    """

    def __init__(self, bandwidth, lambda_c, energy, points=2**5,
                 lambda_lims=None, ASE_scaling=None, direction="co"):
        """
        Parameters
        ----------
        bandwidth : float
            Spectral width of the pump light in m
        lambda_c : float
            Central wavelength of the pump light in m
        energy : float
            Optical energy of the pump light.
            This is usually defined as the pump power divided by the repetition
            rate of the seed laser pulses being amplified.
        points : int
            Power of 2 (default is 2**5).
        lambda_lims : list
            [Lower, Upper] limits of the wavelength grid (in m).
            Defaults to None, which should be used if full ASE simulation is
            not required (co-pumping only).
        ASE_scaling : float
            Scale the ASE contribution in the Giles model
            by 1 - pulse_time_window * repetition_rate. This ensures correct
            scaling of the ASE power 'emitted' outside of the pulse temporal
            window. The pulse class has as scaling factor of
            pulse_time_window * repetition_rate).
        direction : str
            'co' or 'counter' for co-propagating and counter-propagating
            geometries, respectively.
        """
        self.points = points
        self.midpoint = int(self.points / 2)
        self.lambda_c = lambda_c
        self.bandwidth = bandwidth
        self.energy = energy  # starting energy
        axis = np.linspace(-1 * self.midpoint, (self.points - 1) / 2,
                           self.points)
        self.lambda_lims = lambda_lims
        self.ASE_scaling = ASE_scaling

        if lambda_lims is None:  # base wavelength grid on bandwidth
            lambda_range = None
            if self.bandwidth <= 10e-9:
                lambda_range = 10e-9
            else:
                lambda_range = self.bandwidth + 2e-9
            lambda_max = lambda_c + lambda_range / 2
            omega_range = 4 * np.pi * \
                (const.c / lambda_c - const.c / lambda_max)
            self.omega_c = 2 * np.pi * const.c / self.lambda_c
        elif lambda_lims[0] < lambda_c < lambda_lims[1]:
            omega_lims = [2 * np.pi * const.c /
                          lim for lim in lambda_lims[::-1]]
            omega_range = np.abs(np.diff(omega_lims))
            self.omega_c = np.average(omega_lims)
        else:
            raise ValueError(
                "The pump wavelength lies outside the grid wavelength"
                " range:\nlambda_c = %f nm,\nlambda_lims[0] = %f nm"
                "\nlambda_lims[1] = %f nm."
                % (1e9 * lambda_c, 1e9 * lambda_lims[0],
                   1e9 * lambda_lims[1]))

        self.ASE_scaling = ASE_scaling

        self.dOmega = omega_range / self.points
        self.omega_window = self.dOmega * axis + self.omega_c
        self.energy_window = const.hbar * self.omega_window
        self.lambda_window = 2 * np.pi * const.c / self.omega_window
        self.d_wl = np.gradient(-1 * self.lambda_window)

        # Without the following, pump spectrum is often just a single point.
        if self.bandwidth < 2 * np.max(self.d_wl):
            self.bandwidth = 2 * np.max(self.d_wl)

        # Assume top-hat spectral shape.
        self.spectrum = np.zeros((self.points))
        self.spectrum[np.abs(self.lambda_window - self.lambda_c)**2
                      < np.abs(self.bandwidth / 2)**2] = 1
        self.spectrum /= np.sum(self.spectrum * self.dOmega)
        self.spectrum *= energy
        # Polarization
        self.spectrum = 0.5 * self.spectrum[None, :].repeat(2, axis=0)

        self.direction = np.ones_like(self.spectrum)  # propagation step sign
        if direction == "counter":
            # This is the counter-propagated spectrum AT THE SIGNAL INPUT.
            self.propagated_spectrum = np.zeros_like(self.spectrum)
            self.direction *= -1

    def get_ESD_and_PSD(self, spectrum, repetition_rate):
        """
        Calculate the energy spectral density and the power spectral density
        of the pump light.

        Parameters
        ----------
        spectrum : numpy array
            The spectrum to convert to ESD or PSD
        repetition_rate : float
            Repetition rate of the seed laser.
            See pyLaserPulse.pulse.repetition_rate

        Returns
        -------
        numpy array
            spectrum converted to energy spectral density in J / m
        numpy array
            spectrum converted to power spectral density in mW / nm
        """
        energy_spectral_density, power_spectral_density = \
            utils.get_ESD_and_PSD(
                self.lambda_window, spectrum, repetition_rate)
        return energy_spectral_density, power_spectral_density
