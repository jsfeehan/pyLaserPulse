#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 20:39:13 2020

@author: james feehan

Starting pulse definition
"""


import os
from abc import ABC, abstractmethod
import numpy as np
import scipy.constants as const
from itertools import combinations

import pyLaserPulse.utils as utils


def complex_first_order_degree_of_coherence(
        grid, pulse_objects, decimate=True):
    """
    Calculate the complex first-order degree of coherence for all fields in an
    iterable of pulse objects.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    pulse_objects : iterable (list, tuple, etc) of pulse objects.
    decimate : bool
        The complex first-order degree of coherence calculation is memory
        intensive, and it is rare that very high resolution data is required.
        It is recommended that decimate == True, which will result in the
        coherence data set having a lower wavelength-domain resolution (max of
        2^12 points).

    Returns
    -------
    CFODC : numpy array
        complex first-order degree of coherence for both polarization axis
        shape (2, n_points), where n_points is set by either grid.points, or
        the decimation (max 2^12).
    lw : numpy array
        Wavelength grid for the complex first-order degree of coherence.
        This is the decimated grid.lambda_window, or equal to
        grid.lambda_window if no decimation is used.
    """
    spectra = [utils.fft(po.field) for po in pulse_objects]
    if grid.points > 2**12 and decimate:
        decimation = int(grid.points / 2**12)
        spectra = [s[:, ::decimation] for s in spectra]
        lw = grid.lambda_window[::decimation]
    else:
        lw = grid.lambda_window
    spectra = np.asarray(spectra)

    # num_spectra choose 2
    num_spectra = len(pulse_objects)
    index = np.linspace(0, num_spectra-1, num_spectra)
    index = np.array(list(combinations(index, 2)), dtype=int)

    ensemble = spectra[index, :, :]
    numerator = np.mean(
        np.conjugate(ensemble[:, 0, :, :]) * ensemble[:, 1, :, :], 0)
    denominator = np.sqrt(
        np.mean(np.abs(ensemble[:, 0, :, :])**2, 0)
        * np.mean(np.abs(ensemble[:, 1, :, :])**2, 0))
    CFODC = utils.fftshift(np.abs(numerator / denominator), axes=1)
    return CFODC, lw


class _pulse_base(ABC):
    """
    Abstract base for all pulses.
    """

    def __init__(self, grid, repetition_rate, high_res_sampling=False,
                 save_dir=None, initial_delay=0):
        """
        Parameters
        ----------
        grid : pyLaserPulse.grid.grid object
        repetition_rate : float
            Repetition rate of the laser.
        high_res_sampling : bool
            If True, propagation information is saved at intervals throughout
            the propagation.
        save_dir : NoneType or string
            Directory to which data will be saved.
        initial_delay : float
            Starting point of the pulse along the time axis in seconds.
            E.g., if initial_delay = 0, the pulse will be centred at 0 ps in
            the time window. If initial_delay = -1e-12, the pulse will be
            centred at -1 ps in the time window.
            This feature is useful if a large amount of asymmetrical broadening
            is expected in the time domain (e.g., when pumping close to the
            zero-dispersion wavelength in supercontinuum generation).
        """
        self.repetition_rate = repetition_rate
        self.pulse_spacing = 1 / self.repetition_rate
        self.initial_delay = initial_delay

        # Make sure that the ASE emitted within the time window of the pulse
        # is scaled accurately.
        self.ASE_scaling = grid.t_range * self.repetition_rate
        if self.pulse_spacing < grid.t_range:
            # ValueError:
            # Necessary to prevent unphysical scaling of the quantum noise.
            # This is done for the:
            #     signal (see self.add_OPPM_noise)
            #     co- and counter-propagating signals in active fibres
            # and ensures that noise-seeded nonlinear effects (signal), ASE and
            # residual pump (co- and counter-) are estimated accurately.
            raise ValueError(
                "The pulse-to-pulse time interval cannot be less than"
                " the span of the temporal grid.\n Please either decrease"
                " the repetition rate or widen grid.time_window.")

        # Containers for fields sampled during the propagation, at output
        # couplers or taps, etc.
        self.output_samples = []
        self.field_samples = []
        self.output = []

        # Containers for high-resolution field sampling and useful calculated
        # parameters.
        self.field = np.array((), dtype=np.complex128)
        self.high_res_samples = high_res_sampling
        self.num_samples = None
        self.high_res_field_samples = []
        self.high_res_rep_rate_samples = []  # Used later for av. power
        self.high_res_field_sample_points = []
        self.high_res_B_integral_samples = []
        self.save_dir = save_dir
        self.autocorrelation = None
        self.trans_lim_autocorrelation = None
        self.dz = 1e-9  # Starting propagation step size.

    @abstractmethod
    def make_pulse(self):
        """
        Method which defines the pulse (self.field)
        """
        raise NotImplementedError

    def _roll_along_time_axis(self, grid):
        """
        Roll the pulse along the time axis by amount initial_delay seconds.

        Parameters
        ----------
        grid : pyLaserPulse.grid.grid object

        Notes
        -----
        This method is useful if a large amount of asymmetrical broadening is
        expected in the time domain (e.g., when pumping close to the
        zero-dispersion wavelength in supercontinuum generation).
        """
        roll_idx, _ = utils.find_nearest(self.initial_delay, grid.time_window)
        roll_idx -= grid.midpoint
        self.field = np.roll(self.field, roll_idx)

    def add_OPPM_noise(self, grid, topup=False, noise_seed=None):
        """
        Add one photon per mode quantum noise.
        if topup, only adds OPPM noise where the spectrum is less than
        np.abs(OPPM)**2.

        Parameters
        ----------
        grid : pyLaserPulse.grid.grid object
        topup : nool
            Adds OPPM noise to the full spectrum if False.
            Add OPPM noise to the spectrum only where spectrum is less than the
            OPPM energy if True.
        noise_seed : numpy array of shape p.field.shape
            Ranomdly generated seed for the OPPM noise.

        Notes
        -----
        R. Paschotta, ''Noise of mode-locked lasers (Part I)'', Applied
        Physics B 79(2), https://doi.org/10.1007/s00340-004-1547-x

        P. Drummond et al., ''Quantum noise in optical fibers. I.
        Stochastic equations'', J. Opt. Soc. Am. B 18, 139 (2001)
        """
        n = np.size(self.field)
        uniform_random = None

        if noise_seed is None:
            uniform_random = np.random.rand(n).reshape(self.field.shape)
        else:
            uniform_random = noise_seed

        phase = np.exp(-2j * np.pi * uniform_random)  # [0; 2*pi)

        # Shot noise
        A_OPPM = np.sqrt(const.hbar * grid.omega_window / (4 * grid.df))
        A_OPPM = A_OPPM * phase

        # Need to account for the size of the time window with respect to the
        # temporal spacing between the pulses. Without this step, the same
        # pulse in the same time window can produce different amounts of, for
        # example, Raman depending on self.repetition_rate.
        # A_OPPM *= np.sqrt(grid.t_range * self.repetition_rate)
        if topup:
            spec = utils.fft(self.field, axis=-1)
            abs_spec = spec.real**2 + spec.imag**2
            abs_A_OPPM = utils.ifftshift(A_OPPM.real**2 + A_OPPM.imag**2)
            idx = abs_spec * grid.dt**2 / (2 * np.pi) < abs_A_OPPM
            spec[idx] += utils.ifftshift(A_OPPM, axes=-1)[idx] \
                * grid.dOmega / np.sqrt(2 * np.pi)
            self.field = utils.ifft(spec, axis=-1)
        else:
            self.field += \
                utils.ifft(utils.ifftshift(A_OPPM, axes=-1), axis=-1) \
                * grid.dOmega / np.sqrt(2 * np.pi)

    def change_repetition_rate(self, grid, new_rep_rate):
        """
        Change the repetition rate of the pulse train.

        Parameters
        ----------
        grid : pyLaserPulse.grid.grid object
        new_rep_rate : float
            New repetition rate
        """
        self.repetition_rate = new_rep_rate
        self.ASM_scaling = grid.t_range * new_rep_rate

    def get_ESD_and_PSD(self, grid, field):
        """
        Calculate the energy spectral density and the power spectral density
        from self.field.
        ESD in J/m.
        PSD in mW/nm.

        Parameters
        ----------
        grid : pyLaserPulse.grid.grid object
        field : numpy array or list
            Can be self.field or self.output
        """
        spectra = utils.fftshift(utils.fft(field, axis=-1),
                                 axes=-1)
        spectra *= grid.dt / np.sqrt(2 * np.pi)  # Scale FFT
        self.energy_spectral_density \
            = np.abs(spectra)**2 * 2 * np.pi * const.c / grid.lambda_window**2
        self.power_spectral_density = \
            self.energy_spectral_density * self.repetition_rate
        self.power_spectral_density *= 1e-6  # conversion to mW/nm

    def get_ESD_and_PSD_from_spectrum(self, grid, spectrum):
        """
        Calculate the energy spectral density and the power spectral density
        from spectrum.
        ESD in J/m.
        PSD in mW/nm

        Parameters
        ----------
        grid : pyLaserPulse.grid.grid object.
        spectrum : numpy array of spectral data. NOT COMPLEX.
        """
        self.energy_spectral_density, self.power_spectral_density = \
            utils.get_ESD_and_PSD(grid.lambda_window, spectrum,
                                  self.repetition_rate)

    def get_ESD_and_PSD_from_high_res_field_samples(self, grid):
        """
        Calculate the energy spectral density and the power spectral density
        from self.high_res_field_samples. The PSD is normalized to dBm/nm.

        Parameters
        ----------
        grid : pyLaserPulse.grid.grid object

        Notes
        -----
        Creates members high_res_ESD_samples and high_res_PSD_samples, which
        are integrated over polarization.
        """
        hrfs = np.asarray(self.high_res_field_samples)
        spectra = utils.fftshift(utils.fft(hrfs, axis=-1), axes=-1)
        spectra *= grid.dt / np.sqrt(2 * np.pi)
        self.high_res_ESD_samples = np.abs(spectra)**2 * 2 * np.pi * const.c \
            / grid.lambda_window**2
        self.high_res_ESD_samples = np.sum(self.high_res_ESD_samples, axis=1)
        self.high_res_PSD_samples = \
            self.high_res_ESD_samples.T * self.high_res_rep_rate_samples
        self.high_res_PSD_samples = 1e-6 * self.high_res_PSD_samples.T

    def get_ESD_and_PSD_from_output_samples(self, grid):
        """
        Calculate the energy spectral density and power spectral density from
        self.output_samples. The PSD is normalized to dBm/nm.

        Parameters
        ----------
        grid : pyLaserPulse.grid.grid object

        Notes
        -----
        Creates members output_ESD_samples and output_PSD_samples, which are
        integrated over polarization.
        """
        ops = np.asarray(self.output_samples)
        spectra = utils.fftshift(utils.fft(ops, axis=-1), axes=-1)
        spectra *= grid.dt / np.sqrt(2 * np.pi)
        self.output_ESD_samples = np.abs(spectra)**2 * 2 * np.pi * const.c \
            / grid.lambda_window**2
        self.output_ESD_samples = np.sum(self.output_ESD_samples, axis=1)
        self.output_PSD_samples = self.output_ESD_samples * self.repetition_rate
        self.output_PSD_samples *= 1e-6

    def get_photon_spectrum(self, grid, field):
        """
        Calculate the number of photons in each frequency bin of the spectrum.

        Parameters
        ----------
        grid : pyLaserPulse.grid.grid object.
        field : numpy array.
            Can be self.field or self.output.
        """
        spectra = utils.fftshift(utils.fft(field, axis=-1),
                                 axes=-1)
        spectra *= grid.dt / np.sqrt(2 * np.pi)
        abs_spectra = spectra.real**2 + spectra.imag**2
        self.photon_spectrum = abs_spectra / grid.energy_window
        self.photon_spectrum *= \
            np.gradient(grid.omega_window)[None, :].repeat(2, axis=0)
        self.photon_spectrum = utils.ifftshift(self.photon_spectrum,
                                               axes=-1)
        if not np.any(np.isnan(self.photon_spectrum)):
            self.photon_spectrum = self.photon_spectrum.astype(int)
        self.photon_spectrum[self.photon_spectrum < 1] = 1

    def get_transform_limit(self, field):
        """
        Calculate the transform limited pulse associetated with field.

        Parameters
        ----------
        field : numpy array
            Can be self.field or self.output.
        """
        abs_spec = np.abs(utils.fft(field, axis=-1))
        self.transform_limit = utils.ifftshift(utils.ifft(abs_spec, axis=-1),
                                               axes=-1)
        self.transform_limit = np.sum(np.abs(self.transform_limit)**2, axis=0)

    def get_autocorrelation(self, field, polarization_attenuation):
        """
        Calculate the intensity autocorrelation of the pulse and of the
        transform limit.

        Parameters
        ----------
        field: numpy array
            Can be self.field or self.output
        polarization_attenuation : float
            How much to attenuate y-polarzed field component. Accounts for
            things like phasematching of the AC nonlinear process.
        """
        p = np.abs(field[0, :])**2 + np.abs(polarization_attenuation
                                            * field[1, :])**2
        self.autocorrelation = np.correlate(p, p, mode='same')
        self.get_transform_limit(field)
        self.trans_lim_autocorrelation = np.correlate(self.transform_limit,
                                                      self.transform_limit,
                                                      mode='same')
        self.autocorrelation /= np.amax(self.trans_lim_autocorrelation)
        self.trans_lim_autocorrelation \
            /= np.amax(self.trans_lim_autocorrelation)

    def get_energy_and_average_power(self, grid, field):
        """
        Calculate the pulse energy and average power of the laser beam.

        Parameters
        ----------
        grid : pyLaserPulse.grid.grid object
        field : numpy array
            Can be self.field or self.output
        """
        self.pulse_energy = None

        # Logic for cavities which have multiple output couplers.
        # Convert field to array if list. Transparent if array already.
        field = np.asarray(field)
        if len(np.shape(field)) == 2:
            # Single output coupler
            self.pulse_energy = np.sum(np.abs(field)**2) * grid.dt
        else:
            # Multiple output couplers
            self.pulse_energy = np.sum(np.sum(np.abs(field)**2, axis=1),
                                       axis=1) * grid.dt
        self.average_power = self.pulse_energy * self.repetition_rate

    def get_chirp(self, grid, field):
        """
        Calculate the chirp of field.

        Parameters
        ----------
        grid : pyLaerPulse.grid.grid object
        field: numpy array
            Can be self.field or self.output
        """
        phase = np.unwrap(np.angle(field), axis=1)
        self.chirp = 2 * np.pi * const.c / (
            grid.omega_c + np.gradient(phase, grid.dt, axis=1))

    def roundtrip_reset(self, clear_output=False):
        """
        Clear member variables which need to be reset with each round trip.

        Parameters
        ----------
        clear_output : bool
            If True, sets self.output = [].

        Notes
        -----
        This method is mostly inteded for oscillator simulations.
        """
        if clear_output:
            self.output = []
        self.high_res_field_samples = []
        self.high_res_field_sample_points = []
        self.high_res_PSD_samples = []
        self.high_res_rep_rate_samples = []
        self.high_res_B_integral_samples = []

    def update_high_res_samples(
            self, field_samples, B_samples, sample_points):
        """
        Append to:
            self.high_res_field_samples
            self.high_res_rep_rate_samples
            self.high_res_B_integral_samples
            self.high_res_field_sample_points

        Parameters
        ----------
        field_samples : list
            Field sampled as a function of propagation distance
        B_samples : list
            B-integral sampled as a function of propagation distance
        sample_points : list
            Points at which the samples were taken.
        """
        rep_rate_samples = [self.repetition_rate] * len(sample_points)
        self.high_res_rep_rate_samples += rep_rate_samples
        self.high_res_field_samples += field_samples
        self.high_res_B_integral_samples += B_samples
        self.high_res_field_sample_points += sample_points

    def save(self, directory):
        """
        Save the pulse data to file.

        Parameters
        ----------
        directory : string
            Directory in which the data file is saved.

        Notes
        -----
        The data is saved using the numpy.savez method and can be accessed
        using the numpy.load method, which returns a dictionary containing all
        of the data saved by this method. E.g.,:
        pulse_data = numpy.load('/path/pulse.npz')
        field = pulse_data['field']

        Dictionary keys:
        'field' : pulse field
        'power_spectral_density' : power spectral density in mW / nm
        'energy_spectral_density' : energy spectral density in J / m
        'repetition_rate' : repetition rate in Hz
        'sample_points' : points along the fibre at which samples are taken
        'field_samples' : the pulse field sampled at sample_points
        'rep_rate_samples' : the pulse repetition rate sampled at sample_points
        'B_integral_samples' : the B integral sampled at sample_points
        """
        np.savez(directory + "pulse.npz",
                 field=self.field, output=self.output,
                 power_spectral_density=self.power_spectral_density,
                 energy_spectral_density=self.energy_spectral_density,
                 repetition_rate=self.repetition_rate,
                 sample_points=np.cumsum(self.high_res_field_sample_points),
                 field_samples=self.high_res_field_samples,
                 rep_rate_samples=self.high_res_rep_rate_samples,
                 B_integral_samples=self.high_res_B_integral_samples)


class pulse(_pulse_base):
    """
    Laser pulse class.
    """

    def __init__(self, duration, P_0, pulse_shape, repetition_rate, grid,
                 order=2, chirp=0, quantum_noise_seed=None,
                 high_res_sampling=False, save_dir=None, initial_delay=0):
        """
        Parameters
        ----------
        duration : float
            FWHM pulse duration.
        P_0 : list
            [peak_power_x, peak_power_y]
        pulse_shape : string
            "Gauss" or "sech"
        repetition_rate : float
            Repetition rate of the laser.
        grid : pyLaserPulse.grid.grid object
        order : even integer
            If pulse_shape == 'Gauss', this is the supergaussian order (>= 1)
        chirp : float
            initial phase applied to the pulse in the time domain. This can be
            used to increase the pulse bandwidth for a given input duration.
            For a Gaussian pulse with chirp_parameter = C:
                field = exp(2log(2) * (1 - iC) * (T / tau)**2)
            For a sech pulse with chirp_parameter = C:
                field = (1 / cosh(t / tau))**(1 - iC)
        quantum_noise_seed : numpy array or NoneType
            Custom starting quantum noise. If NoneType, default one photon per
            mode is used.
        high_res_sampling : bool
            If True, propagation information is saved at intervals throughout
            the propagation.
        save_dir : NoneType or string
            Directory to which data will be saved.
        initial_delay : float
            Starting point of the pulse along the time axis in seconds.
            E.g., if initial_delay = 0, the pulse will be centred at 0 ps in
            the time window. If initial_delay = -1e-12, the pulse will be
            centred at -1 ps in the time window.
            This feature is useful if a large amount of asymmetrical broadening
            is expected in the time domain (e.g., when pumping close to the
            zero-dispersion wavelength in supercontinuum generation).
        """
        super().__init__(
            grid, repetition_rate, high_res_sampling=high_res_sampling,
            save_dir=save_dir, initial_delay=initial_delay)
        self.make_pulse(
            grid, pulse_shape, P_0, duration, order=order, chirp=chirp)
        self.add_OPPM_noise(grid, noise_seed=quantum_noise_seed)
        self.get_ESD_and_PSD(grid, self.field)
        self.get_photon_spectrum(grid, self.field)
        self.get_transform_limit(self.field)
        self.get_energy_and_average_power(grid, self.field)
        self._roll_along_time_axis(grid)

    def make_pulse(self, grid, pulse_shape, P_0, duration, order=2, chirp=0):
        """
        Make self.field.

        Parameters
        ----------
        grid : pyLaserPulse.grid.grid object
        pulse_shape : str
            'Gauss', 'sech'
        P_0 : list of floats
            Starting peak powers. [P_x, P_y]
        duration : float
            Starting pulse duration.
        order : even integer
            If pulse_shape == 'Gauss', this is the supergaussian order (>= 1)
        chirp : float
            initial phase applied to the pulse in the time domain. This can be
            used to increase the pulse bandwidth for a given input duration.
            For a Gaussian pulse with chirp_parameter = C:
                field = exp(2log(2) * (1 - iC) * (T / tau)**2)
            For a sech pulse with chirp_parameter = C:
                field = (1 / cosh(t / tau))**(1 - iC)
        """
        base_pulse = np.zeros(grid.points)

        # Must state dtype explicitly here. Numpy silently casts to double!
        if pulse_shape == "Gauss":
            if order < 2:
                raise ValueError(
                    "kwarg 'order' must be greater than or equal to 2.")

            if (order % 2) != 0:
                raise ValueError(
                    "kwarg 'order' must be even.")
            base_pulse = np.exp(
                -2 * np.log(2) * (grid.time_window / duration)**order,
                dtype=np.complex128)
            base_pulse *= np.exp(
                2j * np.log(2) * chirp * (grid.time_window / duration)**2)
        elif pulse_shape == "sech":
            base_pulse = 1 / np.power(np.cosh(grid.time_window /
                                              (duration / 1.76)),
                                      1 - 1j * chirp, dtype=complex)
            # Replace NaNs with zeros -- comes from complex power
            base_pulse[np.isnan(base_pulse)] = 0
        else:
            raise ValueError("pulse_shape must be 'Gauss' or 'sech'")

        # Numpy is C contiguous. Operations over columns are faster than
        # operations over rows in general. After broadcasting P_0,
        # transpose so that time domain data is in columns (axis 1) and
        # polarization information is in the rows (axis 0).
        self.field = np.sqrt(P_0) * base_pulse[:, None].repeat(2, axis=1)
        self.field = self.field.T


class pulse_from_measured_PSD(_pulse_base):
    """
    Laser pulse class.
    """

    def __init__(self, grid, spectrum_file, beta_2, repetition_rate,
                 spec_threshold, P_0, quantum_noise_seed=None,
                 high_res_sampling=False, save_dir=None, initial_delay=0):
        """
        Parameters
        ----------
        grid : pyLaserPulse.grid.grid object
        spectrum_file : str
            Absolute path to the file containing the spectral data.
            This file must have a header file_header_length lines long.
            The data must be comma or tab delimited, and must have
            wavelength data in nm in the first column, and the power
            spectral density in dBm/nm in the second column.
        beta_2 : float
            Second-order dispersion that is applied to the spectrum to give
            the correct pulse duration. This can be used to model propagation
            for pulses which are not transform limited.
        repetition_rate : float
            Repetition rate of the laser.
        spec_threshold : float
            Measurement threshold below which all power spectral density
            values are set to 0.
        P_0 : list
            [peak_power_x, peak_power_y]
        quantum_noise_seed : numpy array or NoneType
            Custom starting quantum noise. If NoneType, default one photon per
            mode is used.
        high_res_sampling : bool
            If True, propagation information is saved at intervals throughout
            the propagation.
        save_dir : NoneType or string
            Directory to which data will be saved.
        initial_delay : float
            Starting point of the pulse along the time axis in seconds.
            E.g., if initial_delay = 0, the pulse will be centred at 0 ps in
            the time window. If initial_delay = -1e-12, the pulse will be
            centred at -1 ps in the time window.
            This feature is useful if a large amount of asymmetrical broadening
            is expected in the time domain (e.g., when pumping close to the
            zero-dispersion wavelength in supercontinuum generation).
        """
        super().__init__(
            grid, repetition_rate, high_res_sampling=high_res_sampling,
            save_dir=save_dir, initial_delay=initial_delay)
        self.make_pulse(grid, spectrum_file, spec_threshold, beta_2, P_0)
        self.add_OPPM_noise(grid, noise_seed=quantum_noise_seed)
        self.get_ESD_and_PSD(grid, self.field)
        self.get_photon_spectrum(grid, self.field)
        self.get_transform_limit(self.field)
        self.get_energy_and_average_power(grid, self.field)
        self._roll_along_time_axis(grid)

    def make_pulse(self, grid, spectrum_file, spec_threshold,
                   beta_2, P_0):
        spec = utils.interpolate_data_from_file(
            spectrum_file, grid.lambda_window * 1e9, 1, 1, input_log=True,
            fill_value=0)

        spec -= spec_threshold
        spec[spec < 0] = 0  # 1e-20
        psf = np.exp(-2 * np.log(2) * grid.omega**2 / (10 * grid.dOmega**2))
        spec = np.abs(utils.fft_convolve(spec, psf))
        spec = np.sqrt(spec)  # convert to field
        spec = spec * utils.ifftshift(np.exp(1j * beta_2 * grid.omega**2 / 2))
        self.field = utils.ifftshift(utils.ifft(spec))
        self.field = self.field[:, None].repeat(2, axis=1)
        self.field /= np.sqrt(np.amax(np.abs(self.field)**2))
        self.field *= np.sqrt(P_0)
        self.field = self.field.T


class pulse_from_pyLaserPulse_simulation(_pulse_base):
    """
    Laser pulse class.

    Define a laser pulse using data output by a pyLaserPulse simulation.
    """

    def __init__(self, grid, data_directory, high_res_sampling=False,
                 save_dir=None, initial_delay=0):
        """
        Parameters
        ----------
        grid : pyLaserPulse.grid.grid object
        data_directory : str
            Absolute path of directory containing pulse.npz data file produced
            by a previous pyLaserPulse simulation.
        high_res_sampling : bool
            If True, propagation information is saved at intervals throughout
            the propagation.
        save_dir : NoneType or string
            Directory to which data will be saved.
        initial_delay : float
            Starting point of the pulse along the time axis in seconds.
            E.g., if initial_delay = 0, the pulse will be centred at 0 ps in
            the time window. If initial_delay = -1e-12, the pulse will be
            centred at -1 ps in the time window.
            This feature is useful if a large amount of asymmetrical broadening
            is expected in the time domain (e.g., when pumping close to the
            zero-dispersion wavelength in supercontinuum generation).
        """
        self.filename = str()
        self.pulse_data = dict()
        if not data_directory.endswith(os.sep):
            self.filename = data_directory + os.sep + 'pulse.npz'
            self.pulse_data = np.load(self.filename)
        else:
            self.filename = data_directory + 'pulse.npz'
            self.pulse_data = np.load(self.filename)
        rr = float(self.pulse_data['repetition_rate'])  # otherwise numpy array
        super().__init__(grid, rr, high_res_sampling=high_res_sampling,
                         save_dir=save_dir, initial_delay=initial_delay)
        self.make_pulse(self.pulse_data)
        self.get_ESD_and_PSD(grid, self.field)
        self.get_photon_spectrum(grid, self.field)
        self.get_transform_limit(self.field)
        self.get_energy_and_average_power(grid, self.field)
        self._roll_along_time_axis(grid)

    def make_pulse(self, pulse_data):
        self.field = pulse_data['field']
        self.output = pulse_data['output']

    def load_high_res_sample_data(self):
        """
        Load the high resolution sampling data from the data file used to
        define the pulse.

        These aren't loaded at instantiation because it can take some time if
        the data file is large.

        This method populates:
            self.high_res_B_integral_samples
            self.high_res_field_samples
            self.high_res_rep_rate_samples
            self.high_res_field_sample_points

        Notes
        -----
        self.high_res_rep_rate_samples is unlikely to be directly useful, and
        is generally only required for active fibre simulations and for proper
        scaling of the power spectral density as a function of
        self.high_res_field_sample_points.
        """
        try:
            self.high_res_B_integral_samples = \
                self.pulse_data['B_integral_samples']
            self.high_res_field_samples = self.pulse_data['field_samples']
            self.high_res_rep_rate_samples =self.pulse_data['rep_rate_samples']
            self.high_res_field_sample_points = self.pulse_data['sample_points']
        except KeyError as e:
            msg = self.filename + ' contains no high resolution sampling data'
            raise KeyError(msg) from e


class pulse_from_text_data(_pulse_base):
    """
    Laser pulse class.

    Define a laser pulse using data from a text file.
    """

    def __init__(self, grid, file, P_0, repetition_rate,
                 quantum_noise_seed=None, high_res_sampling=False,
                 save_dir=None, initial_delay=0):
        """
        Parameters
        ----------
        grid : pyLaserPulse.grid.grid object
        file : str
            Absolute path of the text file containing the pulse amplitude data.
            This must be tab delimited with the following format with no
            header:
            Time axis in ps, centred at zero \t pulse intensity data (arb).
        P_0 : list
            [peak_power_x, peak_power_y]
        repetition_rate : float
            Repetition rate of the laser.
        quantum_noise_seed : numpy array or NoneType
            Custom starting quantum noise. If NoneType, default one photon per
            mode is used.
        high_res_sampling : bool
            If True, propagation information is saved at intervals throughout
            the propagation.
        save_dir : NoneType or string
            Directory to which data will be saved.
        initial_delay : float
            Starting point of the pulse along the time axis in seconds.
            E.g., if initial_delay = 0, the pulse will be centred at 0 ps in
            the time window. If initial_delay = -1e-12, the pulse will be
            centred at -1 ps in the time window.
            This feature is useful if a large amount of asymmetrical broadening
            is expected in the time domain (e.g., when pumping close to the
            zero-dispersion wavelength in supercontinuum generation).
        """
        super().__init__(
            grid, repetition_rate, high_res_sampling=high_res_sampling,
            save_dir=save_dir, initial_delay=initial_delay)
        self.make_pulse(grid, file, P_0)
        self.add_OPPM_noise(grid, noise_seed=quantum_noise_seed)
        self.get_ESD_and_PSD(grid, self.field)
        self.get_photon_spectrum(grid, self.field)
        self.get_transform_limit(self.field)
        self.get_energy_and_average_power(grid, self.field)
        self._roll_along_time_axis(grid)

    def make_pulse(self, grid, file, P_0):
        raw_data = utils.interpolate_data_from_file(
            file, grid.time_window, 1, 1)
        tmp_pulse = np.zeros((grid.points, 2))
        max_raw_data = np.amax(raw_data)
        tmp_pulse[:, 0] = P_0[0] * raw_data / max_raw_data
        tmp_pulse[:, 1] = P_0[1] * raw_data / max_raw_data
        self.field = np.sqrt(tmp_pulse, dtype=np.complex128)
        self.field = self.field.T


class pulse_from_numpy_array(_pulse_base):
    """
    Laser pulse class.

    Define a laser pulse using a 1D numpy array containing the pulse shape in
    the time domain.
    """
    def __init__(self, grid, pulse_array, P_0, repetition_rate,
                 quantum_noise_seed=None, high_res_sampling=False,
                 save_dir=None, initial_delay=0):
        """
        Parameters
        ----------
        grid : pyLaserPulse.grid.grid object
        pulse_array : numpy array
            Shape of the intensity profile of the pulse in the time domain.
            Doesn't need to be normalized. Must be 1D and of len grid.points.
        P_0 : list
            [peak_power_x, peak_power_y]
        repetition_rate : float
            Repetition rate of the laser.
        quantum_noise_seed : numpy array or NoneType
            Custom starting quantum noise. If NoneType, default one photon per
            mode is used.
        high_res_sampling : bool
            If True, propagation information is saved at intervals throughout
            the propagation.
        save_dir : NoneType or string
            Directory to which data will be saved.
        initial_delay : float
            Starting point of the pulse along the time axis in seconds.
            E.g., if initial_delay = 0, the pulse will be centred at 0 ps in
            the time window. If initial_delay = -1e-12, the pulse will be
            centred at -1 ps in the time window.
            This feature is useful if a large amount of asymmetrical broadening
            is expected in the time domain (e.g., when pumping close to the
            zero-dispersion wavelength in supercontinuum generation).
        """
        super().__init__(
            grid, repetition_rate, high_res_sampling=high_res_sampling,
            save_dir=save_dir, initial_delay=initial_delay)
        self.make_pulse(grid, pulse_array, P_0)
        self.add_OPPM_noise(grid, noise_seed=quantum_noise_seed)
        self.get_ESD_and_PSD(grid, self.field)
        self.get_photon_spectrum(grid, self.field)
        self.get_transform_limit(self.field)
        self.get_energy_and_average_power(grid, self.field)
        self._roll_along_time_axis(grid)

    def make_pulse(self, grid, pulse_array, P_0):
        tmp_pulse = np.zeros((grid.points, 2))
        max_pulse_array = np.amax(pulse_array)
        tmp_pulse[:, 0] = P_0[0] * pulse_array / max_pulse_array
        tmp_pulse[:, 1] = P_0[1] * pulse_array / max_pulse_array
        self.field = np.sqrt(tmp_pulse, dtype=np.complex128)
        self.field = self.field.T
