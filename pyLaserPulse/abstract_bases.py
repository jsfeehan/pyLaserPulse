#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:26:17 2022

@author: james feehan

Module of abstract base classes for optical components.
"""


from abc import ABC, abstractmethod
import math
import numpy as np
import scipy.interpolate as interp
import scipy.constants as const

import pyLaserPulse.utils as utils
import pyLaserPulse.bessel_mode_solver as bms
import pyLaserPulse.pulse as pls
import pyLaserPulse.pump as pmp
import pyLaserPulse.exceptions as exc
# import pyLaserPulse.sys_info as si


class fibre_base(ABC):
    """
    Abstract type for passive fibres.

    Contains all methods and members required to simulate single-mode pulse
    propagation in passive optical fibre.

    self.get_signal_propagation_parameters will be specific to the structure
    of the fibre being simulated (e.g., PCF vs SMF) and is left as a 'blank'
    abstractmethod which needs to be overloaded by all derived classes.
    """

    def __init__(self, g, length, loss_file, Raman_file, beat_length, n2, fR,
                 tol, verbose=False):
        """
        Parameters
        ----------
        g : grid.
            pyLaserPulse.grid.grid object.
        length : float.
            Fibre length in m.
        loss_file : string.
            Absolute path to file containing fibre loss data. See
            pyLaserPulse.data.paths.materials.loss_spectra
        Raman_file : string.
            Absoltue path to file containing Raman response data.
            Recommend pyLaserPulse.data.paths.materials.Raman_profiles
        beat_length : float.
            Polarization beat length in m.
        n2 : float.
            Nonlinear index in m^2 / W.
        fR : float.
            Fraction contribution of delayed (Raman) effects to the nonlinear
            material response (e.g., fR = 0.18 for fused silica).
        tol : float.
            Propagation error tolerance for the conservation quantity error
            method propagation step sizing algorithm.
        verbose : bool
            Print information to terminal if True
        """
        self.grid = g
        self.L = length
        self.beat_length = beat_length * np.ones((self.grid.points))
        self.n2 = n2
        self.fR = fR
        self.tol = tol
        self.loss = utils.interpolate_data_from_file(
            loss_file, self.grid.lambda_window, 1e-6, 1e-3,
            interp_kind='linear', fill_value='extrapolate', input_log=False,
            return_log=True)
        self.Raman = utils.load_Raman(Raman_file, g.time_window, g.dt)
        self.verbose = verbose

    def make_verbose(self):
        """
        Change self.verbose to True
        Called by optical_assemblies. If the optical assembly verbosity is
        True, all component verbosities are also set to True.
        """
        self.verbose = True

    def make_silent(self):
        """
        Change self.verbose to False
        Called by optical_assemblies. If the optical assembly verbosity is
        False, all component verbosities are also set to False.
        """
        self.verbose = False

    @abstractmethod
    def get_propagation_parameters(cls):
        """
        Specific to each derived class. Must calculate the following:
        signal_ref_index, V, beta_2, D, effective_MFD, signal_mode_area

        Other 'nice to haves', e.g., NA, can also be added.
        """
        raise NotImplementedError

    @staticmethod
    def Petermann_II(V):
        """
        Petermann II method for ratio of mode and core diameters from V.

        Parameters
        ----------
        V : numpy array.
            V-number as a function of frequency.

        Returns
        -------
        numpy array.
            Ratio of mode and core diameters as a function of frequency.
        """
        p_II = 0.65 + (1.619 / V**(3 / 2)) + (2.879 / V**6) \
            - (0.016 + 1.561 / V**7)
        return p_II

    def _make_linear_operator(self):
        """
        Define self.linear_operator.
        """
        self.Taylors, beta = utils.get_Taylor_coeffs_from_beta2(
            self.beta_2, self.grid)
        self.linear_operator = 0.5 * self.loss + 1j * beta
        self._linear_operator_definition()

    def _linear_operator_definition(self):
        if self.linear_operator.shape[0] != 2:
            self.linear_operator = \
                self.linear_operator[None, :].repeat(2, axis=0)
        self.linear_operator[0, :] += -1j * self.beta_1 * self.grid.omega / 2
        self.linear_operator[1, :] += 1j * self.beta_1 * self.grid.omega / 2

        # Create new arrays holding dispersion data used for the propagation
        self.beta_2_Taylors = np.gradient(
            self.linear_operator.imag, self.grid.omega, edge_order=2, axis=1)
        self.beta_2_Taylors = np.gradient(
            self.beta_2_Taylors, self.grid.omega, edge_order=2, axis=1)
        self.D_Taylors = -2 * np.pi * const.c * self.beta_2_Taylors \
            / self.grid.lambda_window**2
        self.linear_operator = utils.fftshift(self.linear_operator, axes=-1)

    def override_dispersion_using_Taylor_coefficients(
            self, Taylor_coefficients):
        """
        Redefine the linear operator using Taylor coefficients for the
        dispersion instead of models based on the fibre geometry.

        Parameters
        ----------
        Taylor_coefficients : list of floats
            Taylor coefficients [beta_2, beta_3, ... beta_n] describing the
            fibre dispersion curve.
        """
        self.Taylors = Taylor_coefficients
        self.Taylors.insert(0, 0)  # 0th
        self.Taylors.insert(1, 0)  # 1st (handled by beat length)

        beta = utils.Taylor_expansion(self.Taylors, self.grid.omega)

        self.linear_operator = 0.5 * self.loss + 1j * beta
        self._linear_operator_definition()

    def _get_birefringence(self):
        """
        Defined self.birefringence using the effective index and polarization
        group velocity mismatch.

        Assumes that polarization mode group velocity mismatch does not vary
        with frequency.
        """
        self.birefringence = 2 * np.pi * const.c / \
            (self.grid.omega_window * self.beat_length)

    def _get_polarization_group_velocity_mismatch(self):
        """
        Defined self.beta_1, the polarization group velocity mismatch.
        """
        vg_x = const.c / self.signal_ref_index
        vg_y = const.c / (self.signal_ref_index - self.birefringence)
        self.beta_1 = (1 / vg_y) - (1 / vg_x)

    def _make_self_steepening_term(self):
        """
        Define self.self_steepening, the self-steepening term for the nonlinear
        operator.
        """
        self.self_steepening = -1j * self.gamma * \
            (1 + self.grid.omega / self.grid.omega_c)
        self.self_steepening = self.self_steepening[None, :].repeat(2, axis=0)
        self.self_steepening = utils.fftshift(self.self_steepening, axes=-1)

    def get_GNLSE_and_birefringence_parameters(self):
        """
        Define self.effective_MFD, self.signal_mode_area, self.gamma, and call
        self._get_birefringence, self._get_polarization_group_velocity_mismatch
        self._make_linear_operator, and self._make_self_steepening_term.

        Can only be called after get_signal_propagation_parameters.
        """
        p_II = self.Petermann_II(self.V)
        self.effective_MFD = self.core_diam * p_II
        self.signal_mode_area = np.pi * (self.effective_MFD / 2)**2
        self.gamma = self.n2 * (2 * np.pi / self.grid.lambda_c) \
            / self.signal_mode_area
        self.gamma = self.gamma[self.grid.midpoint]
        self._get_birefringence()
        self._get_polarization_group_velocity_mismatch()
        self._make_linear_operator()
        self._make_self_steepening_term()

    def _DFWM_phasematching(self, z):
        """
        Calculate the phasematching term for cross-polarization mixing (e.g.,
        DWFM, PMI, etc.)

        Parameters
        ----------
        z : float.
            Propagated distance in m (not total fibre length OR dz).

        Returns
        numpy array.
            Phasematching contribution as a function of frequency for each
            polarization axis.
        -------
        """
        pm = np.array((np.exp(-2j * np.pi * z / self.beat_length),
                       np.exp(2j * np.pi * z / self.beat_length)))
        return pm

    def _nonlinear_step(self, field, dz, pm):
        """
        Apply and return nonlinear contribution to RK4IP step

        Parameters
        ----------
        field : complex numpy array.
            Time-domain field distribution.
        dz : float.
            Step size in m.
        pm : complex numpy array.
            Cross-polarization phasematching term
            calculated using cross_polarization_phasematching(...)

        Returns
        -------
        complex numpy array.
            Contribution to the nonlinear step.
        """
        if np.any(np.isnan(field)):
            return np.ones_like(field) * np.nan
        else:
            # _r indicates switch of pol. axes, i.e., [x, y] --> [y, x]
            field_r = field[::-1, :]
            P = field.real**2 + field.imag**2
            P_r = P[::-1, :]
            P_fft = utils.fft(P)
            P_fft_r = P_fft[::-1, :]
            conjfield = np.conj(field)
            conjfield_r = conjfield[::-1, :]
            conjfield_pm = utils.ifft(utils.fft(conjfield) * pm)

            SPM_XPM_DFWM = field * (1. - self.fR) * (P + (2. / 3.) * P_r) \
                + (1. - self.fR) * field_r**2 * conjfield_pm / 3.
            Raman_SPM_XPM = self.fR * field * self.grid.dt * utils.ifft(
                (self.Raman[:, 0] + self.Raman[:, 1]) * P_fft + self.Raman[:, 0] * P_fft_r)
            Raman_DFWM = self.fR * field_r * self.grid.dt * \
                utils.ifft(0.5 * self.Raman[:, 1] * utils.fft(
                    field * conjfield_r + field_r * conjfield_pm))
            k = self.self_steepening * dz * utils.fft(
                SPM_XPM_DFWM + Raman_SPM_XPM + Raman_DFWM)
            return k

    def _RK4IP(self, phasematching, dz, field_spec):
        """
        Apply the linear and nonlinear contributions over the propagation step
        using the Runge-Kutte 4th-order interaction picture method.

        J. Hult, Journal of Lightwave Technology, 25(12) 2007.

        Loosely based on examples from https://freeopticsproject.org.

        Parameters
        ----------
        phasematching : complex numpy array.
            Phasematching for polarization mixing over dz.
        dz : float.
            Step size in m.
        field_spec : complex numpy array.
            Field spectrum.

        Returns
        -------
        complex numpy array
            The new frequency-domain field distribution after propagating
            distance dz.
        """
        if np.any(np.isnan(field_spec)):
            return np.ones_like(field_spec) * np.nan
        else:
            uu1 = utils.ifft(field_spec)
            half_step = np.exp(-0.5 * self.linear_operator * dz)

            uip = half_step * field_spec

            k1 = self._nonlinear_step(uu1, dz, phasematching)
            k1 *= half_step

            uu2 = utils.ifft(uip + 0.5 * k1)
            k2 = self._nonlinear_step(uu2, dz, phasematching)

            uu3 = utils.ifft(uip + 0.5 * k2)
            k3 = self._nonlinear_step(uu3, dz, phasematching)

            uu4 = utils.ifft(half_step * (uip + k3))
            k4 = self._nonlinear_step(uu4, dz, phasematching)

            return half_step * (uip + k1 / 6. + k2 / 3. + k3 / 3.) + k4 / 6.

    def _CQEM(self, photon_error, dz, distance, aux_spec, field_spec):
        """
        Conservation quantity error method (photon counting) for adaptive step
        sizing and for judging when the propagation error is too high to
        tolerate. If this is the case, repeat the propagation step and discard
        the most recent solution.

        A. Heidt, Journal of Lightwave Technology, 27(18) 2009.

        Paramters
        ---------
        photon_error : float.
            Error in photon count over propagation step.
        dz : float.
            Size of the previous propagation step.
        distance : float.
            Propagated distance.
        aux_spec : complex numpy array.
            Propagated field spectrum.
        field_spec : complex numpy array.
            Field spectrum before propagation.

        Returns
        -------
        float
            Updated propagation step size in m.
        distance
            Updated propagated distance in m.
        complex numpy array
            Updated field spectrum.
        """
        if np.any(np.isnan(aux_spec)):
            distance -= dz
            dz /= 10
        elif photon_error > 2 * self.tol:
            distance -= dz
            dz /= 2
        elif photon_error > self.tol:
            field_spec = aux_spec
            dz /= 2**0.2
        elif photon_error < 0.1 * self.tol:
            field_spec = aux_spec
            dz *= 2**0.2
        else:
            field_spec = aux_spec
        return dz, distance, field_spec

    def _CQEM_photon_count(self, field_spec):
        """
        Calculate the number of photons in a spectrum.

        Parameters
        ----------
        field_spec : complex numpy array.
            Field spectrum.
        dz : float.
            Propagation step size in m.

        Returns
        -------
        float
            Total number of photons in the pulse.
        """
        return np.sum((field_spec.real**2 + field_spec.imag**2)
                      / self.grid.omega_window_shift)

    def _CQEM_photon_count_with_loss(self, field_spec, dz):
        """
        Calculate the number of photons in a spectrum accounting for loss over
        the propagation step.

        Parameters
        ----------
        field_spec : complex numpy array.
            Field spectrum.
        dz : float.
            Propagation step size.

        Returns
        -------
        float
            Total number of photons in the pulse before application of
            propagation loss over step dz.
        """
        loss = np.exp(-dz * self.linear_operator.real)
        return np.sum((field_spec.real**2 + field_spec.imag**2)
                      * loss**2 / self.grid.omega_window_shift)

    def _propagation(
            self, dz, field, sampling=False, num_samples=2):
        """
        Apply passive linear and nonlinear operators iteratively over the fibre
        length and use the conservation quantity error method (photon counting)
        for adaptive step sizing.

        Parameters
        ----------
        dz : float.
            Propagation step size.
        field : complex numpy array.
            Time-domain field distribution.
        sampling : bool.
            Choose whether to sample the field during propagation.
            Default is False.
        num_samples : int.
            Number of samples to draw. Default is 2.

        Returns
        -------
        if sampling:
            complex numpy array
                Field in the time domain
            float
                Updated step size, dz (adjusted by conservation quantity error
                method).
            complex numpy array
                Fields sampled along the fibre.
            list
                Positions along the fibre at which the field samples were taken
        else:
            complex numpy array
                Field in the time domain
            float
                Updated step size, dz (adjusted by conservation quantity error
                method).

        Notes
        -----
        Sampling slows this function down quite a lot. This is because
        dz_updated remains small -- it is rescaled according to sampling
        conditions.
        (dz = propagated_distance % sample_interval results in small dz).
        """
        propagated_distance = 0

        if sampling:
            sample_count = 1  # Keep track of number of samples taken.
            field_samples = []
            dz_samples = []
            sample_interval = self.L / num_samples
            sample = False

        ufft = utils.fft(field, axis=-1)
        while propagated_distance < self.L:
            dz_updated = dz  # Returned so that pulse.dz can be updated.
            if propagated_distance + dz > self.L:
                dz = self.L - propagated_distance
            if sampling:
                if propagated_distance > sample_count * sample_interval:
                    dz = propagated_distance % sample_interval
                    if dz == 0:  # Set to tiny value to avoid infinite loop.
                        dz = 1e-80
                    sample = True

            phasematching = self._DFWM_phasematching(propagated_distance)

            start_photon_number = self._CQEM_photon_count_with_loss(ufft, dz)
            aux_ufft = self._RK4IP(phasematching, dz, ufft)
            finish_photon_number = self._CQEM_photon_count(aux_ufft)

            error = abs(start_photon_number - finish_photon_number) \
                / start_photon_number
            if sampling:
                if sample:
                    dz_samples.append(sample_interval)
                    field_samples.append(utils.ifft(ufft, axis=-1))
                    sample_count += 1
                    sample = False

            propagated_distance += dz
            # print(f'{propagated_distance-dz:.10}', end='\r', flush=True)
            if dz == 1e-80:  # Reset to sensible value after sample condition.
                dz = dz_updated
            else:
                dz, propagated_distance, ufft = self._CQEM(
                    error, dz, propagated_distance, aux_ufft, ufft)
            if np.any(np.isnan(ufft)):
                if sampling:
                    return (np.ones_like(ufft) * np.nan, dz_updated,
                            field_samples, dz_samples)
                else:
                    return np.ones_like(ufft) * np.nan, dz_updated
            if self.verbose:
                print("\t%.1f %%" % (100 * propagated_distance / self.L),
                      end='\r')

        if self.verbose:
            print("\n")
        if sampling:
            dz_samples.append(sample_interval)
            field_samples.append(utils.ifft(ufft, axis=-1))
            return (utils.ifft(ufft, axis=-1), dz_updated, field_samples,
                    dz_samples)
        else:
            return utils.ifft(ufft, axis=-1), dz_updated

    def _update_B_integral_samples(self, field_samples):
        """
        Update self.B_samples.

        Parameters
        ----------
        field_samples : complex numpy array
            Fields sampled along the fibre length.
        """
        fs = np.asarray(field_samples)
        P = np.sum(fs.real**2 + fs.imag**2, axis=1)
        coeff = 2 * np.pi * np.asarray(self.dz_samples) \
            / (self.grid.lambda_c * self.signal_mode_area[self.grid.midpoint])
        self.B_samples = coeff * np.cumsum(self.n2 * np.amax(P, axis=-1))

    def propagate(self, pulse):
        """
        Apply passive linear and nonlinear operators iteratively over the fibre
        length and use the conservation quantity error method (photon counting)
        for adaptive step sizing.

        Parameters
        ----------
        pulse : pulse class.
            Object of type pyLaserPulse.pulse.pulse.

        Returns
        -------
        pulse
            Object of type pyLaserPulse.pulse.pulse.
        """
        if pulse.high_res_samples:
            pulse.field, pulse.dz, samples, self.dz_samples \
                = self._propagation(
                    pulse.dz, pulse.field, sampling=pulse.high_res_samples,
                    num_samples=pulse.num_samples)
            self._update_B_integral_samples(samples)
            if len(pulse.high_res_B_integral_samples) != 0:
                cumulative_B = \
                    self.B_samples + pulse.high_res_B_integral_samples[-1]
                cumulative_B = cumulative_B.tolist()
            else:
                cumulative_B = self.B_samples.tolist()
            pulse.update_high_res_samples(
                samples, cumulative_B, self.dz_samples)
        else:
            pulse.field, pulse.dz \
                = self._propagation(pulse.dz, pulse.field)
        return pulse


class active_fibre_base(ABC):
    """
    Abstract type for active fibres.

    Contains all methods and members required to transform a passive fibre into
    an active fibre, which can be done using multiple inheritance as long as
    the method resolution order is:

    <class 'pyLaserPulse.abstract_bases.active_fibre_base'> followed by
    inheritance of type
    <class 'pyLaserPulse.abstract_bases.fibre_base'>, which can be from the
    base_components module, such as:
    <class 'pyLaserPulse.base_components.step_index_passive_fibre'> or
    <class 'pyLaserPulse.base_components.photonic_crystal_passive_fibre'>.
    """
    class stacker:
        """Class for storing stacked arrays used in full ASE simulations."""

        def __init__(self):
            pass

    def __init__(
            self, g, N_ions, cross_section_file, seed_rep_rate, pump_points,
            pump_wl_lims, Sellmeier_file, lifetime, cladding_pumping,
            time_domain_gain, boundary_conditions, verbose=True):
        """
        Parameters
        ----------
        g : grid.
            pyLaserPulse.grid.grid object.
        N_ions : float
            Rare-earth ion number density in m^-3
        cross_section_file : string.
            Absolute path to file containing emission and absorption cross
            section  data. See paths.fibres.cross_sections
        seed_rep_rate : float
            Repetition rate of the seed laser. See pulse.repetition_rate.
        pump_points : int
            Number of points in the pump/ASE spectrum window.
        pump_wl_lims : list
            [min_wavelength, max_wavelength] for the pump/ASE wavelength
            window in m.
        Sellmeier_file : string.
            Absolute path to file containing Sellmeier coefficients for the
            fibre material.
            See pyLaserPulse.data.paths.materials.Sellmeier_coefficients.
        lifetime : float
            Upper-state lifetime of the rare-earth dopant in s.
        cladding_pumping : bool
            True if cladding pumping, False if core pumping.
        time_domain_gain : bool
            True if mixed-domain gain operator (i.e., both frequency- and
            time-domain gain) is to be used. If False, only frequency-
            domain gain is simulated.
        boundary_conditions : dict
            Set the boundary conditions for resolving the evolution of the
            pump, signal, and ASE light in both directions through the fibre.
            The type of simulation -- i.e., single-pass or full boundary value
            solver -- is determined by this dictionary. Valid dictionary keys:
                co_pump_power : float
                    Power of the co-propagating pump in W
                co_pump_wavelength : float
                    Central wavelength of the co-propagating pump in m
                co_pump_bandwidth : float
                    Bandwidth of the co-propagating pump in m
                counter_pump_power : float
                    Power of the counter-propagating pump in W
                counter_pump_wavelength : float
                    Central wavelength of the counter-propagating pump in m
                counter_pump_bandwidth : float
                    Bandwidth of the counter-propagating pump in m
        verbose : bool
            Print information to terminal if True

        Notes
        -----
        If no counter-pump values are specified, then the propagation is
        completed in a single step and no effort is made to resolve counter-
        propagating signals (i.e., the boundary value problem is not solved).
        This allows for co-pumping only. Example:

        boundary_conditions = {
            'co_pump_power': 1, 'co_pump_wavelength': 976e-9,
            'co_pump_bandwidth': 1e-9}

        The boundary value problem is solved whenever dictionary key
        'counter_pump_power' is present, but it is not necessary to specify the
        wavelength or bandwidth of either the co- or counter-propagating pump
        if the power is zero. Here are a few examples of valid
        boundary_conditions dictionaries:

        1) Co-pumping with full boundary value solver:
            boundary_conditions = {
                'co_pump_power': 1, 'co_pump_wavelength': 976e-9,
                'co_pump_bandwidth': 1e-9, 'counter_pump_power': 0}

        2) Bidirectional pumping with full boundary value solver:
            boundary_conditions = {
                'co_pump_power': 1, 'co_pump_wavelength': 976e-9,
                'co_pump_bandwidth': 1e-9, 'counter_pump_power': 1,
                'counter_pump_wavelength': 976e-9,
                'counter_pump_bandwidth': 1e-9}

        3) Counter-pumping with full boundary value solver:
            boundary_conditions = {
                'co_pump_power': 0, 'counter_pump_power': 1,
                'counter_pump_wavelength': 976e-9,
                'counter_pump_bandwidth': 1e-9}
        """
        self.grid = g
        self.lifetime = lifetime
        self.time_domain_gain = time_domain_gain
        self.Sellmeier_file = Sellmeier_file

        self.boundary_conditions = boundary_conditions
        self._determine_propagation_method_from_boundaries()

        if self.boundary_value_solver:
            self.dz = 5e-3
            self.num_steps = int(np.ceil(self.L / self.dz))

        if self.time_domain_gain:
            self._propagation_func = \
                self._Euler_approximate_mixed_domain_gain_field
        else:
            self._propagation_func = self._Euler_frequency_domain_gain_field

        # self.oscillator == True if used in oscillator simulation
        # This should be changed in optical_assemblies.py
        self.oscillator = False

        self.cladding_pumping = cladding_pumping

        self.N_tot = N_ions

        # Load cross section data and retrieve wavelength limits from file
        self.signal_absorption_cs, self.signal_emission_cs, wl_lims \
            = utils.load_cross_sections(
                cross_section_file, '\t', self.grid.lambda_window, 1e-9,
                'linear')
        self.signal_absorption_cs \
            = utils.fftshift(self.signal_absorption_cs)
        self.signal_emission_cs \
            = utils.fftshift(self.signal_emission_cs)

        # Sort wavelength limits from grid OR cross-section file.
        # Used for plots.
        self.wl_lims = \
            [wl_lims[0] if wl_lims[0] > self.grid.lambda_window.min()
             else self.grid.lambda_window.min(),
             wl_lims[1] if wl_lims[1] < self.grid.lambda_window.max()
             else self.grid.lambda_window.max()]

        # Sort out pump(s) for appropriate geometry (determined by contents of
        # boundary_conditions).
        ASE_scaling = 1 - self.grid.t_range / seed_rep_rate
        if self.boundary_value_solver:
            self.pump = pmp.pump(
                self.boundary_conditions['co_pump_bandwidth'],
                self.boundary_conditions['co_pump_wavelength'],
                self.boundary_conditions['co_pump_power'] / seed_rep_rate,
                points=pump_points, lambda_lims=pump_wl_lims,
                ASE_scaling=ASE_scaling)
            self.counter_pump = pmp.pump(
                self.boundary_conditions['counter_pump_bandwidth'],
                self.boundary_conditions['counter_pump_wavelength'],
                self.boundary_conditions['counter_pump_power'] / seed_rep_rate,
                points=pump_points,
                lambda_lims=pump_wl_lims,
                ASE_scaling=ASE_scaling, direction='counter')
            if cladding_pumping:  # 2 more objects for core ASE
                self.co_core_ASE = pmp.pump(
                    self.boundary_conditions['co_pump_bandwidth'],
                    self.boundary_conditions['co_pump_wavelength'],
                    0, points=pump_points, lambda_lims=pump_wl_lims,
                    ASE_scaling=ASE_scaling)
                self.counter_core_ASE = pmp.pump(
                    self.boundary_conditions['co_pump_bandwidth'],
                    self.boundary_conditions['co_pump_wavelength'],
                    0, points=pump_points, lambda_lims=pump_wl_lims,
                    ASE_scaling=ASE_scaling, direction='counter')
        else:
            self.pump = pmp.pump(
                self.boundary_conditions['co_pump_bandwidth'],
                self.boundary_conditions['co_pump_wavelength'],
                self.boundary_conditions['co_pump_power'] / seed_rep_rate,
                points=pump_points, lambda_lims=pump_wl_lims,
                ASE_scaling=ASE_scaling)
        self.pump_absorption_cs, self.pump_emission_cs, _ \
            = utils.load_cross_sections(cross_section_file, '\t',
                                        self.pump.lambda_window, 1e-9,
                                        'linear')

        self.get_pump_refractive_index()  # Defined in derived class.

        # Overridden by optical assemblies, but still required if the optical
        # assemblies module is not used.
        self.verbose = verbose 

        # Determine signal overlaps
        self.signal_overlaps = self.get_overlaps_core_light(
            self.grid.points, self.grid.lambda_window, self.effective_MFD)
        self.signal_overlaps = utils.fftshift(self.signal_overlaps)

        # Determine pump overlaps
        if self.cladding_pumping:
            self.pump_overlaps, self.pump_mode_area = \
                self._get_cladding_light_overlap_and_effective_area(
                    self.pump.lambda_c, self.pump.points)

            # info. for co and counter core ASE, which are defined above
            # Check if PCF:
            self.core_ASE_cladding_ref_index = utils.Sellmeier(
                self.co_core_ASE.lambda_window, Sellmeier_file)
            if hasattr(self, 'hole_pitch'):  # PCF
                self.core_ASE_V, self.core_ASE_ref_index, _, _ = \
                    utils.PCF_propagation_parameters_K_Saitoh(
                        self.co_core_ASE.lambda_window,
                        self.co_core_ASE.midpoint,
                        self.co_core_ASE.omega_window, self.a, self.b, self.c,
                        self.d, self.hole_pitch, self.hole_diam,
                        self.core_radius, self.Sellmeier_file)
            else:  # step-index
                self.core_ASE_ref_index = self.core_ASE_cladding_ref_index \
                    + self.delta_n
        else:
            self.pump_overlaps = self.get_overlaps_core_light(
                self.pump.points, self.pump.lambda_window,
                self.pump_effective_MFD)

        self._precalculate_propagation_values()

    def make_verbose(self):
        """
        Change self.verbose to True
        Called by optical_assemblies. If the optical assembly verbosity is
        True, all component verbosities are also set to True.
        """
        self.verbose = True

    def make_silent(self):
        """
        Change self.verbose to False
        Called by optical_assemblies. If the optical assembly verbosity is
        False, all component verbosities are also set to False.
        """
        self.verbose = False

    def _precalculate_propagation_values(self):
        """
        Precalculate saturation energy, R12, & R21.
        """
        self.signal_mode_area_shift = utils.fftshift(self.signal_mode_area)
        self.signal_R12_coeff = (
            self.signal_absorption_cs * self.grid.df
            / (self.grid.energy_window_shift * self.signal_mode_area_shift))
        self.signal_R21_coeff = (
            self.signal_emission_cs * self.grid.df
            / (self.grid.energy_window_shift * self.signal_mode_area_shift))

        # Prevent division by zero - stop numpy from complaining.
        mask = np.abs(self.signal_absorption_cs + self.signal_emission_cs) == 0
        denom = self.signal_absorption_cs + self.signal_emission_cs
        denom[mask] = 1e-100
        self.Esat = (self.signal_mode_area_shift
                     * self.grid.energy_window_shift / denom)

    def _determine_propagation_method_from_boundaries(self):
        """
        Determine whether a full boundary value solver is required from the
        boundary conditions.
        """
        msg = (
            "Active fibre boundary values were not defined properly."
            " The dictionary keys for properly definining the boundary"
            " conditions are as follows:\n\n")
        msg += (
            "Co-pumping only, single-pass (no boundary value solver):\n"
            "\t'co_pump_power' (w)\n"
            "\t'co_pump_wavelength' (m)\n"
            "\t'co_pump_bandwidth' (m).\n"
            "\tNo counter-pump keys required.\n"
            "Co-pumping only (with boundary value solver):\n"
            "\t'co_pump_power' (w)\n"
            "\t'co_pump_wavelength' (m)\n"
            "\t'co_pump_bandwidth' (m)\n"
            "\t'counter_pump_power=0' (W).\n"
            "\tNo other counter-pump keys are required.\n"
            "Counter-pumping only (with boundary value solver):\n"
            "\t'counter_pump_power' (W)\n"
            "\t'counter_pump_wavelength' (m)\n"
            "\t'counter_pump_bandwidth' (m).\n"
            "\tNo co-pump keys are required.\n"
            "Bidirectional pumping (with boundary value solver):\n"
            "\t'co_pump_power' (W)\n"
            "\t'co_pump_wavelength' (m)\n"
            "\t'co_pump_bandwidth' (m)\n"
            "\t'counter_pump_power' (W)\n"
            "\t'counter_pump_wavelength' (m)\n"
            "\t'counter_pump_bandwidth' (m).")

        # Co-pumping / bidirectional pumping
        if ('co_pump_power' in self.boundary_conditions
                and 'co_pump_wavelength' in self.boundary_conditions
                and 'co_pump_bandwidth' in self.boundary_conditions):

            # Co-pumping only, no boundary value solver
            if ('counter_pump_power' not in self.boundary_conditions and
                    'counter_pump_wavelength' not in self.boundary_conditions
                    and 'counter_pump_bandwidth' not in
                        self.boundary_conditions):
                self.boundary_value_solver = False

            # Bidirectional pumping
            elif ('counter_pump_power' in self.boundary_conditions
                  and 'counter_pump_wavelength' in self.boundary_conditions
                  and 'counter_pump_bandwidth' in self.boundary_conditions):
                self.boundary_value_solver = True

            # Co-pumping, boundary value solver (no need to specify wavelength
            # and bandwidth for counter pump)
            elif ('counter_pump_power' in self.boundary_conditions
                  and self.boundary_conditions['counter_pump_power'] == 0):
                self.boundary_value_solver = True
                self.boundary_conditions['counter_pump_wavelength'] = \
                    self.boundary_conditions['co_pump_wavelength']
                self.boundary_conditions['counter_pump_bandwidth'] = \
                    self.boundary_conditions['co_pump_bandwidth']

            # Boundary conditions not understood
            else:
                raise exc.BoundaryConditionError(msg)

        # Counter-pumping
        elif ('counter_pump_power' in self.boundary_conditions
              and 'counter_pump_wavelength' in self.boundary_conditions
              and 'counter_pump_bandwidth' in self.boundary_conditions):

            # Counter-pumping only, still requires boundary value solver
            # (no need to specify wavelength and bandwidth for co pump).
            if ('co_pump_power' not in self.boundary_conditions
                    and 'co_pump_wavelength' not in self.boundary_conditions
                    and 'co_pump_bandwidth' not in self.boundary_conditions):
                self.boundary_value_solver = True
                self.boundary_conditions['co_pump_power'] = 0
                self.boundary_conditions['co_pump_wavelength'] = \
                    self.boundary_conditions['counter_pump_wavelength']
                self.boundary_conditions['co_pump_bandwidth'] = \
                    self.boundary_conditions['counter_pump_bandwidth']
            else:
                raise exc.BoundaryConditionError(msg)
        else:
            raise exc.BoundaryConditionError(msg)

    @abstractmethod
    def get_pump_refractive_index(cls):
        """
        Specific to each derived class. Must determine the pump refractive
        index.

        abstractmethod because this can be different for different fibre types
        and pumping geometries.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_pump_and_ASE_propagation_parameters():
        """
        Specific to each derived class. Must calculate the following
        parameters for the pump and ASE light:

        V, effective_MFD, mode_area.

        These parameters should be returned (staticmethod) so that this
        function can be used for the core ASE/pump and cladding ASE/pump.
        """
        raise NotImplementedError()

    def get_overlaps_core_light(self, points, lambda_window, MFD):
        """
        Calculate the overlap integrals for pump, ASE, or signal with the doped
        core. Assumes infinite rotational symmetry.

        Parameters
        ----------
        points : int
            Number of grid points.
        lambda_window : numpy array
            Wavelength window in m. Recommend using grid.lambda_window.
        MFD : numpy array
            Mode field diameter as a function of lambda_window

        Returns
        -------
        numpy array
            Overlap integral as a function of lambda_window.
        """
        # Limit the size of the MFD calculation by decimating if the number of
        # grid points is >= 512. This is necessary because the mode profile
        # needs to be calculated for all wavelengths, which is resource heavy
        # for large grid sizes. Interpolation is accurate because the MFD is
        # a slowly-varying function of wavelength.
        x_points = 512  # number of spatial grid points
        w_points = None  # number of frequency grid points
        decimate = False
        if points > 128:
            w_points = 128
            decimate = True
        else:
            w_points = points
        dx = 2 * self.pump_core_diam / x_points

        x_axis = dx * np.linspace(0, x_points - 1, x_points)
        x_axis = x_axis[:, None].repeat(w_points, axis=1)
        dx = np.gradient(x_axis, axis=0)
        fibre_profile = np.zeros((x_points, w_points))
        fibre_profile[np.abs(x_axis) < (self.core_diam / 2)] = 1

        # Overlap def.: Giles, et al 'Modeling Er-doped fibre amplifiers'
        overlaps = None
        if decimate:  # Calculate MFD with low resolution and interpolate.
            indices = np.linspace(0, w_points - 1, w_points, dtype=int) \
                * int(points / w_points)
            decimated_lambda_window = lambda_window[indices]
            decimated_MFD = MFD[indices][None, :].repeat(x_points, axis=0)
            decimated_overlaps = np.zeros((x_points, w_points))

            # Field; no factor of 2log(2) required for spatial distribution.
            mode_profile = np.exp(-1 * x_axis**2 / (decimated_MFD / 2)**2)
            # mode_profile /= np.sum(
            #     mode_profile * x_axis * dx * 2*np.pi, axis=0)
            # decimated_overlaps = np.sum(
            #     mode_profile * fibre_profile * x_axis * dx * 2*np.pi, axis=0)
            mode_profile /= np.sum(mode_profile, axis=0)
            decimated_overlaps = np.sum(mode_profile * fibre_profile, axis=0)
            overlaps = interp.interp1d(
                decimated_lambda_window, decimated_overlaps,
                fill_value='extrapolate', kind='quadratic')
            overlaps = overlaps(lambda_window)
        else:
            tiled_MFD = MFD[None, :].repeat(x_points, axis=0)
            mode_profile = np.exp(-1 * x_axis**2 / (tiled_MFD / 2)**2)
            mode_profile /= np.sum(mode_profile, axis=0)
            overlaps = np.sum(mode_profile * fibre_profile, axis=0)
        return overlaps

    def _get_cladding_light_overlap_and_effective_area(self, lambda_c, points):
        """
        Calculate the overlap integral of the cladding light with the doped
        fibre core and the effective mode area of the cladding light.

        Parameters
        ----------
        lambda_c : float
            Central value of the wavelength grid of the light in the
            cladding in m.
        points : int
            Interpolated size required for overlap and effective area

        Returns
        -------
        numpy array
            Overlap integral as a function of wavelength.
        numpy array
            Effective area as a function of wavelength.

        Notes
        -----
        Assumes LP modes. The cladding light amplitude distribution is assumed
        to be have infinite rotational symmetry and is given by a scaled
        incoherent sum of up to the first 4000 LP modes supported by the
        cladding. The calculation is done for the central wavelength only.
        """
        solver = bms.bessel_mode_solver(
            self.pump_core_diam / 2, 1.25 * self.pump_core_diam / 2,
            self.pump_ref_index, self.pump_cladding_ref_index, lambda_c)
        r = np.linspace(0, solver.clad_rad, 2**11)  # polar axis
        dr = np.diff(r)[0]
        fibre_profile = np.zeros_like(r)
        fibre_profile[r <= self.core_diam / 2] = 1
        solver.solve(max_modes=4000)
        solver.make_modes(r, num_modes=4000)
        solver.get_amplitude_distribution(std=4000)
        effective_area = np.sum(
            solver.amplitude_distribution**2 * r * dr * 2 * np.pi)**2
        effective_area /= np.sum(
            solver.amplitude_distribution**4 * r * dr * 2 * np.pi)
        overlap = np.pi * (self.core_diam / 2)**2 / effective_area
        overlap = overlap.repeat(points)
        effective_area = effective_area.repeat(points)
        return overlap, effective_area

    def add_pump(
            self, wavelength, bandwidth, power, repetition_rate, direction):
        """
        Add another pump source.

        This method permits multiple pump wavelengths with a common propagation
        direction. Co- and counter-pump light can be defined independently at
        instantiation using the boundary conditions, but if two co- OR counter-
        pump wavelengths are required, the second must be added using this
        method after instantiation. The second pump source is added to the pump
        spectrum which propagates in the same direction, i.e.,
        self.pump.spectrum if direction == 'co', or self.counter_pump.spectrum
        if direction == 'counter'.

        Parameters
        ----------
        wavelength : float
            Pump wavelength, nm
        bandwidth : float
            Pump bandwidth, nm
        power : float
            Pump power, W
        repetition_rate : float
            Laser repetition rate, Hz
        direction : str
            Pumping direction (either with, 'co', or against, 'counter', the
            signal light).
            Possible values: 'co', 'counter'

        Notes
        -----
        wavelength must be within the existing pump and ASE wavelength grid.
        If cladding pumping, all new pumps will be added to the cladding too.
        If cladding pumping, all pump propagation parameters are assumed to be
        the same as those calculated using the pump source passed when
        instantiating the active fibre.
        """
        if direction not in ['co', 'counter']:
            raise ValueError("Parameter direction must be either 'co' or"
                             " 'counter")
        if not (self.pump.lambda_lims[0] < wavelength
                < self.pump.lambda_lims[1]):
            raise ValueError(
                "Parameter wavelength must be a float between the pump grid "
                "wavelength limits %f nm and %f nm"
                % (self.pump.lambda_lims[0], self.pump.lambda_lims[1]))

        # Without the following, pump spectrum is often just a single point
        if bandwidth < 2 * np.amax(self.pump.d_wl):
            bandwidth = 2 * np.amax(self.pump.d_wl)

        spec = np.zeros_like(self.pump.spectrum)
        spec[:, np.abs(
            self.pump.lambda_window - wavelength)**2 < (bandwidth / 2)**2] = 1
        spec /= np.sum(spec * self.pump.dOmega)
        spec *= power / repetition_rate

        if direction == 'co':
            self.pump.spectrum += spec
            esd, _ = self.pump.get_ESD_and_PSD(
                self.pump.spectrum, repetition_rate)
            self.pump.energy = np.sum(esd * self.pump.d_wl)
        elif direction == 'counter':
            self.counter_pump.spectrum += spec
            esd, _ = self.counter_pump.get_ESD_and_PSD(
                self.counter_pump.spectrum, repetition_rate)
            self.counter_pump.energy = np.sum(esd * self.counter_pump.d_wl)

    def _stack_propagation_arrays_co_signal(
            self, pulse_spec, pulse_ASE_scaling, repetition_rate):
        """
        Prepare stacked arrays for co-propagating signals only.

        Parameters
        ----------
        pulse_spec : complex numpy array
            pulse field in the frequency domain.
        pulse_ASE_scaling : numpy array
            Scaling to be applied to the pulse (and pump) ASE terms in the
            Giles equation to account for the size of grid.time_window in
            relation to the seed laser repetition rate.
        repetition_rate : float.
            Repetition rate of the seed laser. See pulse.repetition_rate.

        Return
        ------
        stacker object
            Contains numpy arrays formed by hstacking signal, pump, and ASE
            propagation parameters.
        """
        stacks = self.stacker()

        # Modify co and counter pump spectra dimensions for polarization axes
        # stacks.spectra has shape (2, num_points) to accommodate polarization
        # all other arrays have shape (num_points).
        stacks.spectra = np.hstack((pulse_spec, self.pump.spectrum))
        stacks.directions = np.hstack(
            (np.ones((self.grid.points)), np.ones((self.pump.points))))
        stacks.abs_cs = np.hstack(
            (self.signal_absorption_cs, self.pump_absorption_cs))
        stacks.em_cs = np.hstack(
            (self.signal_emission_cs, self.pump_emission_cs))
        stacks.overlaps = np.hstack(
            (self.signal_overlaps, self.pump_overlaps))
        stacks.mode_area = np.hstack(
            (self.signal_mode_area_shift, self.pump_mode_area))
        stacks.energy_window = np.hstack(
            (self.grid.energy_window_shift, self.pump.energy_window))
        stacks.dOmega = np.hstack(
            (self.grid.dOmega * np.ones((self.grid.points)),
             self.pump.dOmega * np.ones((self.pump.points))))
        stacks.ASE_scaling = np.hstack(
            (pulse_ASE_scaling * np.ones((self.grid.points)),
             self.pump.ASE_scaling * np.ones((self.pump.points))))

        # Used for plotting
        stacks.lambda_window = np.hstack(
            (self.grid.lambda_window, self.pump.lambda_window))

        # Slices dictionary to keep track of indices
        stacks.slices = {}
        stacks.slices['co_signal'] = slice(0, self.grid.points, 1)
        stacks.slices['co_pump'] = \
            slice(stacks.slices['co_signal'].stop,
                  stacks.slices['co_signal'].stop + self.pump.points, 1)

        if self.cladding_pumping:
            half = int(len(stacks.energy_window) / 2)
            stacks.spectra = np.hstack(
                (stacks.spectra[:, :half], self.co_core_ASE.spectrum))
            stacks.directions = np.hstack(
                (stacks.directions[:half], np.ones((self.pump.points))))
            stacks.abs_cs = np.hstack(
                (stacks.abs_cs[:half], self.pump_absorption_cs))
            stacks.em_cs = np.hstack(
                (stacks.em_cs[:half], self.pump_emission_cs))
            stacks.overlaps = np.hstack(
                (stacks.overlaps[:half], self.core_ASE_overlaps))
            stacks.mode_area = np.hstack(
                (stacks.mode_area[:half], self.core_ASE_mode_area))
            stacks.energy_window = np.hstack(
                (stacks.energy_window[:half], self.co_core_ASE.energy_window))
            stacks.dOmega = np.hstack(
                (stacks.dOmega[:half],
                 self.co_core_ASE.dOmega * np.ones((self.co_core_ASE.points))))
            stacks.ASE_scaling = np.hstack(
                (stacks.ASE_scaling[:half],
                 self.co_core_ASE.ASE_scaling
                 * np.ones((self.co_core_ASE.points))))

            # Used for plotting
            stacks.lambda_window = np.hstack(
                (stacks.lambda_window[:half],
                 self.co_core_ASE.lambda_window))

            # Slot co_core_ASE slice after co-propagating slices
            # Append counter_core_ASE slice to end
            stacks.slices['co_core_ASE'] = \
                slice(half, half + self.co_core_ASE.points, 1)

        # Containers for constant arrays
        # Factor of 2 in ASE_coeff NOT used here; polarization resolved,
        # so it is included instead by defining pump/ASE polarization axes.
        stacks.inversion_abs_coeff = stacks.abs_cs * stacks.dOmega \
            / (stacks.mode_area * stacks.energy_window)
        stacks.inversion_em_coeff = stacks.em_cs * stacks.dOmega \
            / (stacks.mode_area * stacks.energy_window)
        stacks.ASE_coeff = stacks.ASE_scaling * self.N_tot * stacks.overlaps \
            * stacks.em_cs * stacks.energy_window / repetition_rate
        return stacks

    def _stack_propagation_arrays_boundary_value_solver(
            self, pulse_spec, pulse_ASE_scaling, repetition_rate):
        """
        Prepare stacked arrays for full ASE propagation.

        Parameters
        ----------
        pulse_spec : complex numpy array
            pulse field in the frequency domain.
        pulse_ASE_scaling : numpy array
            Scaling to be applied to the pulse (and pump) ASE terms in the
            Giles equation to account for the size of grid.time_window in
            relation to the seed laser repetition rate.
        repetition_rate : float.
            Repetition rate of the seed laser. See pulse.repetition_rate.

        Return
        ------
        stacker object
            Contains numpy arrays formed by hstacking signal and pump
            propagation parameters.
        """
        stacks = self.stacker()

        # Modify co and counter pump spectra dimensions for polarization axes
        # stacks.spectra has shape (2, num_points) to accommodate polarization
        # all other arrays have shape (num_points).
        stacks.spectra = np.hstack(
            (pulse_spec, self.pump.spectrum, np.zeros_like(pulse_spec),
             self.counter_pump.spectrum))
        stacks.directions = np.hstack(
            (np.ones((self.grid.points)), np.ones((self.pump.points)),
             -1 * np.ones((self.grid.points)),
             -1 * np.ones((self.counter_pump.points))))
        stacks.abs_cs = np.hstack(
            (self.signal_absorption_cs, self.pump_absorption_cs,
             self.signal_absorption_cs, self.pump_absorption_cs))
        stacks.em_cs = np.hstack(
            (self.signal_emission_cs, self.pump_emission_cs,
             self.signal_emission_cs, self.pump_emission_cs))
        stacks.overlaps = np.hstack(
            (self.signal_overlaps, self.pump_overlaps, self.signal_overlaps,
             self.pump_overlaps))
        stacks.mode_area = np.hstack(
            (self.signal_mode_area_shift, self.pump_mode_area,
             self.signal_mode_area_shift, self.pump_mode_area))
        stacks.energy_window = np.hstack(
            (self.grid.energy_window_shift, self.pump.energy_window,
             self.grid.energy_window_shift, self.counter_pump.energy_window))
        stacks.dOmega = np.hstack(
            (self.grid.dOmega * np.ones((self.grid.points)),
             self.pump.dOmega * np.ones((self.pump.points)),
             self.grid.dOmega * np.ones((self.grid.points)),
             self.counter_pump.dOmega * np.ones((self.counter_pump.points))))
        stacks.ASE_scaling = np.hstack(
            (pulse_ASE_scaling * np.ones((self.grid.points)),
             self.pump.ASE_scaling * np.ones((self.pump.points)),
             pulse_ASE_scaling * np.ones((self.grid.points)),
             self.counter_pump.ASE_scaling
             * np.ones((self.counter_pump.points))))

        # Used for plotting
        stacks.lambda_window = np.hstack(
            (self.grid.lambda_window, self.pump.lambda_window,
             self.grid.lambda_window, self.counter_pump.lambda_window))

        # Slices dictionary to keep track of indices
        stacks.slices = {}
        stacks.slices['co_signal'] = slice(0, self.grid.points, 1)
        stacks.slices['co_pump'] = \
            slice(stacks.slices['co_signal'].stop,
                  stacks.slices['co_signal'].stop + self.pump.points, 1)
        stacks.slices['counter_signal'] = \
            slice(stacks.slices['co_pump'].stop,
                  stacks.slices['co_pump'].stop + self.grid.points, 1)
        stacks.slices['counter_pump'] = \
            slice(stacks.slices['counter_signal'].stop,
                  stacks.slices['counter_signal'].stop + self.pump.points, 1)

        if self.cladding_pumping:
            half = int(len(stacks.energy_window) / 2)
            stacks.spectra = np.hstack(
                (stacks.spectra[:, :half], self.co_core_ASE.spectrum,
                 stacks.spectra[:, half:], self.counter_core_ASE.spectrum))
            stacks.directions = np.hstack(
                (stacks.directions[:half], np.ones((self.pump.points)),
                 stacks.directions[half:], -1 * np.ones((self.pump.points))))
            stacks.abs_cs = np.hstack(
                (stacks.abs_cs[:half], self.pump_absorption_cs,
                 stacks.abs_cs[half:], self.pump_absorption_cs))
            stacks.em_cs = np.hstack(
                (stacks.em_cs[:half], self.pump_emission_cs,
                 stacks.em_cs[half:], self.pump_emission_cs))
            stacks.overlaps = np.hstack(
                (stacks.overlaps[:half], self.core_ASE_overlaps,
                 stacks.overlaps[half:], self.core_ASE_overlaps))
            stacks.mode_area = np.hstack(
                (stacks.mode_area[:half], self.core_ASE_mode_area,
                 stacks.mode_area[half:], self.core_ASE_mode_area))
            stacks.energy_window = np.hstack(
                (stacks.energy_window[:half], self.co_core_ASE.energy_window,
                 stacks.energy_window[half:],
                 self.counter_core_ASE.energy_window))
            stacks.dOmega = np.hstack(
                (stacks.dOmega[:half],
                 self.co_core_ASE.dOmega * np.ones((self.co_core_ASE.points)),
                 stacks.dOmega[half:],
                 self.counter_core_ASE.dOmega
                 * np.ones((self.counter_core_ASE.points))))
            stacks.ASE_scaling = np.hstack(
                (stacks.ASE_scaling[:half],
                 self.co_core_ASE.ASE_scaling
                 * np.ones((self.co_core_ASE.points)),
                 stacks.ASE_scaling[half:],
                 self.counter_core_ASE.ASE_scaling
                 * np.ones((self.counter_core_ASE.points))))

            # Used for plotting
            stacks.lambda_window = np.hstack(
                (stacks.lambda_window[:half],
                 self.co_core_ASE.lambda_window,
                 stacks.lambda_window[half:],
                 self.counter_core_ASE.lambda_window))

            # Slot co_core_ASE slice after co-propagating slices
            # Append counter_core_ASE slice to end
            stacks.slices['co_core_ASE'] = \
                slice(half, half + self.co_core_ASE.points, 1)
            stacks.slices['counter_signal'] = \
                slice(stacks.slices['co_core_ASE'].stop,
                      stacks.slices['co_core_ASE'].stop + self.grid.points, 1)
            stacks.slices['counter_pump'] = \
                slice(stacks.slices['counter_signal'].stop,
                      stacks.slices['counter_signal'].stop
                      + self.counter_pump.points, 1)
            stacks.slices['counter_core_ASE'] = \
                slice(stacks.slices['counter_pump'].stop,
                      stacks.slices['counter_pump'].stop
                      + self.counter_core_ASE.points, 1)

        # Containers for constant arrays
        # Factor of 2 in ASE_coeff NOT used here; polarization resolved,
        # so it is included instead by defining pump/ASE polarization axes.
        stacks.inversion_abs_coeff = stacks.abs_cs * stacks.dOmega \
            / (stacks.mode_area * stacks.energy_window)
        stacks.inversion_em_coeff = stacks.em_cs * stacks.dOmega \
            / (stacks.mode_area * stacks.energy_window)
        stacks.ASE_coeff = stacks.ASE_scaling * self.N_tot * stacks.overlaps \
            * stacks.em_cs * stacks.energy_window / repetition_rate
        return stacks

    def _CQEM_adjust_dz_only(self, photon_error, dz):
        """
        Conservation quantity error method (photon counting) for adaptive step
        sizing ONLY (i.e., no judgement is made about propagation errors in the
        current solution, unlike CQEM). Used to monitor and control step size
        in active fibres.

        A. Heidt, Journal of Lightwave Technology, 27(18) 2009.

        Parameters
        ----------
        photon_error : float
            Difference in photon number before and after the most recent
            propagation step.
        dz : float
            Size of the most recent propagation step.

        Returns
        -------
        float
            Size of the next propagation step.
        """
        if photon_error > 2. * self.tol:
            dz /= 2.
        elif photon_error > self.tol:
            dz /= 2.**0.2
        elif photon_error < 0.1 * self.tol:
            dz *= 2.**0.2
        return dz

    def _get_gain_two_level(self, N2, overlaps, emission_cs, absorption_cs):
        """
        Return the frequency domain gain.

        Parameters
        ----------
        N2 : float
            Fractional population inversion for a two-level system (i.e.,
            N_total = N1 + N2)
        overlaps : numpy array
            Overlap integrals.
        emission_cs : numpy array
            Emission cross sections.
        absorption_cs : numpy array
            Absorption cross sections.

        Returns
        -------
        numpy array
            Two-level gain for the current propagation step.
        """
        N1 = 1 - N2
        return self.N_tot * overlaps * (emission_cs * N2 - absorption_cs * N1)

    def _get_inversion_two_level(
            self, spectrum, abs_coeff, em_coeff, repetition_rate):
        """
        Return the fractional population inversion of a two-level active fibre.

        Parameters
        ----------
        spectrum : numpy array
            Energy spectral density.
        abs_coeff : numpy array
            overlaps * abs_cs * dOmega / (mode_area * energy_window)
        em_coeff : numpy array
            overlaps * em_cs * dOmega / (mode_area * energy_window)
        repetition_rate : float
            Repetition frequency of the seed laser. See pulse.repetition_rate.

        Returns
        -------
        numpy array
            Fractional population inversion

        Notes
        -----
        All numpy arrays are to be hstacked as follows (i.e., using one of the
        stacking methods):
            np.hstack(
                (co signal + ASE, co pump + ASE, counter signal + ASE,
                 counter pump + ASE))
        """
        R12 = np.sum(abs_coeff * spectrum)
        R21 = np.sum(em_coeff * spectrum)
        return R12 / (R12 + R21 + 1 / (self.lifetime * repetition_rate))

    @staticmethod
    def _ASE_energy_term_Giles_model(ASE_coeff, inversion):
        """
        Return the ASE term of Giles PDE for power evolution along the length
        of an active fibre. See C. R. Giles et al., JLT 9(2) pp 271-283 (1991).

        Parameters
        ----------
        coeff : numpy array
            See stacks.ASE_coeff.
            ASE_scaling * N_tot * overlap * em_cs \
                * energy_window / repetition_rate
        inversion : float
            Fractional population inversion.

        Returns
        -------
        numpy array
            ASE added over the propagation step.

        Notes
        -----
        The term is scaled for energy spectral density, not power (as was the
        case in Giles' paper) because this has been most convenient when
        including nonlinearity and dispersion.

        All numpy arrays are to be hstacked as follows (i.e., using one of the
        stacking methods):
            np.hstack(
                (co signal + ASE, co pump + ASE, counter signal + ASE,
                 counter pump + ASE))
        """
        # staticmethod because ASE_coeff sometimes needs indexing. This is
        # easier to do before calling _ASE_energy_term_Giles_model because the
        # latter option would require an extra slice parameter to be passed as
        # an argument.
        ase = ASE_coeff * inversion
        return ase

    def _propagate_energy_spectra_boundary_value_solver(
            self, indices, spectrum, repetition_rate, sample=False):
        """
        Propagate energy spectral densities using the Giles model.

        Parameters
        ----------
        indices : slice object
            Indices to of spectrum propagate over.
        spectrum : numpy array.
            Energy spectral density (ESD).
        repetition_rate : float.
            Repetition rate of the seed laser. See pulse.repetition_rate
        sample: bool.
            Choose whether to sample the ESD as a function of propagation
            distance. Default False.

        Returns
        -------
        numpy array
            if sample:
                ESD samples as a function of propagation distance.
            else:
                ESD at the fibre output.
        numpy array
            Population inversion as a function of propagation distance.

        Notes
        -----
        This method can be used if all input spectra are already known
        (including counter-propagating spectra at signal input).
        If input spectra need retrieving (i.e., the counter-propagating light
        emitted by the signal input is not known), use
        _solve_boundaries_ASE_ESD_only, which calls
        _get_propagation_samples_boundary_value_solver.

        All numpy arrays are to be hstacked as follows:
            np.hstack(
                (co signal + ASE, co pump + ASE, counter signal + ASE,
                 counter pump + ASE))
        """
        dz = self.L / self.num_steps
        N2 = np.zeros((self.num_steps))
        if sample:
            num_points = indices.stop - indices.start
            samples = np.zeros((
                self.num_steps + 1, spectrum.shape[0], num_points))
            samples[0, :, :] = spectrum[:, indices]

        propagated = 0
        for i in range(self.num_steps):
            if propagated + dz > self.L:
                dz = self.L - propagated
            N2[i] = self._get_inversion_two_level(
                spectrum[:, indices], self.stacks.inversion_abs_coeff[indices],
                self.stacks.inversion_em_coeff[indices], repetition_rate)
            g = self._get_gain_two_level(N2[i], self.stacks.overlaps[indices],
                                         self.stacks.em_cs[indices],
                                         self.stacks.abs_cs[indices])
            ase = self._ASE_energy_term_Giles_model(
                self.stacks.ASE_coeff[indices], N2[i])
            spectrum = spectrum[:, indices] * \
                (1 + dz * self.stacks.directions[indices] * g) \
                + self.stacks.directions[indices] * ase * dz
            propagated += dz
            if sample:
                samples[i + 1, :, :] = spectrum[:, indices]

        if sample:
            return samples, N2
        else:
            return spectrum, N2

    def _get_propagation_samples_boundary_value_solver(
            self, repetition_rate, spectrum, static_idx, update_idx,
            static_samples):
        """
        Propagate energy spectral densities using the Giles model.

        Parameters
        ----------
        repetition_rate : float
            Repetition rate of the laser source. See pulse.repetition_rate
        spectrum : numpy array
            Energy spectral density (ESD).
        static_idx : slice object
            Indices of spectrum to be held constant
        update_idx : slice object
            Indices of spectrum to be updated.
        static_samples : numpy array
            Static spectra at each propagation step.

        Returns
        -------
        numpy array
            ESD as a function of propagation distance.
        numpy array
            Population inversion as a function of propagation distance.

        Notes
        -----
        The energy spectral densities are sliced into 'updated' and 'static'
        sections as follows:
            spectrum[static_idx] are not modified, but are used to calculate N2
            spectrum[update_idx] are both modified and used to calculate N2

        All numpy arrays are to be hstacked as follows:
            np.hstack(
                (co signal + ASE, co pump + ASE, counter signal + ASE,
                 counter pump + ASE))
        """
        dz = self.L / self.num_steps
        N2 = np.zeros((self.num_steps))
        samples = np.zeros((
            self.num_steps + 1, spectrum.shape[0], spectrum.shape[1]))
        samples[0, :, :] = spectrum

        propagated = 0
        for i in range(self.num_steps):
            spectrum[:, static_idx] = static_samples[i, :, static_idx]
            if propagated + dz > self.L:
                dz = self.L - propagated
            N2[i] = self._get_inversion_two_level(
                spectrum, self.stacks.inversion_abs_coeff,
                self.stacks.inversion_em_coeff, repetition_rate)
            g = self._get_gain_two_level(
                N2[i], self.stacks.overlaps[update_idx],
                self.stacks.em_cs[update_idx], self.stacks.abs_cs[update_idx])
            ase = self._ASE_energy_term_Giles_model(
                self.stacks.ASE_coeff[update_idx], N2[i])
            spectrum[:, update_idx] = spectrum[:, update_idx] * \
                (1 + dz * self.stacks.directions[update_idx] * g) \
                + self.stacks.directions[update_idx] * ase * dz
            propagated += dz
            samples[i + 1, :, update_idx] = spectrum[:, update_idx]
        return samples, N2

    def _solve_boundaries_ASE_ESD_only(self, spectrum, repetition_rate):
        """
        Retrieve energy spectral densities at both the signal input and output.

        Parameters
        ----------
        spectrum : numpy array
            Energy spectral density (ESD)
        repetition_rate : float
            Repetition rate of the laser source. See pulse.repetition_rate.

        Returns
        -------
        numpy array
            co-propagating ESD at the signal output end of the fibre, and
            counter-propagating ESD at the signal input end of the fibre
            (i.e., the solutions to the boundary value problem).
        numpy array
            ESD as a function of propagation distance
        numpy array
            Population inversion as a function of propagation distance
        float
            Error of the boundary value solution calculated as the
            summed absolute difference between the population inversion
            as a function of propagation distance for the Nth and the
            (N-1)th iteration.

        Raises
        ------
        PropagationMethodNotConvergingError
            The boundary value solver has stagnated or returned a diverging
            error after 50 iterations.

        Notes
        -----
        See R. Lindberg, et al., Scientific Reports 6, article number 34742
        (2016).

        Unlike R. Lindberg, solution quality is evaluated using
        error = sum((inversion_i(z) - inversion_i+1(z)) / inversion_i+1(z))
        where inversion_i and inversion_i+1 are the fractional population
        inversions at each propagation step for iterations i and i+1,
        respectively.

        All numpy arrays are to be hstacked as follows if core pumping:
        np.hstack(
            (co signal + ASE, co pump + ASE, counter signal + ASE,
             counter pump + ASE))
        and as follows if cladding pumping:
        np.hstack(
            (co signal + ASE, co pump + ASE, co core ASE, counter signal + ASE,
             counter pump + ASE, counter core ASE))

        Procedure:
        1) FORWARDS simulation of co-propagating signals only.
        2) BACKWARDS Simulation of co- and counter-propagating signals.
            N2 is calculated using co-signal ESDs from step 1.
            co-propagating signals are NOT updated.
            counter-propagating signals ARE updated.
        3) FORWARDS simulation of co- and counter-propagating signals.
            N2 is calculated using counter-signal ESDs from step 2.
            co-propagating signals ARE updated.
            counter-propagating signals are NOT updated.
        4) Steps 2 and 3 are repeated until convergence is reached.
        """
        # Indices for conveniently splitting stacks into co- and counter-
        # propagating signals
        size = spectrum.shape[1]
        half_1 = slice(0, int(size / 2), 1)
        half_2 = slice(int(size / 2), size, 1)
        split_idx = (half_1, half_2)

        # Copy boundary conditions
        co_light = spectrum[:, half_1].copy()
        counter_light = spectrum[:, half_2].copy()

        # Co-propagating signals only at first -- STEP 1 ABOVE
        samples, prev_N2 = \
            self._propagate_energy_spectra_boundary_value_solver(
                half_1, spectrum, repetition_rate, sample=True)
        spectrum[:, half_1] = samples[-1, :, :]

        # Begin solution-finding loop
        # Even iterations are backwards propagation with:
        #   static co-propagating signals
        #   updated counter-propagating signals
        # Odd iterations are forwards propagation with:
        #   updated co-propagating signals
        #   static counter-propagating signals
        err = []

        if self.verbose:
            print('Convergence error (spectral density only):')
        for i in range(self.num_iters):
            if (i % 2) == 0:  # co signals static, counter signals updated
                static = split_idx[0]
                update = split_idx[1]
            else:             # counter signals static, co signals updated
                static = split_idx[1]
                update = split_idx[0]
            self.forward_propagation = not self.forward_propagation

            # Direction needs reversing and sampled spectra need to be flipped;
            # Signals are static if they are being propagated backwards
            self.stacks.directions *= -1
            samples = samples[::-1, :, :]

            # Get updates and replace update indices in spectra array
            samples, N2 = self._get_propagation_samples_boundary_value_solver(
                repetition_rate, spectrum, static, update, samples)

            # Enforce boundary conditions for static indices, assess error
            # odd (forward) iteration, and terminate if err < tol or if no
            # convergence is detected.
            if (i % 2) == 0:
                spectrum[:, static] = co_light
                samples[-1, :, static] = co_light
                counter_at_input = samples[-1, :, update]
            else:
                spectrum[:, static] = counter_light
                samples[-1, :, static] = counter_light
                co_at_output = samples[-1, :, update]
                err.append(np.sum(np.abs(prev_N2 - N2) / N2))
                if self.verbose:
                    print('\t', err[-1])
                if len(err) > 10:  # Dodge RuntimeWarning: Mean of empty slice
                    diff_err = np.diff(err[-10::])
                    mean_diff_err = np.mean(diff_err)
                if err[-1] < self.convergence_tol:
                    break
                if i >= 50 and mean_diff_err >= 0:
                    msg = (
                        "\n\nThe propagation method isn't converging after %d "
                        "iterations.\nPlease revise the pump, fibre, or "
                        "signal parameters." % i)
                    raise exc.PropagationMethodNotConvergingError(msg)
                else:
                    prev_N2 = N2
        return np.append(co_at_output, counter_at_input, axis=1), samples, \
            N2, err

    @staticmethod
    def _Euler_frequency_domain_gain_field(spec_field, dz, g, direction=1):
        """
        Euler integration with frequency-domiain gain for complex spectral
        fields.

        Parameters
        ----------
        spec_field : np.array
            Complex spectrum of the pulse.
        dz : float
            Integration step size.
        g : np.array
            Energy gain spectrum to be applied to the pulse.
        direction : float or np.array of size spec_field
            All elements are +1 if forward propagation, and -1 if backward
            propagation.

        Returns
        -------
        numpy array
            spec_field after the gain for step dz has been applied.
        """
        return spec_field * np.sqrt(1. + dz * direction * g,
                                    dtype=np.complex128)

    def _Euler_approximate_mixed_domain_gain_field(
            self, spec_field, dz, g, direction=1):
        """
        Euler integration with frequency-domain gain for complex spectral
        fields and a time-domain 'correction factor' which approximates the
        effects of gain saturation in the time domain.

        Parameters
        ----------
        spec_field : np.array
            Complex spectrum of the pulse.
        dz : float
            Propagation step size.
        g : np.array
            Energy gain spectrum to be applied to the pulse.
        direction : float or np.array of size spec_field.
            All elements are +1 if forward propagation, and -1 if backward
            propagation.

        Returns
        -------
        numpy array
            spec_field after the gain for step dz has been applied.

        Notes
        -----
        See G. P. Agrawal, IEEE Journ. Quantum Electron., 27(6), 1991 for the
        frequency-independent time-domain gain operator.
        """
        spec_field = self._Euler_frequency_domain_gain_field(
            spec_field, dz, g, direction=direction)
        spec = np.sum(spec_field.real**2 + spec_field.imag**2, axis=0)
        av_Esat = np.average(self.Esat, weights=spec)
        f = utils.ifft(spec_field)
        P = np.sum(f.real**2 + f.imag**2, axis=0)
        f *= np.exp(-1 * np.cumsum(P * self.grid.dt / av_Esat) * dz / 2)
        spec_field = utils.fft(f)
        return spec_field

    def _add_ASE_field(self, spec_field, ase, dz, direction=1):
        """
        Make the ASE field from the ASE energy spectrum (calculated using
        self._ASE_energy_term_Giles_model) and add to spec_field.

        Parameters
        ----------
        spec_field : np.array
            Complex spectrum of the pulse.
        ase : np.array
            Energy spectrum of the ASE contribution to the pulse.
        dz : float
            Integration step size.
        direction : float or np.array of size spec_field.
            All elements are +1 if forward propagation, and -1 if backward
            propagation.

        Returns
        -------
        numpy array
            spec_field after the ASE spectrum for propagation step dz has
            been added.
        """
        noise = np.random.rand(spec_field.size).reshape(spec_field.shape)
        phase = np.exp(-2j * np.pi * noise)  # [0; 2*pi)
        return (spec_field + np.sqrt(ase * dz) * phase * self.grid.FFT_scale
                * direction)

    def _propagate_boundary_value_solver_field(
            self, field, repetition_rate, samples, spectrum, prev_N2,
            sampling=False, num_samples=np.inf):
        """
        Iteratively retrieve energy spectral densities at both the signal input
        and output of an active fibre, incorporating full-field propagation for
        the signal light.

        Parameters
        ----------
        field : complex numpy array
            Time-domain field distribution.
        repetition_rate : float
            repetition_rate of the input pulses.
        samples : numpy array
            Spectra as a function of z for an initial guess of how the
            energy spectral densities evolve along the fibre length.
        spectrum : numpy array
            Energy spectral density (ESD).
        prev_N2 : numpy array
            Guesses for the inversion vs length.
        sampling : bool
            Spectral samples drawn at intervals along the fibre when True.
        num_samples :  int
            Number of samples to draw along the fibre length.

        Returns
        -------
        if sampling:
            numpy array
                co-propagating ESD at the signal output end of the fibre, and
                counter-propagating ESD at the signal input end of the fibre
                (i.e., the solutions to the boundary value problem).
            numpy array
                Pulse field at the signal output of the fibre.
            numpy array
                ESD as a function of propagation distance
            numpy array
                Population inversion as a function of propagation distance
            float
                Error of the boundary value solution calculated as the
                summed absolute difference between the population inversion
                as a function of propagation distance for the Nth and the
                (N-1)th iteration.
            list
                Points along the fibre at which samples were taken.
            list
                Pulse field as a function of propagation distance.
            numpy array
                ESD as a function of propagation distance
        else:
            numpy array
                co-propagating ESD at the signal output end of the fibre, and
                counter-propagating ESD at the signal input end of the fibre
                (i.e., the solutions to the boundary value problem).
            numpy array
                Pulse field at the signal output of the fibre.
            numpy array
                ESD as a function of propagation distance
            numpy array
                Population inversion as a function of propagation distance
            float
                Error of the boundary value solution calculated as the
                summed absolute difference between the population inversion
                as a function of propagation distance for the Nth and the
                (N-1)th iteration.

        Raises
        ------
        PropagationMethodNotConvergingError
            The boundary value solver has stagnated or returned a diverging
            error after 50 iterations.

        Notes
        -----
        See R. Lindberg, et al., Scientific Reports 6, article number 34742
        (2016).

        Unlike R. Lindberg, solution quality is evaluated using
        error = sum((inversion_i(z) - inversion_i+1(z)) / inversion_i+1(z))
        where inversion_i and inversion_i+1 are the fractional population
        inversions at each propagation step for iterations i and i+1,
        respectively.

        All numpy arrays are to be hstacked as follows:
            np.hstack(
                (co signal + ASE, co pump + ASE, counter signal + ASE,
                 counter pump + ASE))

        Procedure:
        1) FORWARDS simulation of co-propagating signals only.
        2) BACKWARDS Simulation of co- and counter-propagating signals.
            N2 is calculated using co-signal ESDs from step 1.
            co-propagating signals are NOT updated.
            counter-propagating signals ARE updated.
        3) FORWARDS simulation of co- and counter-propagating signals.
            N2 is calculated using counter-signal ESDs from step 2.
            co-propagating signals ARE updated.
            counter-propagating signals are NOT updated.
        4) Steps 2 and 3 are repeated until convergence is reached.

        """
        # Indices for conveniently splitting stacks into co- and counter-
        # propagating signals
        size = spectrum.shape[1]
        half_1 = slice(0, int(size / 2), 1)
        half_2 = slice(int(size / 2), size, 1)
        split_idx = (half_1, half_2)

        # Copy boundary conditions
        co_light = spectrum[:, half_1].copy()
        counter_light = spectrum[:, half_2].copy()

        # indices of signal and pump light within halves 1 and 2
        i_sig = slice(0, self.grid.points, 1)
        i_pump = slice(self.grid.points, size, 1)

        # Starting complex spectrum.
        # FFT_scale required to properly scale ASE contribution for field.
        # This is faster than scaling the complex spectrum.
        # FFT_scale = np.sqrt(2 * np.pi / self.grid.dt**2)
        dz = self.L / self.num_steps
        sample_interval = int(self.num_steps / num_samples)
        err = []

        if self.verbose:
            print('\nConvergence error (full field):')
        # START BOUNDARY CONDITION SOLVER
        for j in range(self.num_iters):
            if (j % 2) == 0:  # co signals static, counter signals updated
                static = split_idx[0]
                update = split_idx[1]
            else:             # counter signals static, co signals updated
                static = split_idx[1]
                update = split_idx[0]
            self.forward_propagation = not (self.forward_propagation)
            # Direction needs reversing and sampled spectra need to be flipped;
            # Signals are static if they are being propagated backwards
            self.stacks.directions *= -1
            samples = samples[::-1, :, :]

            # Get updates and replace update indices in spectra array
            if self.forward_propagation:
                ufft = utils.fft(field, axis=-1)
            N2 = np.zeros((self.num_steps))
            propagated = 0

            if sampling and self.forward_propagation:
                take_samples = True  # False later unless solution is close.
                field_samples = []  # required for pulse info
                dz_samples = []  # required for axes
                spectrum_samples = []  # required for pump/ASE spectral info

            # START PROPAGATION
            for i in range(self.num_steps):
                if propagated + dz > self.L:
                    dz = self.L - propagated

                # RK4IP z --> z + dz / 2
                if self.forward_propagation:
                    phasematching = self._DFWM_phasematching(propagated)
                    ufft = self._RK4IP(phasematching, dz / 2., ufft)

                # Gain z --> z + dz
                spectrum[:, static] = samples[i, :, static]
                N2[i] = self._get_inversion_two_level(
                    spectrum, self.stacks.inversion_abs_coeff,
                    self.stacks.inversion_em_coeff, repetition_rate)
                g = self._get_gain_two_level(
                    N2[i], self.stacks.overlaps[update],
                    self.stacks.em_cs[update], self.stacks.abs_cs[update])
                ase = self._ASE_energy_term_Giles_model(
                    self.stacks.ASE_coeff[update], N2[i])

                # Gain z --> z + dz for the field AND RK4IP z --> z + dz / 2
                if self.forward_propagation:
                    spec_update = spectrum[:, update]
                    dir_update = self.stacks.directions[update]
                    ufft = self._propagation_func(
                        ufft, dz, g[i_sig], dir_update[i_sig])
                    ufft = self._add_ASE_field(
                        ufft, ase[i_sig], dz, dir_update[i_sig])
                    phasematching = self._DFWM_phasematching(propagated)
                    ufft = self._RK4IP(phasematching, dz / 2., ufft)

                    # Update spectrum
                    aux_pump = spec_update[:, i_pump] * \
                        (1 + dz * dir_update[i_pump] * g[i_pump]) \
                        + dir_update[i_pump] * ase[i_pump] * dz
                    aux_spec = (ufft.real**2 + ufft.imag**2) \
                        / self.grid.FFT_scale**2
                    spectrum[:, update] = np.hstack((aux_spec, aux_pump))
                    output_ufft = ufft

                    # Sample
                    if sampling:
                        if (take_samples
                                and (i == 0 or i % sample_interval == 0)):
                            dz_samples.append(dz * sample_interval)
                            field_samples.append(
                                utils.ifft(ufft, axis=-1).copy())
                            spectrum_samples.append(spectrum.copy())

                else:  # counter propagation
                    spectrum[:, update] = spectrum[:, update] * \
                        (1 + dz * self.stacks.directions[update] * g) \
                        + self.stacks.directions[update] * ase * dz

                samples[i + 1, :, update] = spectrum[:, update]
                propagated += dz

            # Enforce boundary conditions for static indices, assess error
            # odd (forward) iteration, and terminate if err < tol or if no
            # convergence is detected.
            if (j % 2) == 0:
                spectrum[:, static] = co_light
                samples[-1, :, static] = co_light
                counter_at_input = samples[-1, :, update]
            else:
                spectrum[:, static] = counter_light
                samples[-1, :, static] = counter_light
                co_at_output = samples[-1, :, update]
                err.append(np.sum(np.abs(prev_N2 - N2) / N2))
                if self.verbose:
                    print('\t', err[-1])
                if len(err) > 5:  # Dodge RuntimeWarning: Mean of empty slice
                    diff_err = np.diff(err)
                    mean_diff_err = np.mean(diff_err)
                if err[-1] < self.convergence_tol:
                    break
                if j >= 50 and mean_diff_err >= 0:
                    msg = (
                        "\n\nThe propagation method isn't converging after %d "
                        "iterations.\nPlease revise the pump, fibre, or "
                        "signal parameters." % j)
                    raise exc.PropagationMethodNotConvergingError(msg)
                else:
                    prev_N2 = N2
                    if sampling and min(err) <= 10 * self.convergence_tol:
                        take_samples = True
                    elif sampling and min(err) >= 10 * self.convergence_tol:
                        take_samples = False

            if self.forward_propagation and np.any(np.isnan(ufft)):
                msg = ("Propagation results in NaN field elements. Please"
                       " decrease the propagation tolerance or change the"
                       " amplifier parameters.")
                raise exc.NanFieldError(msg)

        if sampling:
            dz_samples.append(dz * sample_interval)
            field_samples.append(utils.ifft(ufft, axis=-1).copy())
            spectrum_samples.append(spectrum.copy())
            return np.append(co_at_output, counter_at_input, axis=1), \
                utils.ifft(output_ufft, axis=-1), samples, N2, err, \
                dz_samples, field_samples, np.asarray(spectrum_samples)
        else:
            return np.append(co_at_output, counter_at_input, axis=1), \
                utils.ifft(output_ufft, axis=-1), samples, N2, err

    def _propagate_co_signal_field(
            self, field, repetition_rate, dz, sampling=False,
            sample_interval=1e-2):
        """
        Apply passive linear and nonlinear operators and gain iteratively
        over the fibre length. Frequency-domain gain only. Partial ASE only.

        Parameters
        ----------
        field : complex numpy array
            Time-domain field distribution.
        repetition_rate : float
            repetition_rate of the input pulses.
        sampling : bool
            Samples drawn at 10% of the fibre length.
        sample_interval : float
            Distance between sampling points.

        Returns
        -------
        if sampling:
            numpy array
                Pulse field at the signal output end of the fibre.
            float
                Final propagation step size
            list
                Pulse field as a function of propagation distance
            list
                Points along the fibre length at which pulse field samples are
                taken.
        else:
            numpy array
                Pulse field at teh signal outptu end of the fibre.
            float
                Final propagation step size
        """
        propagated_distance = 0
        size = self.stacks.spectra.shape[1]
        ufft = utils.fft(field, axis=-1)
        i_sig = slice(0, self.grid.points, 1)
        i_pump = slice(self.grid.points, size, 1)

        if sampling:
            sample_count = 1  # Keep track of number of samples taken.
            field_samples = []
            dz_samples = []
            if self.L <= sample_interval:
                sample_interval = self.L / 2.  # Minimum of 2 samples
            sample = False

        while propagated_distance < self.L:
            dz_updated = dz  # Returned so that pulse.dz can be updated.
            if propagated_distance + dz > self.L:
                dz = self.L - propagated_distance

            if sampling:
                if propagated_distance > sample_count * sample_interval:
                    dz = propagated_distance % sample_interval
                    sample = True

            phasematching = self._DFWM_phasematching(propagated_distance)

            # RK4IP: z --> z + dz/2
            start_photon_number = self._CQEM_photon_count_with_loss(
                ufft, dz / 2.)
            aux_ufft = self._RK4IP(phasematching, dz, ufft)
            finish_photon_number = self._CQEM_photon_count(aux_ufft)
            ufft = aux_ufft
            error = abs(start_photon_number - finish_photon_number) \
                / start_photon_number
            aux_dz_1 = self._CQEM_adjust_dz_only(error, dz)

            # GAIN: z --> z + dz
            N2 = self._get_inversion_two_level(
                self.stacks.spectra, self.stacks.inversion_abs_coeff,
                self.stacks.inversion_em_coeff, repetition_rate)
            g = self._get_gain_two_level(
                N2, self.stacks.overlaps, self.stacks.em_cs,
                self.stacks.abs_cs)
            ase = self._ASE_energy_term_Giles_model(
                self.stacks.ASE_coeff, N2)
            ufft = self._propagation_func(ufft, dz, g[i_sig])
            ufft = self._add_ASE_field(ufft, ase[i_sig], dz)

            # Update spectrum
            aux_pump = self.stacks.spectra[:, i_pump] * (1 + dz * g[i_pump]) \
                + ase[i_pump] * dz
            aux_spec = (ufft.real**2 + ufft.imag**2) / self.grid.FFT_scale**2
            self.stacks.spectra = np.hstack((aux_spec, aux_pump))

            # RK4IP: z + dz/2 --> z + dz
            start_photon_number = self._CQEM_photon_count_with_loss(
                ufft, dz / 2.)
            aux_ufft = self._RK4IP(phasematching, dz, ufft)
            finish_photon_number = self._CQEM_photon_count(aux_ufft)
            ufft = aux_ufft
            error = abs(start_photon_number - finish_photon_number) \
                / start_photon_number
            aux_dz_2 = self._CQEM_adjust_dz_only(error, dz)

            if sampling:
                if sample:
                    dz_samples.append(sample_interval)
                    field_samples.append(utils.ifft(ufft, axis=-1))
                    sample_count += 1
                    sample = False

            # Apply CQEM adjustment to step size for next iteration.
            propagated_distance += dz
            dz = (aux_dz_1 + aux_dz_2) / 2.

            if np.any(np.isnan(ufft)):
                if sampling:
                    return (np.ones_like(ufft) * np.nan, dz_updated,
                            field_samples, dz_samples)
                else:
                    return np.ones_like(ufft) * np.nan, dz_updated
        if sampling:
            dz_samples.append(sample_interval)
            field_samples.append(utils.ifft(ufft, axis=-1))
            return (utils.ifft(ufft, axis=-1), dz_updated, field_samples,
                    dz_samples)
        else:
            return utils.ifft(ufft, axis=-1), dz_updated

    def _co_signal_propagation(self, pulse, pulse_field, dz):
        """
        Propagation for co-propagating signals only, organise data samples if
        sampling, and return an updated pulse object.

        Parameters
        ----------
        pulse : pyLaserPulse.pulse.pulse
        pulse_field : numpy array
            pulse.field
        dz : float
            Initial proapgation step size.

        Returns
        -------
        pyLaserPulse.pulse.pulse
        """
        if pulse.high_res_samples:
            pass
        else:
            pulse_field, dz = self._propagate_co_signal_field(
                pulse_field, pulse.repetition_rate, dz)
        pulse.field = pulse_field
        pulse.dz = dz
        return pulse

    def propagate(self, pulse):
        """
        Apply passive linear and nonlinear operators and gain over iteratively
        over the fibre length.

        Parameters
        ----------
        pulse : pyLaserPulse.pulse.pulse

        Returns
        -------
        pyLaserPulse.pulse.pulse

        Raises
        ------
        Exception
            Full boundary value solver not yet supported for oscillator
            simulations (amplifier only)

        Notes
        -----
        oscillator=True: False if amplifier only (ie., single-pass).
        """
        # Propagate the pulse according to user settings
        if pulse.dz > 1e-2:
            pulse.dz = 1e-2

        if self.boundary_value_solver:
            # Integration tolerance.
            # convergence_tol <= 1 gives a point-to-point difference in N2 of
            # 1e-4 -- 1e-7 between subsequent interations.
            self.convergence_tol = .1
            self.num_iters = 100

            # Create counter-pulse object (back reflections etc)
            self.counter_pulse = pls.pulse(
                1e-12, [0, 0], 'Gauss', pulse.repetition_rate, self.grid)

            # ESD for signal
            ufft = utils.fft(pulse.field, axis=-1)
            sig_ESD = (ufft.real**2 + ufft.imag**2) \
                * self.grid.dt**2 / (2 * np.pi)

            # Get array stacks
            self.stacks = self._stack_propagation_arrays_boundary_value_solver(
                sig_ESD, pulse.ASE_scaling, pulse.repetition_rate)

            # Keep track of which iterations are forwards and which are
            # backwards. Starting  value of True is replaced with False if
            # counter/bidirectional pumping.
            # NB: This parameter is passed to the boundary condition solvers
            # by reference, so is modified by them.
            self.forward_propagation = True

            # # If counter or bidirectional pumping, start with counter signals
            # # Swap first and second halves of all stacks except directions
            # # and slices
            # if (np.sum(self.counter_pump.spectrum)
            #         >= np.sum(self.pump.spectrum)):
            #     self.forward_propagation = False
            #     for v in vars(self.stacks):
            #         if v != 'slices' and v != 'directions':
            #             self.stacks.__dict__[v] = utils.swap_halves(
            #                 self.stacks.__dict__[v])

            if pulse.high_res_samples and pulse.num_samples > self.num_steps:
                self.num_steps = pulse.num_samples

            # Solve for ESDs to estimate spectra and inversion along fibre
            # stacks.spectra.copy() as passed by reference
            pulse.dz = self.dz
            stack_spec = self.stacks.spectra.copy()
            ESD_at_signal_input, spec_samples, inversion, ESD_err = \
                self._solve_boundaries_ASE_ESD_only(
                    stack_spec, pulse.repetition_rate)

            # Use spectra and inversion as a starting point for resolving the
            # co- and counter-propagating light with nonlinear and dispersive
            # operators.
            # Tolerance of 0.1 gives 1:1e-6 error for N2 at each step.
            stack_spec = self.stacks.spectra.copy()
            if pulse.high_res_samples:
                ESD_at_signal_input, pulse.field, spec_samples, N2, \
                    field_err, self.dz_samples, field_samples, \
                    self.spectral_samples = \
                    self._propagate_boundary_value_solver_field(
                        pulse.field, pulse.repetition_rate, spec_samples,
                        stack_spec, inversion, sampling=pulse.high_res_samples,
                        num_samples=self.num_steps)
                pulse.high_res_field_samples += field_samples
                pulse.high_res_rep_rate_samples += \
                    [pulse.repetition_rate] * len(self.dz_samples)
                tmp_samples = np.sum(np.abs(field_samples)**2, axis=1)
                self.B_samples = np.cumsum(
                    self.n2 * np.amax(tmp_samples, axis=-1))
                self.B_samples *= np.asarray(self.dz_samples) * 2 * np.pi
                self.B_samples /= (
                    self.grid.lambda_c
                    * self.signal_mode_area[self.grid.midpoint])

                if len(pulse.high_res_B_integral_samples) != 0:
                    cumulative_B = \
                        self.B_samples + pulse.high_res_B_integral_samples[-1]
                else:
                    cumulative_B = self.B_samples
                pulse.high_res_B_integral_samples += cumulative_B.tolist()
                pulse.high_res_field_sample_points += self.dz_samples

                # Make field samples in gain fibre an attribute.
                self.pulse_field_samples = np.asarray(field_samples)

                # Sampling for pump light
                self.pump.high_res_samples = \
                    utils.get_ESD_and_PSD(
                        self.pump.lambda_window,
                        spec_samples[:, :, self.stacks.slices['co_pump']],
                        pulse.repetition_rate)[1]
                self.pump.high_res_sample_points = self.dz_samples
                self.counter_pump.high_res_samples = \
                    utils.get_ESD_and_PSD(
                        self.counter_pump.lambda_window,
                        spec_samples[:, :, self.stacks.slices['counter_pump']],
                        pulse.repetition_rate)[1]
                self.counter_pump.high_res_sample_points = self.dz_samples
            else:
                ESD_at_signal_input, pulse.field, spec_samples, N2, \
                    field_err = \
                    self._propagate_boundary_value_solver_field(
                        pulse.field, pulse.repetition_rate, spec_samples,
                        stack_spec, inversion)

            # # Reverse the stack swap if counter or bidirectional pumping.
            # # (NB: input spectra required here).
            # if (np.sum(self.counter_pump.spectrum)
            #         >= np.sum(self.pump.spectrum)):
            #     for v in vars(self.stacks):
            #         if v != 'slices' and v != 'directions':
            #             self.stacks.__dict__[v] = utils.swap_halves(
            #                 self.stacks.__dict__[v])
            #     spec_samples = utils.swap_halves(spec_samples)
            #     spec_samples = spec_samples[::-1, :, :]
            #     if pulse.high_res_samples:
            #         self.spectral_samples = utils.swap_halves(
            #             self.spectral_samples)
            #     N2 = N2[::-1]

            # Unstack spec_samples for results, make optimization loss arrays
            # from lists
            self.pump.propagated_spectrum = \
                spec_samples[-1, :, self.stacks.slices['co_pump']]
            counter_signal_input = utils.fftshift(
                spec_samples[0, :, self.stacks.slices['counter_signal']],
                axes=-1)
            self.counter_pump.propagated_spectrum = spec_samples[
                0, :, self.stacks.slices['counter_pump']]
            if self.cladding_pumping:
                self.co_core_ASE.propagated_spectrum = \
                    spec_samples[-1, :, self.stacks.slices['co_core_ASE']]
                self.counter_core_ASE.propagated_spectrum = \
                    spec_samples[0, :, self.stacks.slices['counter_core_ASE']]
            self.boundary_value_solver_ESD_optimization_loss = \
                np.asarray(ESD_err)
            self.boundary_value_solver_field_optimization_loss = \
                np.asarray(field_err)
            self.inversion_vs_distance = 100 * N2

            _, self.pump.starting_PSD = self.pump.get_ESD_and_PSD(
                self.pump.spectrum, pulse.repetition_rate)
            _, self.pump.propagated_PSD = self.pump.get_ESD_and_PSD(
                self.pump.propagated_spectrum, pulse.repetition_rate)
            _, self.counter_pump.starting_PSD \
                = self.counter_pump.get_ESD_and_PSD(
                    self.counter_pump.spectrum, pulse.repetition_rate)
            _, self.counter_pump.propagated_PSD \
                = self.counter_pump.get_ESD_and_PSD(
                    self.counter_pump.propagated_spectrum,
                    pulse.repetition_rate)
            self.counter_pulse.get_ESD_and_PSD_from_spectrum(
                self.grid, counter_signal_input)
            if self.cladding_pumping:
                _, self.co_core_ASE.starting_PSD = \
                    self.co_core_ASE.get_ESD_and_PSD(
                        self.co_core_ASE.spectrum, pulse.repetition_rate)
                _, self.co_core_ASE.propagated_PSD = \
                    self.co_core_ASE.get_ESD_and_PSD(
                        self.co_core_ASE.propagated_spectrum,
                        pulse.repetition_rate)
                _, self.counter_core_ASE.starting_PSD = \
                    self.counter_core_ASE.get_ESD_and_PSD(
                        self.counter_core_ASE.spectrum, pulse.repetition_rate)
                _, self.counter_core_ASE.propagated_PSD = \
                    self.counter_core_ASE.get_ESD_and_PSD(
                        self.counter_core_ASE.propagated_spectrum,
                        pulse.repetition_rate)
            if pulse.high_res_samples:
                # FFTshift signals. No extra requirements for cladding pumping.
                self.spectral_samples[
                    :, :, self.stacks.slices['co_signal']] = \
                    utils.fftshift(
                        self.spectral_samples[
                            :, :, self.stacks.slices['co_signal']],
                        axes=-1)
                self.spectral_samples[
                    :, :, self.stacks.slices['counter_signal']] = \
                    utils.fftshift(
                        self.spectral_samples[
                            :, :, self.stacks.slices['counter_signal']],
                        axes=-1)
        else:  # Co-signal propagation only
            # ESD for signal
            ufft = utils.fft(pulse.field, axis=-1)
            sig_ESD = (ufft.real**2 + ufft.imag**2) \
                * self.grid.dt**2 / (2 * np.pi)

            # Get array stacks
            self.stacks = self._stack_propagation_arrays_co_signal(
                sig_ESD, pulse.ASE_scaling, pulse.repetition_rate)
            pulse = self._co_signal_propagation(
                pulse, pulse.field, pulse.dz)
            self.pump.propagated_spectrum = \
                self.stacks.spectra[:, self.stacks.slices['co_pump']]

        if self.oscillator:  # Redefine pump each call so it is never depleted
            # if self.boundary_value_solver:
            #     raise Exception(
            #         "Full ASE simulations are currently not supported with"
            #         " mixed time and frequency domain gain.")
            # else:
            self.pump = pmp.pump(self.pump.bandwidth, self.pump.lambda_c,
                                    self.pump.energy, points=self.pump.points,
                                    lambda_lims=self.pump.lambda_lims,
                                    ASE_scaling=self.pump.ASE_scaling)
        return pulse


class loss_spectrum_base(ABC):
    """
    Abstract type for loss spectra that can be applied with a single-step
    propagator (e.g., components, but not fibres).
    """

    def __init__(self):
        pass

    def _include_partition_noise(self, photon_spectrum):
        """
        Return the transmission spectrum of the component after applying an
        additional operator that accounts for partition noise.

        Parameters
        ----------
        photon_spectrum : numpy array
            Number of photons in each frequency bin of the pulse spectrum.

        Returns
        -------
        numpy array
            Transmission as a function of frequency.

        Notes
        -----
        Seen lots of time-domain noise on the Intel-based Windows machines.
        The time-domain noise is not seen on openBLAS AMD Linux computers.
        """
        # BUG -- lots of time-domain noise for long time windows on both
        # Intel and AMD computers.
        return self.transmission_spectrum
        # if si.OS == 'Windows':
        #     return self.transmission_spectrum
        # elif si.OS == 'Linux':
        #     standard_dev = np.sqrt(
        #         photon_spectrum * self.transmission_spectrum *
        #         (1 - self.transmission_spectrum))
        #     transmission = np.random.normal(
        #         self.transmission_spectrum * photon_spectrum, standard_dev)
        #     transmission /= photon_spectrum
        #     transmission[transmission < 0] = 0
        #     return transmission


class component_base(loss_spectrum_base, ABC):
    """
    Abstract type for components.

    Contains all methods and members required to simulate time-frequency
    domain propagation effects of many standard optical components.
    """

    def __init__(self, loss, transmission_bandwidth, lambda_c, epsilon, theta,
                 beamsplitting, g, crosstalk, order=2, output_coupler=False,
                 coupler_type="polarization", beta_list=None, gdm=0):
        """
        Parameters
        ----------
        loss : float
            Insertion loss at the signal wavelength.
        transmission_bandwidth : float
            Transmission bandwidth. Can also be interpreted as reflection
            bandwidth for reflective optics.
        lambda_c : float
            Central wavelength of transmission window.
        epsilon : complex
            Defines type of component (polarizer or retarder).
            Defined for field. For example:
                epsilon = 0 + 1j for a quarter wave plate,
                epsilon = -1 for a half wave plate,
                epsilon = 0.1 for 20 dB polarization extinction.
        theta : float
            Angle subtended by component optical axis and x-axis.
        beamsplitting : float
            Intensity fraction remaining in pulse.field if output coupler
            and coupler_type="beamsplitter"
        g : pyLaserPulse.grid.grid object
        crosstalk : float
            Polarization degradation caused by the component.
        order : int
            Steepness of the super-Gaussian transmission window edges.
        output_coupler : boolean
            If True, the component is an output coupler and some of the
            main beam (pulse.field) is tapped off.
        coupler_type : string
            "polarization" or "beamsplitter". If the former, Jones matrix
            for polarization-based output coupler is used. If the latter,
            Jones matrix for polarization-independent beam splitting is
            used for the output coupler.
        beta_list : list
            None by default. Contains Taylor coefficients
            [beta_2, beta_3, ..., beta_n] which define the dispersion
            profile of the component.
        gdm : float
            Polarization group delay mismatch. Units: s.
            Delay accumulated between polarization components after
            propagating through the component. If gdm > 0, x is the slow
            axis.
        """
        super().__init__()
        self.loss = loss
        self.transmission_bandwidth = transmission_bandwidth
        self.lambda_c = lambda_c
        self.order = order
        self.epsilon = epsilon
        self.theta = theta
        self.grid = g
        self.make_transmission_spectrum()
        self.beamsplitting = beamsplitting
        self.make_Jones_matrix()
        self.crosstalk = crosstalk
        self.coupler_type = coupler_type
        self.output_coupler = output_coupler
        self.GDM = gdm
        if np.abs(self.GDM) > 0:
            self.make_birefringence()
        self.beta_list = beta_list
        if self.beta_list:
            self.make_dispersion()

        if self.coupler_type not in ('beamsplitter', 'polarization'):
            raise Exception("base_component.coupler_type should be"
                            " 'beamsplitter' or 'polarization'")
        if coupler_type not in ('beamsplitter', 'polarization'):
            raise Exception("base_components.coupler_type should be"
                            " 'beamsplitter' or 'polarization'")
        if self.beamsplitting < 0 or self.beamsplitting > 1:
            raise ValueError("Inappropriate value %f for beamsplitting,"
                             " which must be a floating point value between 0"
                             " and 1." % beamsplitting)
        if self.crosstalk < 0 or self.crosstalk > 1:
            raise ValueError("Inapproporate value %r for crosstalk,"
                             " which must be a floating point value between 0"
                             " and 1." % crosstalk)

    def make_verbose(self):
        """
        Change self.verbose to True
        Called by optical_assemblies. If the optical assembly verbosity is
        True, all component verbosities are also set to True.

        Notes
        -----
        Verbosity is not set in __init__ because there is little to report
        about standard component propagation. However, make_verbose is included
        as a method because some derived classes CAN make use of verbosty
        (e.g., grating compressors).
        """
        self.verbose = True

    def make_silent(self):
        """
        Change self.verbose to False
        Called by optical_assemblies. If the optical assembly verbosity is
        False, all component verbosities are also set to False.

        Notes
        -----
        Verbosity is not set in __init__ because there is little to report
        about standard component propagation. However, make_silent is included
        as a method because some derived classes CAN make use of verbosty
        (e.g., grating compressors).
        """
        self.verbose = False

    def make_transmission_spectrum(self):
        """
        Define the spectral transmission window of the component.
        """
        sigma = self.transmission_bandwidth / 2
        argument = (self.grid.lambda_window - self.lambda_c)**2 / sigma**2
        argument = argument**self.order
        self.transmission_spectrum \
            = np.exp(-1 * argument)[None, :].repeat(2, axis=0)
        self.transmission_spectrum *= (1 - self.loss)
        self.transmission_spectrum = utils.fftshift(
            self.transmission_spectrum, axes=1)

    def make_birefringence(self):
        """
        Use the GDM to make the birefringence operator.
        """
        self.birefringence = np.zeros((2, self.grid.points),
                                      dtype=np.complex128)
        self.birefringence[0, :] += -1j * self.GDM * self.grid.omega / 2
        self.birefringence[1, :] += 1j * self.GDM * self.grid.omega / 2
        self.birefringence = utils.fftshift(self.birefringence, axes=1)

    def make_dispersion(self):
        """
        Make the dispersion profile as a function of the grid omega_window.
        """
        self.dispersion = np.zeros(
            (self.grid.points), dtype=np.complex128)
        if self.beta_list is not None:
            self.beta_list.insert(0, 0)
            self.beta_list.insert(0, 0)  # GDM included elsewhere
            for i, b in enumerate(self.beta_list):
                self.dispersion += \
                    1j * b * self.grid.omega**i / np.math.factorial(i)
        self.dispersion = utils.fftshift(self.dispersion)
        self.dispersion = self.dispersion[None, :].repeat(2, axis=0)

    def apply_transmission_spectrum(self, field, photon_spectrum):
        """
        Apply the spectral transmission window of the component to the field.

        Parameters
        ----------
        field : numpy array
            Pulse field.
        photon_spectrum : numpy array
            Number of photonc in each frequency bin of field.

        Returns
        -------
        numpy array
            Pulse field after the transmission spectrum has been applied.
        """
        spectrum = utils.fft(field)
        trans_window = self._include_partition_noise(photon_spectrum)
        spectrum *= np.sqrt(trans_window)
        field = utils.ifft(spectrum)
        return field

    def apply_dispersion(self, field):
        """
        Apply component dispersion to the field.

        Parameters
        ----------
        field : numpy array
            Pulse field.

        Returns
        -------
        numpy array
            Pulse field after the dispersion has been applied.
        """
        spectrum = utils.fft(field)
        spectrum *= np.exp(-1j * self.dispersion.imag)
        field = utils.ifft(spectrum)
        return field

    def apply_birefringence(self, field):
        """
        Apply component birefringence to the field.

        Parameters
        ----------
        field : numpy array
            Pulse field

        Returns
        -------
        numpy array
            Pulse field after the birefringence has been applied.
        """
        spectrum = utils.fft(field)
        spectrum *= np.exp(-1j * self.birefringence.imag)
        field = utils.ifft(spectrum)
        return field

    def make_Jones_matrix(self):
        """
        Define the Jones matrix for the component. Handles pol.-dependent
        retardation and attenuation.
        """
        sin2 = np.sin(self.theta)**2
        cos2 = np.cos(self.theta)**2
        sincos = np.sin(self.theta) * np.cos(self.theta)

        real00 = cos2 + self.epsilon.real * sin2
        real10 = sincos - self.epsilon.real * sincos
        real01 = sincos - self.epsilon.real * sincos
        real11 = sin2 + self.epsilon.real * cos2

        imag00 = self.epsilon.imag * sin2
        imag10 = -1 * self.epsilon.imag * sincos
        imag01 = -1 * self.epsilon.imag * sincos
        imag11 = self.epsilon.imag * cos2

        self.Jones = np.array([[real00 + 1j * imag00, real01 + 1j * imag01],
                               [real10 + 1j * imag10, real11 + 1j * imag11]],
                              dtype=np.complex128)

    def apply_Jones_matrix(self, field):
        """
        Apply the Jones matrix to the field.

        Parameters
        ----------
        field : numpy array
            Pulse field.

        Returns
        -------
        numpy array
            Pulse field after the Jones matrix has been applied.
        """
        if self.output_coupler and self.coupler_type == "beamsplitter":
            return np.matmul(
                self.Jones * np.sqrt(self.beamsplitting), field)
        else:
            return np.matmul(self.Jones, field)

    @classmethod
    def propagator(cls, func):
        """
        Field propagation decorator.

        Parameters
        ----------
        func : propagate method of the derived class.

        Notes
        -----
        All derived classes should propagate the pulse through this method.
        If no additional functionality is required by the method func in
        the derived class, use the following syntax:

            @_component_base.propagator
            def propagate(self, pulse):
                return pulse

        If additional functionality is required, use the following syntax:

            @_component_base.propagator
            def propagate(self, pulse):
                self.another_method(pulse)
        """

        def wrapper(self, pulse):
            pulse = func(self, pulse)
            return self._propagate(pulse)
            # pulse = self._propagate(pulse)
            # return func(self, pulse)
        return wrapper

    def _propagate(self, pulse):
        """
        Apply the default component operator to an input field.

        Parameters
        ----------
        pulse : pyLaserPulse.pulse.pulse

        Returns
        -------
        pyLaserPulse.pulse.pulse

        Notes
        -----
        Internal use by _component_base only. For propagation, please see
        component_base.propagator decorator.
        """
        pulse.field = self.apply_transmission_spectrum(
            pulse.field, pulse.photon_spectrum)
        if self.beta_list:
            pulse.field = self.apply_dispersion(pulse.field)
        if self.GDM:
            pulse.field = self.apply_birefringence(pulse.field)
        tmp_field = self.apply_Jones_matrix(pulse.field.copy())
        if self.output_coupler:
            # See pg 67 -- 76 of note book starting 28/2/2021
            if self.coupler_type == "beamsplitter":
                a = np.sqrt(1 - self.beamsplitting)
                OC_matrix = np.array([[(a * self.Jones[0, 0]),
                                       -1 * a * self.Jones[0, 1]],
                                      [-1 * a * self.Jones[1, 0],
                                       (a * self.Jones[1, 1])]],
                                     dtype=np.complex128)
            elif self.coupler_type == "polarization":
                OC_matrix = np.array([[(1 - self.Jones[0, 0]),
                                       -1 * self.Jones[0, 1]],
                                      [-1 * self.Jones[1, 0],
                                       (1 - self.Jones[1, 1])]],
                                     dtype=np.complex128)
            out = np.matmul(OC_matrix, pulse.field)
            out_prime = np.sqrt(1 - self.crosstalk) * out \
                + np.sqrt(self.crosstalk) * out[::-1, :]
            pulse.output.append(out_prime)

        # Manage crosstalk
        xtalk = np.array(
            [[1 - self.crosstalk, self.crosstalk],
             [self.crosstalk, 1 - self.crosstalk]])
        xtalk = np.sqrt(xtalk)
        tmp_field = np.matmul(xtalk, tmp_field)
        pulse.field = tmp_field

        # pulse.add_OPPM_noise(self.grid, topup=True)

        if pulse.high_res_samples:
            pulse.high_res_field_samples.append(pulse.field)
            pulse.high_res_rep_rate_samples.append(pulse.repetition_rate)
            if len(pulse.high_res_B_integral_samples) > 0:
                pulse.high_res_B_integral_samples.append(
                    pulse.high_res_B_integral_samples[-1])
            else:
                pulse.high_res_B_integral_samples.append(0)

            # sample_interval for bulk components doesn't have to be realistic,
            # just useful for plotting afterwards.
            interval = 0.05 * np.sum(pulse.high_res_field_sample_points)
            pulse.high_res_field_sample_points.append(interval)
        return pulse

    @classmethod
    def ESD_propagator(cls, func):
        """
        Energy spectral density propagation decorator.

        Parameters
        ----------
        func : propagate method of the derived class

        Notes
        -----
        All derived classes should propagate the ESD through this method.
        If no additional functionality is required by the method func in
        the derived class, use the following syntax:

            @_component_base.ESD_propagator
            def propagate_spectrum(self, pulse):
                pass

        If additional functionality is required, use the following syntax:

            @_component_base.ESD_propagator
            def propagate_spectrum(self, pulse):
                self.another_method(pulse)
        """

        def wrapper(self, spectrum, omega_axis):
            spectrum = self._propagate_spectrum(spectrum, omega_axis)
            return func(self, spectrum, omega_axis)
        return wrapper

    def _propagate_spectrum(self, spectrum, omega_axis):
        """
        Apply the full component operator to an input energy spectral density.
        This is useful for modelling the effect of a component on, for example,
        ASE and pump light.

        Parameters
        ----------
        spectrum : numpy array. dtype float64.
            shape(2, n_points). energy spectrum of the light.
        omega_axis : numpy array. float 64.
            shape(n_points). Angular frequency grid corresponding to spectrum.

        Notes
        -----
        Applies the transmission window and the Jones matrix, then recasts to
        float64.
        Internal use by _component_base only. For propagation, please use
        _component_base.propagator decorator.
        """
        trans_spec = np.zeros((2, len(omega_axis)))
        shifted_trans_spec = utils.fftshift(
            self.transmission_spectrum, axes=-1)

        # If spectrum omega grid is in self.grid.omega_window:
        if (self.grid.omega_window.min() < omega_axis.min()
           and self.grid.omega_window.max() > omega_axis.max()):
            for i in range(2):
                trans_spec[i, :] = np.interp(
                    omega_axis, self.grid.omega_window,
                    shifted_trans_spec[i, :])

        # If spectrum omega grid has values outside of self.grid.omega_window
        else:
            max_idx = utils.find_nearest(
                omega_axis.max(), self.grid.omega_window)[0]
            min_idx = utils.find_nearest(
                omega_axis.min(), self.grid.omega_window)[0]
            upper_val = shifted_trans_spec[:, max_idx]
            lower_val = shifted_trans_spec[:, min_idx]
            for i in range(2):
                trans_spec[i, :] = np.interp(
                    omega_axis, self.grid.omega_window,
                    shifted_trans_spec[i, :], left=lower_val[i],
                    right=upper_val[i])
        spectrum = np.sqrt(spectrum)
        spectrum *= trans_spec
        spectrum = np.matmul(self.Jones, spectrum)
        spectrum = np.abs(spectrum)**2
        return spectrum


class coupling_transmission_base(loss_spectrum_base):
    """
    Class for coupling loss between components. Used in optical_assemblies.py
    when components are 'assembled' into the full structure (see, for example,
    sm_fibre_laser.__init__).
    """

    def __init__(self, grid):
        """
        Parameters
        ----------
        grid : pyLaserPulse.grid.grid object.
        """
        super().__init__()
        self.grid = grid

    @abstractmethod
    def _make_transmission_spectrum():
        """
        Make the transmission spectrum.

        Notes
        -----
        Transmission spectrum must have shape [2, grid.points]
        """
        pass

    def propagate(self, pulse):
        """
        Apply the coupling loss to the pulse spectrum.

        Parameters
        ----------
        pulse : pyLaserPulse.pulse.pulse

        Returns
        -------
        pyLaserPulse.pulse.pulse
        """
        spectrum = utils.fft(pulse.field)
        trans_spectrum = self._include_partition_noise(pulse.photon_spectrum)
        spectrum *= np.sqrt(trans_spectrum)
        pulse.field = utils.ifft(spectrum)
        return pulse

    def propagate_spectrum(self, spectrum, omega_axis):
        """
        Apply the full coupling transmission operator to an input energy
        spectral density. This is useful for modelling the effect of, for
        example, a splice on ASE and pump light.

        Parameters
        ----------
        spectrum : numpy array, dtype float64.
            shape(2, n_points). Energy spectrum of the light.
        omega_axis : numpy array. dtype float64.
            Angular frequency grid corresponding to spectrum

        Returns
        -------
        numpy array
            Energy spectral density after the transmission spectrum has
            been applied.

        Notes
        -----
        This is the coupling_transmission equivalent of
        base_components.component.propagate_spectrum.
        """
        trans_spec = None
        shifted_trans_spec = utils.fftshift(self.transmission_spectrum)

        # If spectrum omega grid is in self.grid.omega_window:
        if (self.grid.omega_window.min() < omega_axis.min()
           and self.grid.omega_window.max() > omega_axis.max()):
            trans_spec = np.interp(
                omega_axis, self.grid.omega_window, shifted_trans_spec)

        # If spectrum omega grid has values outside of self.grid.omega_window
        else:
            max_idx = utils.find_nearest(
                omega_axis.max(), self.grid.omega_window)[0]
            min_idx = utils.find_nearest(
                omega_axis.min(), self.grid.omega_window)[0]
            upper_val = shifted_trans_spec[max_idx]
            lower_val = shifted_trans_spec[min_idx]
            trans_spec = np.interp(
                omega_axis, self.grid.omega_window, shifted_trans_spec,
                left=lower_val, right=upper_val)
        return spectrum * trans_spec
