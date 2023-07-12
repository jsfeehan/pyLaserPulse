#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:41:17 2020

@author: james feehan

Module of base classes for optical components.
"""

import numpy as np
import scipy.optimize as opt
import scipy.constants as const
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

import pyLaserPulse.abstract_bases as bases
import pyLaserPulse.utils as utils


class step_index_passive_fibre(bases.fibre_base):
    """
    Base class for passive step index fibres.

    Contains all methods and members required to simulate single-mode pulse
    propagation in passive optical fibre.

    self.get_signal_propagation_parameters overides the same method in
    bases.fibre_base and is specific to step-index passive fibres. It is based
    on:
        D. Gloge, "Weakly guiding fibres", Applied Optics 10(10),
        pp 2252--2258 (1971)
    """

    def __init__(self, g, length, loss_file, Raman_file, core_diam, NA,
                 beat_length, n2, fR, tol, Sellmeier_file, verbose=False):
        """
        Parameters
        ----------
        g : pyLaserPulse.grid.grid object
        length : float
            Fibre length in m
        loss_file : string
            Absolute path to the fibre loss as a function of wavelength.
            See pyLaserPulse.data.paths.materials.loss_spectra
        Raman_file : string
            Absolute path to the Raman response as a function of time.
            See pyLaserPulse.data.paths.materials.Raman_profiles
        core_diam : float
            Core diameter in m
        NA : float
            Numerical aperture
        beat_length : float
            Polarization beat length in m
        n2 : float
            Nonlinear index in m^2 / W
        fR : float
            Fractional Raman contribution to the fibre nonlinear response
            (e.g., fR = 0.18 for silica)
        tol : float
            Maximum propagation error used to adjust the proapgation step size
        Sellmeier_file : string
            Absolute path to the Sellmeier coefficients.
            See pyLaserPulse.data.paths.materials.Sellmeier_coefficients.
        verbose : bool
            Print information to terminal if True
        """
        super().__init__(
            g, length, loss_file, Raman_file, beat_length, n2, fR, tol,
            verbose=verbose)
        self.core_diam = core_diam
        self.NA = NA
        self.cladding_ref_index = utils.Sellmeier(
            self.grid.lambda_window, Sellmeier_file)
        self.signal_ref_index = (NA**2 + self.cladding_ref_index**2)**0.5
        self.delta_n = self.signal_ref_index[g.midpoint] \
            - self.cladding_ref_index[g.midpoint]
        self.get_propagation_parameters()
        self.get_GNLSE_and_birefringence_parameters()

    def get_propagation_parameters(self):
        """
        Calculate NA, V, beta_2, D, effective_MFD, signal_mode_area, gamma

        Notes
        -----
        D. Gloge, "Weakly guiding fibres", Applied Optics 10(10),
        pp 2252--2258 (1971)
        """
        k = 2 * np.pi / self.grid.lambda_window
        self.NA = np.sqrt(self.signal_ref_index**2
                          - self.cladding_ref_index**2)
        self.V = k * self.core_diam * self.NA / 2

        delta = (self.signal_ref_index - self.cladding_ref_index) \
            / self.cladding_ref_index
        u = (1 + np.sqrt(2)) * self.V / (1 + (4 + self.V**4)**0.25)
        beta = k * (1 + delta - delta * (u**2 / self.V**2))
        b = ((beta / (k / self.cladding_ref_index)) -
             self.cladding_ref_index) / (self.delta_n)

        decimate = 1
        k_vac = k / self.signal_ref_index
        if self.grid.points > 1024:  # i.e., grid size is > 1024
            # Required because very fine grids can result in noisy gradient
            # calculations
            decimate = int(self.grid.points / 1024)

        part_1 = np.gradient(k[::decimate], k_vac[::decimate], edge_order=2)
        part_2 = np.gradient(
            self.V[::decimate] * b[::decimate], self.V[::decimate],
            edge_order=2)
        part_2 *= self.cladding_ref_index[::decimate] * delta[::decimate]

        beta_1 = (part_1 + part_2) / const.c
        self.beta_2 = np.gradient(beta_1, self.grid.omega_window[::decimate],
                                  edge_order=2)

        if decimate > 1:  # Interpolate D and beta_2 onto original grid
            # Some noise still present for *really* fine grids, so do some
            # smoothing as well
            self.beta_2 = savgol_filter(self.beta_2, 4, 1)
            f = interp1d(self.grid.lambda_window[::decimate], self.beta_2,
                         kind='linear', fill_value='extrapolate')
            self.beta_2 = f(self.grid.lambda_window)

        self.D = -2 * np.pi * const.c * self.beta_2 / self.grid.lambda_window**2


class photonic_crystal_passive_fibre(bases.fibre_base):
    """
    Base class for passive photonic crystal fibres.

    Contains all methods and members required to simulate single-mode pulse
    propagation in passive optical fibre.

    self.get_signal_propagation_parameters overides the same method in
    bases.fibre_base and is specific to hexagonal-lattice photonic crystal
    passive fibres. It is based on:
        K. Saitoh et al., "Empirical relations for simple design of
        photonic crystal fibres", Opt. Express 13(1), 267--274 (2005).
    """

    def __init__(
            self, g, length, loss_file, Raman_file, hole_pitch,
            hole_diam_over_pitch, beat_length, n2, fR, tol, Sellmeier_file,
            verbose=False):
        """
        Parameters
        ----------
        g : pyLaserPulse.grid.grid object
        length : float
            Fibre length in m
        loss_file : string
            Absolute path to the fibre loss as a function of wavelength.
            See pyLaserPulse.data.paths.materials.loss_spectra
        Raman_file : string
            Absolute path to the Raman response as a function of time.
            See pyLaserPulse.data.paths.materials.Raman_profiles
        hole_pitch : float
            Separation of neighbouring air holes in the hexagonal-lattice PCF
            structure.
        hole_diam_over_pitch : float
            Ratio of the air hole diameter to the hole pitch.
        beat_length : float
            Polarization beat length in m
        n2 : float
            Nonlinear index in m^2 / W
        fR : float
            Fractional Raman contribution to the fibre nonlinear response
            (e.g., fR = 0.18 for silica)
        tol : float
            Maximum propagation error used to adjust the proapgation step size
        Sellmeier_file : string
            Absolute path to the Sellmeier coefficients.
            See pyLaserPulse.data.paths.materials.Sellmeier_coefficients.
        """
        super().__init__(
            g, length, loss_file, Raman_file, beat_length, n2, fR, tol,
            verbose=verbose)
        self.hole_diam = hole_diam_over_pitch
        self.hole_pitch = hole_pitch
        self.core_diam = 2 * self.hole_pitch / np.sqrt(3)
        self.core_radius = self.core_diam / 2
        self.Sellmeier_file = Sellmeier_file
        self.material_ref_index = utils.Sellmeier(
            g.lambda_window, self.Sellmeier_file)

        # Matrices required for propagation parameter calculations
        self.a = np.array((
            (0.54808, 0.71041, 0.16904, -1.52736),
            (5.00401, 9.73491, 1.85765, 1.06745),
            (-10.43248, 47.41496, 18.96849, 1.93229),
            (8.22992, -437.50962, -42.4318, 3.89)))
        self.b = np.array((
            (5, 1.8, 1.7, -0.84),
            (7, 7.32, 10, 1.02),
            (9, 22.8, 14, 13.4)))
        self.c = np.array((
            (-0.0973, 0.53193, 0.24876, 5.29801),
            (-16.70566, 6.70858, 2.72423, 0.05142),
            (67.13845, 52.04855, 13.28649, -5.18302),
            (-50.25518, -540.66947, -36.80372, 2.7641)))
        self.d = np.array((
            (7, 1.49, 3.85, -2),
            (9, 6.58, 10, 0.41),
            (10, 24.8, 15, 6)))

        self.get_propagation_parameters(
            g.lambda_window, g.midpoint, g.omega_window)
        self.get_GNLSE_and_birefringence_parameters()

    def get_propagation_parameters(
            self, lambda_window, grid_midpoint, omega_window):
        """
        Calculate signal_ref_index, D, beta_2, effective_MFD,
        signal_mode_area, gamma

        Parameters
        ----------
        lambda_window : numpy array
            Wavelength grid in m. See pyLaserPulse.grid.grid.lambda_window
        grid_midpoint : int
            Middle index of the time-frequency grid.
            See pyLaserPulse.grid.grid.midpoint
        omega_window : numpy array
            Angular frequency grid in rad Hz.
            See pyLaserPulse.grid.grid.omega_window

        Returns
        -------
        numpy array
            V-number as a function of wavelength
        numpy array
            Refractive index as a function of wavelength
        numpy array
            Fibre dispersion in ps / (nm km)
        numpy array
            Fibre dispersion in s^2 / m
        """
        self.V, self.signal_ref_index, self.D, self.beta_2 = \
            utils.PCF_propagation_parameters_K_Saitoh(
                lambda_window, grid_midpoint, omega_window, self.a, self.b,
                self.c, self.d, self.hole_pitch, self.hole_diam,
                self.core_radius, self.Sellmeier_file)


class step_index_active_fibre(
        bases.active_fibre_base, step_index_passive_fibre):
    """
    Class for step-index active fibres.
    """

    def __init__(self, g, length, loss_file, Raman_file, core_diam,
                 NA, beat_length, n2, fR, tol, doping_concentration,
                 cross_section_file, seed_repetition_rate, pump_points,
                 ASE_wl_lims, Sellmeier_file, boundary_conditions,
                 lifetime=1.5e-3, cladding_pumping={}, time_domain_gain=False,
                 verbose=False):
        """
        Parameters
        ----------
        g : pyLaserPulse.grid.grid object
        length : float
            Fibre length in m
        loss_file : string
            Absolute path to the fibre loss as a function of wavelength.
            See pyLaserPulse.data.paths.materials.loss_spectra
        Raman_file : string
            Absolute path to the Raman response as a function of time.
            See pyLaserPulse.data.paths.materials.Raman_profiles
        core_diam : float
            Core diameter in m
        NA : float
            Numerical aperture
        beat_length : float
            Polarization beat length in m
        n2 : float
            Nonlinear index in m^2 / W
        fR : float
            Fractional Raman contribution to the fibre nonlinear response
            (e.g., fR = 0.18 for silica)
        tol : float
            Maximum propagation error used to adjust the proapgation step size
        doping_concentration : float
            Number density of the active ions in m^-3
        cross_section_file : string
            Absolute path to the emission and absorption cross sections as a
            function of wavelength.
            See pyLaserPulse.data.paths.fibres.cross_sections.
        seed_repetition_rate : float
            Repetition rate of the seed laser.
            See pyLaserPulse.pulse.pulse.repetition_rate
        pump_points : int
            Number of points in the pump/ASE spectrum window.
        ASE_wl_lims : list
            [min_wavelength, max_wavelength] for the pump/ASE wavelength
            window in m.
        Sellmeier_file : string.
            Absolute path to file containing Sellmeier coefficients for the
            fibre material.
            See pyLaserPulse.data.paths.materials.Sellmeier_coefficients.
        boundary_conditions : dict
            Set the boundary conditions for resolving the evolution of the
            pump, signal, and ASE light in both directions through the fibre.
            The type of simulation -- i.e., single-pass or full boundary value
            solver -- is determined by this dictionary.
            See the Notes section below for a full description of how this
            parameters is defined.
        lifetime : float
            Upper-state lifetime of the rare-earth dopant in s.
        cladding_pumping : dict.
            Parameters required to define a cladding-pumped fibre.
            Leave empty if core pumping. Otherwise, the following keys are
            required:
                pump_core_diam : float
                    Diameter of the pump cladding in m
                pump_delta_n : float
                    Refractive index difference between the fibre coating and
                    the pump cladding (usually n_silica - 1.375)
                pump_cladding_n : float
                    Refractive index of the fibre coating (usually 1.375)
        time_domain_gain : bool
            True if mixed-domain gain operator (i.e., both frequency- and
            time-domain gain) is to be used. If False, only frequency-
            domain gain is simulated.

        Notes
        -----
        The boundary_conditions dictionary accepts the following keys:
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

        If no counter-pump values are specified, then the propagation is
        completed in a single step and no effort is made to resolve
        counter-propagating signals (i.e., the boundary value problem is not
        solved). This allows for co-pumping only. Example:

            boundary_conditions = {
                'co_pump_power': 1, 'co_pump_wavelength': 976e-9,
                'co_pump_bandwidth': 1e-9}

        The boundary value problem is solved whenever dictionary key
        'counter_pump_power' is present, but it is not necessary to specify
        the wavelength or bandwidth of either the co- or counter-propagating
        pump if the power is zero. Here are a few examples of valid
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
        # Instantiate passive fibre
        step_index_passive_fibre.__init__(
            self, g, length, loss_file, Raman_file, core_diam, NA, beat_length,
            n2, fR, tol, Sellmeier_file, verbose=verbose)

        # Determine if core or cladding pumping and run checks on dictionaries
        # for cladding pumping and full ASE.
        self.cladding_pumping = cladding_pumping
        if self.cladding_pumping:
            key_list = ['pump_core_diam', 'pump_delta_n', 'pump_cladding_n']
            utils.check_dict_keys(
                key_list, cladding_pumping, 'cladding_pumping')
            self.pump_core_diam = self.cladding_pumping['pump_core_diam']
            self.pump_delta_n = self.cladding_pumping['pump_delta_n']
            self.pump_cladding_ref_index = \
                self.cladding_pumping['pump_cladding_n']
        else:
            self.pump_core_diam = self.core_diam
            self.pump_delta_n = self.delta_n

        # Instantiate active fibre
        bases.active_fibre_base.__init__(
            self, g, doping_concentration, cross_section_file,
            seed_repetition_rate, pump_points, ASE_wl_lims, Sellmeier_file,
            lifetime, cladding_pumping, time_domain_gain, boundary_conditions,
            verbose=verbose)

        # Get core ASE parameters if cladding pumping.
        if self.cladding_pumping:
            self.core_ASE_effective_MFD, self.core_ASE_mode_area = \
                self.get_pump_and_ASE_propagation_parameters(
                    self.Petermann_II, self.co_core_ASE.lambda_window,
                    self.core_ASE_ref_index, self.core_ASE_cladding_ref_index,
                    self.core_diam)
            self.core_ASE_overlaps = \
                self.get_overlaps_core_light(
                    self.co_core_ASE.points, self.co_core_ASE.lambda_window,
                    self.core_ASE_effective_MFD)

    @staticmethod
    def get_pump_and_ASE_propagation_parameters(
            P_II, lambda_window, core_index, cladding_index, core_diam):
        """
        Parameters
        ----------
        P_II : function.
            Petermann_II calculation. Must take V-number as an argument and
            return the ratio of mode and cladding diameters as a function of
            wavelength.
        lambda_window : numpy array
            Wavelength grid in m. See pyLaserPulse.grid.grid.lambda_window
        core_index : float
            Refractive index of the fibre core.
        cladding_index : float
            Refractive index of the fibre cladding.
        core_diam : float
            Diameter of the fibre core in m.

        Returns
        -------
        numpy array
            Effective mode field diameter in m as a function of wavelength
        numpy array
            Effective mode field area in m^2 as a fucntion of wavelength.
        """
        k = 2 * np.pi / lambda_window
        NA = np.sqrt(core_index**2 - cladding_index**2)
        V = k * core_diam * NA / 2
        p_II = P_II(V)
        effective_MFD = core_diam * p_II
        mode_area = np.pi * (effective_MFD / 2)**2
        return effective_MFD, mode_area

    def get_pump_refractive_index(self):
        """
        Calculate the effective refractive index for the pump light.
        """
        if self.cladding_pumping:
            self.pump_ref_index = \
                self.pump_cladding_ref_index + self.pump_delta_n
        else:
            self.pump_cladding_ref_index \
                = utils.Sellmeier(
                    self.pump.lambda_window, self.Sellmeier_file)
            self.pump_ref_index = \
                self.pump_cladding_ref_index + self.pump_delta_n
            self.pump_effective_MFD, self.pump_mode_area = \
                self.get_pump_and_ASE_propagation_parameters(
                    self.Petermann_II, self.pump.lambda_window,
                    self.pump_ref_index, self.pump_cladding_ref_index,
                    self.pump_core_diam)


class photonic_crystal_active_fibre(
        bases.active_fibre_base, photonic_crystal_passive_fibre):
    """
    Class for photonic crystal active fibres.
    """

    def __init__(
            self, g, length, loss_file, Raman_file, hole_pitch,
            hole_diam_over_pitch, beat_length, n2, fR, tol,
            doping_concentration, cross_section_file, seed_repetition_rate,
            pump_points, ASE_wl_lims, Sellmeier_file, boundary_conditions,
            lifetime=1.5e-3, cladding_pumping={}, time_domain_gain=False,
            verbose=False):
        """
        Parameters
        ----------
        g : pyLaserPulse.grid.grid object
        length : float
            Fibre length in m
        loss_file : string
            Absolute path to the fibre loss as a function of wavelength.
            See pyLaserPulse.data.paths.materials.loss_spectra
        Raman_file : string
            Absolute path to the Raman response as a function of time.
            See pyLaserPulse.data.paths.materials.Raman_profiles
        hole_pitch : float
            Separation of neighbouring air holes in the hexagonal-lattice PCF
            structure.
        hole_diam_over_pitch : float
            Ratio of the air hole diameter to the hole pitch.
        beat_length : float
            Polarization beat length in m
        n2 : float
            Nonlinear index in m^2 / W
        fR : float
            Fractional Raman contribution to the fibre nonlinear response
            (e.g., fR = 0.18 for silica)
        tol : float
            Maximum propagation error used to adjust the proapgation step size
        doping_concentration : float
            Number density of the active ions in m^-3
        cross_section_file : string
            Absolute path to the emission and absorption cross sections as a
            function of wavelength.
            See pyLaserPulse.data.paths.fibres.cross_sections.
        seed_repetition_rate : float
            Repetition rate of the seed laser.
            See pyLaserPulse.pulse.pulse.repetition_rate
        pump_points : int
            Number of points in the pump/ASE spectrum window.
        ASE_wl_lims : list
            [min_wavelength, max_wavelength] for the pump/ASE wavelength
            window in m.
        Sellmeier_file : string
            Absolute path to the Sellmeier coefficients.
            See pyLaserPulse.data.paths.materials.Sellmeier_coefficients.
        boundary_conditions : dict
            Set the boundary conditions for resolving the evolution of the
            pump, signal, and ASE light in both directions through the fibre.
            The type of simulation -- i.e., single-pass or full boundary value
            solver -- is determined by this dictionary.
            See the Notes section below for a full description of how this
            parameters is defined.
        lifetime : float
            Upper-state lifetime of the rare-earth dopant in s.
        cladding_pumping : dict.
            Parameters required to define a cladding-pumped fibre.
            Leave empty if core pumping. Otherwise, the following keys are
            required:
                pump_core_diam : float
                    Diameter of the pump cladding in m
                pump_delta_n : float
                    Refractive index difference between the fibre coating and
                    the pump cladding (usually n_silica - 1.375)
                pump_cladding_n : float
                    Refractive index of the fibre coating (usually 1.375)
        time_domain_gain : bool
            True if mixed-domain gain operator (i.e., both frequency- and
            time-domain gain) is to be used. If False, only frequency-
            domain gain is simulated.
        verbose : bool
            Print information to terminal if True

        Notes
        -----
        The boundary_conditions dictionary accepts the following keys:
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

        If no counter-pump values are specified, then no effort is made to
        resolve counter-propagating signals (i.e., the boundary value problem
        is not solved). This allows for co-pumping only. Example:

            boundary_conditions = {
                'co_pump_power': 1, 'co_pump_wavelength': 976e-9,
                'co_pump_bandwidth': 1e-9}

        The boundary value problem is solved whenever dictionary key
        'counter_pump_power' is present, but it is not necessary to specify
        the wavelength or bandwidth of either the co- or counter-propagating
        pump if the power is zero. Here are a few examples of valid
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
        # Instantiate passive fibre
        photonic_crystal_passive_fibre.__init__(
            self, g, length, loss_file, Raman_file, hole_pitch,
            hole_diam_over_pitch, beat_length, n2, fR, tol, Sellmeier_file,
            verbose=verbose)

        # Determine if core or cladding pumping and run checks on dictionaries
        # for cladding pumping and full ASE.
        self.cladding_pumping = cladding_pumping
        if self.cladding_pumping:
            key_list = ['pump_core_diam', 'pump_delta_n', 'pump_cladding_n']
            utils.check_dict_keys(
                key_list, cladding_pumping, 'cladding_pumping')
            self.pump_core_diam = self.cladding_pumping['pump_core_diam']
            self.pump_delta_n = self.cladding_pumping['pump_delta_n']
            self.pump_cladding_ref_index = \
                self.cladding_pumping['pump_cladding_n']
        else:
            self.pump_core_diam = self.core_diam  # photonic_crystal_passive

        # Instantiate active fibre
        bases.active_fibre_base.__init__(
            self, g, doping_concentration, cross_section_file,
            seed_repetition_rate, pump_points, ASE_wl_lims, Sellmeier_file,
            lifetime, cladding_pumping, time_domain_gain, boundary_conditions,
            verbose=verbose)

        # Get core ASE parameters if cladding pumping.
        if self.cladding_pumping:
            p_II = self.Petermann_II(self.core_ASE_V)
            self.core_ASE_effective_MFD, self.core_ASE_mode_area = \
                self.get_pump_and_ASE_propagation_parameters(
                    p_II, self.core_diam)
            # self.co_core_ASE.lambda_window, self.core_ASE_ref_index,
            # self.core_ASE_cladding_ref_index, self.core_diam)
            self.core_ASE_overlaps = \
                self.get_overlaps_core_light(
                    self.co_core_ASE.points, self.co_core_ASE.lambda_window,
                    self.core_ASE_effective_MFD)

    @staticmethod
    def get_pump_and_ASE_propagation_parameters(p_II, core_diam):
        """
        Parameters
        ----------
        p_II : numpy array
            Ratio of mode and core diameters given by Petermann II.
        core_diam : float
            Core diameter

        Returns
        -------
        numpy array
            Effective mode field diameter as a function of wavelength
        numpy array
            effective mode field area as a function of wavelength
        """
        effective_MFD = core_diam * p_II
        mode_area = np.pi * (effective_MFD / 2)**2
        return effective_MFD, mode_area

    def get_pump_refractive_index(self):
        """
        Calculate the effective refractive index for the pump light.
        """
        if self.cladding_pumping:
            self.pump_ref_index = \
                self.pump_cladding_ref_index + self.pump_delta_n
        else:
            V, self.pump_ref_index, _, _ = \
                self.get_propagation_parameters(
                    self.pump.lambda_window, self.pump.midpoint,
                    self.pump.omega_window)
            p_II = self.Petermann_II(V)
            self.pump_effective_MFD, self.pump_mode_area = \
                self.get_pump_and_ASE_propagation_parameters(
                    p_II, self.core_diam)


class component(bases.component_base):
    """
    Base class for defining optical components.
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
        super().__init__(
            loss, transmission_bandwidth, lambda_c, epsilon, theta,
            beamsplitting, g, crosstalk, order=order,
            output_coupler=output_coupler, coupler_type=coupler_type,
            beta_list=beta_list, gdm=gdm)

    @bases.component_base.propagator
    def propagate(self, pulse):
        """
        Apply the component operator to an input field.

        Parameters
        ----------
        pulse : pyLaserPulse.pulse.pulse

        Returns
        -------
        pyLaserPulse.pulse.pulse
        """
        return pulse

    @bases.component_base.ESD_propagator
    def propagate_spectrum(self, spectrum, omega_axis):
        """
        Apply the full component operator to an input energy spectral density.
        This is useful for modelling the effect of a component on, for example,
        ASE and pump light.

        Parameters
        ----------
        spectrum : numpy array, dtype float64.
            shape(2, n_points). Energy spectrum of the light.
        omega_axis : numpy array, dtype float64.
            Angular frequency axis.

        Returns
        -------
        numpy array
            spectrum after propagating through the component.

        Notes
        -----
        omega_axis not used directly here; required for
        component_base.ESD_propagator decoration.
        """
        return spectrum


class fibre_component():
    """
    Base class for fibre components.
    """

    def __init__(self, g, input_fibre, output_fibre, loss,
                 transmission_bandwidth, lambda_c, epsilon, theta,
                 beamsplitting, crosstalk, order=2, output_coupler=False,
                 coupler_type="beamsplitter", beta_list=None,
                 component_gdm=0, verbose=False):
        """
        Parameters
        ----------
        g : pyLaserPulse.grid.grid object
        input_fibre : optical fibre object
            Dfined using, for example:
            pyLaserPulse.base_components.step_index_passive_fibre
            pyLaserPulse.base_components.photonic_crystal_passive_fibre
            pyLaserPulse.base_components.step_index_active_fibre
            pyLaserPulse.base_components.photonic_crystal_active_fibre
            etc.
        output_fibre : optical fibre object
            Defined using, for example:
            pyLaserPulse.base_components.step_index_passive_fibre
            pyLaserPulse.base_components.photonic_crystal_passive_fibre
            pyLaserPulse.base_components.step_index_active_fibre
            pyLaserPulse.base_components.photonic_crystal_active_fibre
            etc.
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
        crosstalk : float
            Polarization degradation caused by the component.
        order : int
            Steepness of the super-Gaussian transmission window edges.
        output_coupler : bool
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
        verbose : bool
            Print information to terminal if True
        """
        self.input_fibre = input_fibre
        self.output_fibre = output_fibre
        self.component = component(loss, transmission_bandwidth, lambda_c,
                                   epsilon, theta, beamsplitting, g, crosstalk,
                                   order=order, output_coupler=output_coupler,
                                   coupler_type=coupler_type,
                                   beta_list=beta_list, gdm=component_gdm)
        if verbose:
            self.make_verbose()

    def make_verbose(self):
        """
        Change self.verbose to True
        Called by optical_assemblies. If the optical assembly verbosity is
        True, all component verbosities are also set to True.
        """
        self.input_fibre.make_verbose()
        self.component.make_verbose()
        self.output_fibre.make_verbose()

    def make_silent(self):
        """
        Change self.verbose to False
        Called by optical_assemblies. If the optical assembly verbosity is
        False, all component verbosities are also set to False.
        """
        self.input_fibre.make_silent()
        self.component.make_silent()
        self.output_fibre.make_silent()

    def propagate(self, pulse):
        """
        Apply the input_fibre, component, and output_fibre operators to a pulse

        Parameters
        ----------
        pulse : pyLaserPulse.pulse.pulse object

        Returns
        -------
        pyLaserPulse.pulse.pulse object
        """
        pulse = self.input_fibre.propagate(pulse)
        pulse = self.component.propagate(pulse)
        pulse = self.output_fibre.propagate(pulse)
        return pulse

    def propagate_spectrum(self, spectrum, omega_axis):
        """
        Apply the full component operator to an input energy spectral density.
        This is useful for modelling the effect of a component on, for example,
        ASE and pump light.

        Parameters
        ----------
        spectrum : numpy array, dtype float64.
            shape(2, n_points). Energy spectrum of the light.
        omega_axis : numpy array, dtype float64/
            Angular frequency axis corresponding to spectrum

        Returns
        -------
        numpy array
            Spectrum after propagating through the component.
        """
        spectrum = self.component.propagate_spectrum(spectrum, omega_axis)
        return spectrum


class pulse_picker(bases.component_base):
    """
    Class for AOMs, EOMs and, assuming the input repetition rate is low enough
    and only one pulse is let through, choppers as well.
    """
    def __init__(self, loss, transmission_bandwidth, lambda_c, epsilon, theta,
                 beamsplitting, g, crosstalk, time_open, rate_reduction_factor,
                 input_rep_rate, order=2, output_coupler=False,
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
        time_open : float
            Time in s that the pulse picker is 'open', i.e., lets light through
            See Notes below.
        rate_reduction_factor : int
            Ratio of the input and output repetition rates of the pulse picker.
            See Notes below.
        input_rep_rate : float
            Repetition rate of the seed laser before the pulse picker.
            See Notes below.
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
        super().__init__(loss, transmission_bandwidth, lambda_c, epsilon,
                         theta, beamsplitting, g, crosstalk, order,
                         output_coupler, coupler_type, beta_list, gdm)
        self.time_open = time_open
        if not isinstance(rate_reduction_factor, int):
            raise ValueError(
                "pulse_picker.rate_reduction_factor must be an integer")
        self.rate_reduction_factor = rate_reduction_factor
        self.input_rep_rate = input_rep_rate
        self.output_rep_rate = input_rep_rate / self.rate_reduction_factor
        self.duty_cycle = self.time_open * self.output_rep_rate
        self.make_temporal_transmission_window()

    def make_temporal_transmission_window(self):
        """
        Define the temporal transmission window of the component.

        Loss, including from diffraction, taken into account in the spectral
        domain.
        """
        sigma = self.time_open / 2
        argument = (self.grid.time_window)**2 / sigma**2
        argument = argument**10
        self.temporal_transmission_window \
            = np.exp(-1 * argument)[None, :].repeat(2, axis=0)

    def apply_temporal_transmission_window(self, field):
        """
        Apply the temporal transmission window of the component to the field.

        Parameters
        ----------
        field : numpy array
            Pulse field in the time domain.
            See pyLaserPulse.pulse.pulse.field

        Returns
        -------
        numpy array
            Pulse field in the time domain after the pulse picking.
        """
        field *= np.sqrt(self.temporal_transmission_window)
        return field

    @bases.component_base.propagator
    def propagate(self, pulse):
        """
        Apply the full component operator to an input field.

        Parameters
        ----------
        pulse : pyLaserPulse.pulse.pulse object

        Returns
        -------
        pyLaserPulse.pulse.pulse object
        """
        pulse.repetition_rate = pulse.repetition_rate / self.rate_reduction_factor
        pulse.field = self.apply_temporal_transmission_window(pulse.field)
        return pulse

    @bases.component_base.ESD_propagator
    def propagate_spectrum(self, spectrum, omega_axis):
        """
        Apply the full component operator to an input energy spectral density.
        This is useful for modelling the effect of a component on, for example,
        ASE and pump light.

        Parameters
        ----------
        spectrum : numpy array, dtype float64.
            shape(2, n_points). Energy spectrum of the light.
        omega_axis : numpy array, dtype float64/
            Angular frequency axis corresponding to spectrum

        Returns
        -------
        numpy array
            Spectrum after propagating through the component.
        """
        spectrum *= self.duty_cycle
        return spectrum


class fibre_pulse_picker():
    """
    Base class for fibre pulse pickers (e.g., EOMs and AOMs).

    See also base_components.pulse_picker.
    """

    def __init__(self, g, input_fibre, output_fibre, loss,
                 transmission_bandwidth, lambda_c, epsilon, theta,
                 beamsplitting, crosstalk, time_open, rate_reduction_factor,
                 input_rep_rate, order=2, output_coupler=False,
                 coupler_type="polarization", beta_list=None, gdm=0,
                 verbose=False):
        """
        Parameters
        ----------
        g : pyLaserPulse.grid.grid object
        input_fibre : optical fibre object
            Dfined using, for example:
            pyLaserPulse.base_components.step_index_passive_fibre
            pyLaserPulse.base_components.photonic_crystal_passive_fibre
            pyLaserPulse.base_components.step_index_active_fibre
            pyLaserPulse.base_components.photonic_crystal_active_fibre
            etc.
        output_fibre : optical fibre object
            Dfined using, for example:
            pyLaserPulse.base_components.step_index_passive_fibre
            pyLaserPulse.base_components.photonic_crystal_passive_fibre
            pyLaserPulse.base_components.step_index_active_fibre
            pyLaserPulse.base_components.photonic_crystal_active_fibre
            etc.
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
        crosstalk : float
            Polarization degradation caused by the component.
        time_open : float
            Time in s that the pulse picker is 'open', i.e., lets light through
            See Notes below.
        rate_reduction_factor : int
            Ratio of the input and output repetition rates of the pulse picker.
            See Notes below.
        input_rep_rate : float
            Repetition rate of the seed laser before the pulse picker.
            See Notes below.
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
        verbose : bool
            Print information to terminal if True
        """
        self.input_fibre = input_fibre
        self.output_fibre = output_fibre
        self.pulse_picker = pulse_picker(
            loss, transmission_bandwidth, lambda_c, epsilon, theta,
            beamsplitting, g, crosstalk, time_open, rate_reduction_factor,
            input_rep_rate, order=order, output_coupler=output_coupler,
            coupler_type=coupler_type, beta_list=beta_list, gdm=gdm)

        if verbose:
            self.make_verbose()

    def make_verbose(self):
        """
        Change self.verbose to True
        Called by optical_assemblies. If the optical assembly verbosity is
        True, all component verbosities are also set to True.
        """
        self.input_fibre.make_verbose()
        self.pulse_picker.make_verbose()
        self.output_fibre.make_verbose()

    def make_silent(self):
        """
        Change self.verbose to False
        Called by optical_assemblies. If the optical assembly verbosity is
        False, all component verbosities are also set to False.
        """
        self.input_fibre.make_silent()
        self.component.make_silent()
        self.output_fibre.make_silent()

    def propagate(self, pulse):
        """
        Apply the input_fibre, pulse picker, and output_fibre operators to an
        input field.

        Parameters
        ----------
        pulse : pyLaserPulse.pulse.pulse object

        Returns
        -------
        pyLaserPulse.pulse.pulse object
        """
        pulse = self.input_fibre.propagate(pulse)
        pulse = self.pulse_picker.propagate(pulse)
        pulse = self.output_fibre.propagate(pulse)
        return pulse

    def propagate_spectrum(self, spectrum, omega_axis):
        """
        Apply the full component operator to an input energy spectral density.
        This is useful for modelling the effect of a component on, for example,
        ASE and pump light.

        Parameters
        ----------
        spectrum: numpy array, dtype float64.
            shape(2, n_points). Energy spectrum of the light.
        omega_axis : numpy array, dtype float64/
            Angular frequency axis corresponding to spectrum

        Returns
        -------
        numpy array
            Spectrum after the fibre pulse picker.
        """
        spectrum = self.pulse_picker.propagate_spectrum(spectrum, omega_axis)
        return spectrum


class grating_compressor(component):
    """
    Compressor class. Inherits from the components class.
    """

    def __init__(self, loss, transmission_bandwidth, coating_material_path,
                 lambda_c, epsilon, theta, beamsplitting, crosstalk,
                 grating_separation, input_angle, groove_density, g, order=2,
                 optimize=False, verbose=False, output_coupler=False,
                 coupler_type="polarization"):
        """
        Parameters
        ----------
        loss : float
            Insertion loss at the signal wavelength.
        transmission_bandwidth : float
            Transmission bandwidth. Can also be interpreted as reflection
            bandwidth for reflective optics.
        coating_material : string
            Absolute path to a coating reflectivity spectrum.
            See pyLaserPulse.data.paths.materials.reflectivities
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
        crosstalk : float
            Polarization degradation caused by the component.
        grating_separation : float
            Propagated distance between reflections from the gratings in m
        input_angle : float
            Grating incidence angle in radians.
        groove_density : float
            Grating lines per mm
        g : pyLaserPulse.grid.grid object
        order : int
            Steepness of the super-Gaussian transmission window edges.
        optimize : bool
            Uses the Nelder-Mead algorithm from scipy.optimize.minimize if True
            to find the grating_separation and input_angle that minimizes the
            difference between the peak intensity of the compressed pulse and
            the transform limit.
        verbose : bool
            Returns information about the Nelder-Mead optimization and the
            optimized compressor parameters if True.
        output_coupler : boolean
            If True, the component is an output coupler and some of the
            main beam (pulse.field) is tapped off.
        coupler_type : string
            "polarization" or "beamsplitter". If the former, Jones matrix
            for polarization-based output coupler is used. If the latter,
            Jones matrix for polarization-independent beam splitting is
            used for the output coupler.

        Notes
        -----
        Diffraction efficiency is calculated using the method outlined in
        R. Casini et al., "On the intensity distribution function of blazed
        reflective diffraction gratings", JOSA A 31(10), pp 2179-2184 (2014)

        The second- and third-order compressor dispersion is calculated using
        the method outlined in
        R. L. Fork et al. "Compression of optical pulses to six femtoseconds by
        using cubic phase compensation", Optics Letters 12(7), pp 483-485
        (1987)
        and the correction to the second-order dispersion given in
        F. Kienle, "Advanced high-power optical parametric oscillators
        synchronously pumped by ultrafast fibre-based sources", PhD Thesis,
        Optoelectronics Research Centre, University of Southampton (2012)
            See page 37.
        """
        super().__init__(loss, transmission_bandwidth, lambda_c, epsilon,
                         theta, beamsplitting, g, crosstalk, order,
                         output_coupler, coupler_type)
        self.grating_separation = np.abs(grating_separation)
        self.input_angle = input_angle
        self.groove_density = groove_density * 1000  # Convert to lines/m
        self.groove_spacing = 1 / self.groove_density

        # Blaze angle for grid central wavelength
        self.lambda_blaze = self.grid.lambda_c
        self.blaze_angle = np.arcsin(self.lambda_blaze
                                     / (2 * self.groove_spacing))
        self.get_diffraction_angle()
        self.get_diffraction_efficiency()
        self.coating_filename = coating_material_path
        self.reflectivity \
            = utils.interpolate_data_from_file(self.coating_filename,
                                               self.grid.lambda_window,
                                               1e-9, 1, 'quadratic', False)
        self.reflectivity = self.reflectivity[None, :].repeat(2, axis=0)
        self.reflectivity = utils.ifftshift(self.reflectivity, axes=-1)
        self.run_optimization = optimize
        self.verbose = verbose

        # Get Taylor coefficients for user-defined angle and separation
        # These will be updated later if self.run_optimization == True
        self.get_Taylors()

    def get_diffraction_angle(self):
        """
        Calculate the diffraction angle for the blaze wavelength.
        """
        self.diff_angle = np.arcsin((1 * self.lambda_blaze
                                     / self.groove_spacing)
                                    - np.sin(self.input_angle))

    def get_diffraction_efficiency(self):
        """
        Scalar model of diffraction efficiency.
        """
        # Diffraction angle as a function of wavelength & ensuring valid
        # argument for arcsin
        arg = (self.grid.lambda_window / self.groove_spacing) \
            - np.sin(self.input_angle)
        arg[np.abs(arg) >= 1] = 1
        diff_angles = np.arcsin(arg)

        rho = None
        if self.input_angle >= self.blaze_angle:
            rho = np.cos(self.input_angle) / np.cos(self.input_angle
                                                    - self.blaze_angle)
        else:
            rho = np.cos(self.blaze_angle)

        term1 = 0.5 * np.pi * rho
        term2 = np.cos(self.blaze_angle)
        term3 = np.sin(self.blaze_angle) / np.tan((self.input_angle
                                                   + diff_angles) / 2)
        self.diff_efficiency = np.sinc(term1 * (term2 - term3))**2
        self.diff_efficiency[np.isnan(self.diff_efficiency)] = 0
        self.diff_efficiency = self.diff_efficiency[None, :].repeat(2, axis=0)
        self.diff_efficiency = utils.ifftshift(self.diff_efficiency, axes=1)

    def apply_transmission_spectrum(self, field, photon_spec):
        """
        Applies the spectral transmission window of the grating compressor.

        Parameters
        ----------
        field : numpy array
            Pulse field in the time domain.
            See pyLaserPulse.pulse.pulse.field
        photon_spec : numpy array
            Number of photonics in each frequency bin of the pulse spectrum.

        Returns
        -------
        numpy array
            Pulse field
        """
        spectrum = utils.fft(field)

        # Four passes of grating in total.
        self.transmission_spectrum = \
            self.transmission_spectrum * self.diff_efficiency \
            * self.reflectivity
        self.transmission_spectrum = self.transmission_spectrum**4
        trans_window = self._include_partition_noise(photon_spec)
        spectrum *= np.sqrt(trans_window)
        field = utils.ifft(spectrum)
        self.make_transmission_spectrum()  # refresh
        return field

    def get_Taylors(self):
        """
        Calculate the second order dispersion of the compressor.
        """
        factor_1 = -8 * np.pi**2 * const.c / (self.grid.omega_window**3
                                              * self.groove_spacing**2)
        factor_2 = self.grating_separation / np.cos(self.diff_angle)
        factor_3 = 1
        factor_4 = 2 * np.pi * const.c
        factor_5 = self.grid.omega_window * self.groove_spacing
        factor_6 = np.sin(self.input_angle)
        self.beta_2 = factor_1 * factor_2 / (factor_3
                                             - ((factor_4 / factor_5)
                                                - factor_6)**2)

        factor_1 = -(3 / self.grid.omega_window)
        factor_2 = (1 + (2 * np.pi * const.c
                         / (self.grid.omega_window * self.groove_spacing))
                    * np.sin(self.input_angle)
                    - np.sin(self.input_angle)**2)
        factor_3 = 1 / (1 - (((2 * np.pi * const.c)
                              / (self.grid.omega_window * self.groove_spacing))
                             - np.sin(self.input_angle))**2)
        self.beta_3 = factor_1 * factor_2 * factor_3 * self.beta_2

        self.beta_4 = np.gradient(self.beta_3, self.grid.omega, edge_order=2)
        self.beta_5 = np.gradient(self.beta_4, self.grid.omega, edge_order=2)

        self.beta_2 = self.beta_2[self.grid.midpoint]
        self.beta_3 = self.beta_3[self.grid.midpoint]
        self.beta_4 = self.beta_4[self.grid.midpoint]
        self.beta_5 = self.beta_5[self.grid.midpoint]

        self.Taylors = np.array((0, 0, self.beta_2, self.beta_3, self.beta_4,
                                 self.beta_5))

    def make_phase(self):
        """
        Turn beta2 into a phase
        """
        self.phase_argument = np.zeros_like(self.grid.omega,
                                            dtype=np.complex128)
        for idx, val in enumerate(self.Taylors):
            self.phase_argument += -1j * val * self.grid.omega**idx \
                / np.math.factorial(idx)

        self.phase = np.exp(self.phase_argument)
        self.phase = self.phase[None, :].repeat(2, axis=0)
        self.phase = utils.fftshift(self.phase)

    def propagate(self, pulse):
        """
        Apply the full compressor to an input pulse.

        Parameters
        ----------
        pulse : pyLaserPulse.pulse.pulse object

        Returns
        -------
        pyLaserPulse.pulse.pulse object
        """
        self.make_transmission_spectrum()
        self.make_Jones_matrix()
        input_field = pulse.field.copy()
        abs_spec = np.abs(utils.fft(input_field, axis=-1))
        transform_limit = utils.ifftshift(
            utils.ifft(abs_spec, axis=-1), axes=-1)
        transform_limit = np.sum(np.abs(transform_limit)**2, axis=0)
        if self.run_optimization:
            if self.verbose:
                print("\nOptimizing\n\n")

            def func(params): return np.abs(
                (np.amax(transform_limit) - np.amax(np.sum(np.abs(
                    self.optimize(params, pulse))**2, axis=0)))) \
                / np.amax(transform_limit)
            compressor_parameters = [self.grating_separation,
                                     self.input_angle]

            options = {'maxiter': 100, 'adaptive': True, 'fatol': .001,
                       'xatol': 0.001}
            kword = {'method': 'Nelder-Mead', 'options': options}
            self.optimization_output = opt.basinhopping(func,
                                                        compressor_parameters,
                                                        niter=10, T=100,
                                                        stepsize=0.5e-3,
                                                        minimizer_kwargs=kword)
            self.grating_separation = np.abs(self.optimization_output.x[0])
            self.input_angle = self.optimization_output.x[1]

            pulse.field = self.apply_operators(pulse)
            if self.verbose:
                self.print_optimization_information()
        else:
            pulse.field = self.apply_operators(pulse)

        if self.verbose:
            self.print_compressor_and_pulse_information(
                input_field, pulse.field, transform_limit)
        if pulse.high_res_samples:
            pulse.high_res_field_samples.append(pulse.field)
            pulse.high_res_rep_rate_samples.append(pulse.repetition_rate)
            if len(pulse.high_res_B_integral_samples) > 0:
                pulse.high_res_B_integral_samples.append(
                    pulse.high_res_B_integral_samples[-1])
            else:
                pulse.high_res_B_integral_samples.append(0)
            pulse.high_res_field_sample_points.append(
                pulse.high_res_sample_interval)
        return pulse

    def apply_operators(self, pulse):
        """
        Called to prevent looped recursion, which gave a marginally faster
        Nelder-Mead optimization in profiling.

        Parameters
        ----------
        pulse : pyLaserPulse.pulse.pulse object

        Returns
        -------
        numpy array
            Pulse field in the time domain.
        """
        pulse_field = pulse.field.copy()
        pulse.get_photon_spectrum(self.grid, pulse_field)
        photon_spec = pulse.photon_spectrum.copy()
        self.get_Taylors()
        self.make_phase()
        pulse_field = self.apply_transmission_spectrum(
            pulse_field, photon_spec)
        pulse_field = self.apply_Jones_matrix(pulse_field)
        pulse_field = utils.ifft(utils.fft(pulse_field) * self.phase)
        return pulse_field

    def optimize(self, compressor_parameters, pulse):
        """
        Method used by the  Nelder-Mead optimizer to find the optimal grating
        separation and incidence angle for maximizing the pulse peak power.

        Parameters
        ----------
        compressor_parameters: list
            zeroth index is separation, first index is angle.

        Returns
        -------
        numpy array
            Pulse field in the time domain
        """
        self.grating_separation = np.abs(compressor_parameters[0])
        self.input_angle = compressor_parameters[1]
        self.get_diffraction_angle()
        self.get_diffraction_efficiency()
        pulse_field = self.apply_operators(pulse)
        return pulse_field

    def print_optimization_information(self):
        print("Convergence reached: ",
              self.optimization_output.lowest_optimization_result.success)
        print("Optimization info.: ", self.optimization_output.message)
        print("Number of optimization iterations: ",
              self.optimization_output.nit)

    def print_compressor_and_pulse_information(self, input_field, output_field,
                                               transform_limit):
        ratio_before = 100 * np.amax(np.sum(np.abs(input_field)**2, axis=0)) \
            / np.amax(transform_limit)
        abs_spec = np.abs(utils.fft(output_field, axis=-1))
        transform_limit = utils.ifftshift(utils.ifft(abs_spec, axis=-1),
                                          axes=-1)
        transform_limit = np.sum(np.abs(transform_limit)**2, axis=0)
        ratio_after = 100 * np.amax(np.sum(np.abs(output_field)**2, axis=0)) \
            / np.amax(transform_limit)

        print("\nPulse compression data\n----------------------")
        print("Grating separation: %.3f mm\nIncident angle: %.3f degrees."
              % (1e3 * self.grating_separation,
                 float(np.rad2deg(self.input_angle))))
        print("\nPulse peak power with respect to peak power of transform "
              "limit:\n\tBefore compressor: %.2f %%"
              "\n\tAfter compressor: %.2f %%"
              % (ratio_before, ratio_after))


class step_index_fibre_compressor(step_index_passive_fibre):
    """
    Single mode fibre compressor class. Inherits from the step index fibre
    class.

    Nonlinear opertors are not applied. Dispersion compensation only.
    """

    def __init__(self, g, length, loss_file, Raman_file, core_diam, NA,
                 beat_length, n2, fR, tol, Sellmeier_file, optimize=True,
                 verbose=False):
        """
        Parameters
        ----------
        g : pyLaserPulse.grid.grid object
        length : float
            Fibre length in m
        loss_file : string
            Absolute path to the fibre loss as a function of wavelength.
            See pyLaserPulse.data.paths.materials.loss_spectra
        Raman_file : string
            Absolute path to the Raman response as a function of time.
            See pyLaserPulse.data.paths.materials.Raman_profiles
        core_diam : float
            Core diameter in m
        NA : float
            Numerical aperture
        beat_length : float
            Polarization beat length in m
        n2 : float
            Nonlinear index in m^2 / W
        fR : float
            Fractional Raman contribution to the fibre nonlinear response
            (e.g., fR = 0.18 for silica)
        tol : float
            Maximum propagation error used to adjust the proapgation step size
        Sellmeier_file : string
            Absolute path to the Sellmeier coefficients.
            See pyLaserPulse.data.paths.materials.Sellmeier_coefficients.
        optimize : bool
            Uses the Nelder-Mead algorithm from scipy.optimize.minimize if True
            to find the fibre length that minimizes the difference between the
            peak intensity of the compressed pulse and the transform limit.
        verbose : bool
            Returns information about the Nelder-Mead optimization and the
            optimized compressor parameters if True.
        """
        super().__init__(g, length, loss_file, Raman_file, core_diam, NA,
                         beat_length, n2, fR, tol, Sellmeier_file)
        self.optimize = optimize
        self.verbose = verbose

    def propagate(self, pulse):
        """
        Apply the fibre compressor to an input field.

        Parameters
        ----------
        pulse : pyLaserPulse.pulse.pulse object

        Returns
        -------
        pyLaserPulse.pulse.pulse object
        """
        input_field = pulse.field.copy()
        pulse.get_transform_limit(input_field)
        transform_limit = pulse.transform_limit.copy()
        if self.optimize:
            if self.verbose:
                print("\nOptimizing the compressor\n-------------------------")

            def func(length): return np.abs((
                np.amax(transform_limit) - np.amax(np.sum(np.abs(
                    self.apply_operators(length, pulse))**2, axis=0)))) \
                / np.amax(transform_limit)
            compressor_parameters = [self.L]

            options = {'maxiter': 1000, 'adaptive': True, 'fatol': .001,
                       'xatol': 0.001}
            kword = {'method': 'Nelder-Mead', 'options': options}
            self.optimization_output = opt.basinhopping(func,
                                                        compressor_parameters,
                                                        niter=10, T=100,
                                                        stepsize=0.5e-3,
                                                        minimizer_kwargs=kword)
            self.L = np.abs(self.optimization_output.x[0])

            pulse.field = self.apply_operators(self.L, pulse)
            if self.verbose:
                self.print_optimization_information()
            pass

        else:
            pulse.field = self.apply_operators(self.L, pulse)

        if self.verbose:
            self.print_compressor_and_pulse_information(input_field,
                                                        pulse.field,
                                                        transform_limit)
        if pulse.high_res_samples:
            if len(pulse.high_res_B_integral_samples) > 0:
                pulse.high_res_B_integral_samples.append(
                    pulse.high_res_B_integral_samples[-1])
            else:
                pulse.high_res_B_integral_samples.append(0)
            pulse.high_res_field_samples.append(pulse.field)
            pulse.high_res_rep_rate_samples.append(pulse.repetition_rate)
            pulse.high_res_field_sample_points.append(
                pulse.high_res_sample_interval)
        return pulse

    def apply_operators(self, length, pulse):
        """
        Apply the linear operator over length and return the resulting field.

        Parameters
        ----------
        length : float
            Fibre length in m
        pulse : pyLaserPulse.pulse.pulse object

        Returns
        -------
        numpy array
            Pulse field in the time domain.
            See pyLaserPulse.pulse.pulse.field
        """
        field = pulse.field.copy()
        spec = utils.fft(field)
        spec *= np.exp(-1 * self.linear_operator * length)
        field = utils.ifft(spec)
        return field

    def print_optimization_information(self):
        print("Convergence reached: ",
              self.optimization_output.lowest_optimization_result.success)
        print("Optimization info.: ", self.optimization_output.message)
        print("Number of optimization iterations: ",
              self.optimization_output.nit)

    def print_compressor_and_pulse_information(
            self, input_field, output_field, transform_limit):
        ratio_before = 100 * np.amax(np.sum(np.abs(input_field)**2, axis=0)) \
            / np.amax(transform_limit)
        abs_spec = np.abs(utils.fft(output_field, axis=-1))
        transform_limit = utils.ifftshift(
            utils.ifft(abs_spec, axis=-1), axes=-1)
        transform_limit = np.sum(np.abs(transform_limit)**2, axis=0)
        ratio_after = 100 * np.amax(np.sum(np.abs(output_field)**2, axis=0)) \
            / np.amax(transform_limit)

        print("\nPulse compression data\n----------------------")
        print("Fibre length: %.3f m." % self.L)
        print("\nPulse peak power with respect to peak power of transform "
              "limit:\n\tBefore compressor: %.2f %%"
              "\n\tAfter compressor: %.2f %%"
              % (ratio_before, ratio_after))


class rotated_splice:
    """
    Class for simulating a splice between PM fibres where the stress rods are
    rotated through rotation_angle.
    """

    def __init__(self, rotation_angle):
        """
        Parameters
        ----------
        rotation_angle : float
            Angle of rotated splice. degrees.
        """
        self.rot_angle = np.deg2rad(rotation_angle)

        self.rot_matrix = np.array((
            (np.cos(self.rot_angle), -1 * np.sin(self.rot_angle)),
            (np.sin(self.rot_angle), np.cos(self.rot_angle))
        ))

    def propagate(self, pulse):
        """
        Apply a rotation matrix to pulse.field to simulate the effect that a
        rotated splice would have on the field.

        Parameters
        ----------
        pulse : pyLaserPulse.pulse.pulse object

        Returns
        -------
        pyLaserPulse.pulse.pulse object
        """
        field = pulse.field.copy()
        field = np.dot(self.rot_matrix, field)
        pulse.field = field
        return pulse
