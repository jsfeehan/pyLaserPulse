#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 09:16:08 2023

@author: james feehan

Module of classes for branded active optical fibre.
"""

from pyLaserPulse.data import paths
import pyLaserPulse.base_components as bc


class nLight_Yb1200_4_125(bc.step_index_active_fibre):
    """
    step_index_active_fibre with default parameters which provide fibre
    properties matching nLight (Liekki) Yb1200-4/125 specifications.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    beat_length : float
        Polarization beat length in m
    seed_repetition_rate : float
        Repetition rate of the seed laser pulses
    pump_points : int
        Number of points in the pump light grid.
    ASE_wl_lims : list
        Wavelength limits of the pump and ASE grid, [min_wl, max_wl] in m.
    boundary_conditions : dict
        Set the boundary conditions for resolving the evolution of the pump,
        signal, and ASE light in both directions through the fibre.
        The type of simulation -- i.e., single-pass or full boundary value
        solver -- is determined by this dictionary.
        See pyLaserPulse.base_components.step_index_active_fibre
    time_domain_gain : Boolean
        Time domain gain included if True.
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True
    """
    def __init__(self, grid, length, beat_length, seed_repetition_rate,
                 pump_points, ASE_wl_lims, boundary_conditions,
                 time_domain_gain=False, n2=2.19e-20, verbose=False):
        core_diam = 4e-6
        NA = 0.12
        fR = 0.18
        tol = 1e-5
        doping_concentration = 9.15e25  # nominal 1200 dB/m
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, doping_concentration,
            paths.fibres.cross_sections.Yb_Al_silica, seed_repetition_rate,
            pump_points, ASE_wl_lims,
            paths.materials.Sellmeier_coefficients.silica, boundary_conditions,
            lifetime=1.5e-3, cladding_pumping=False,
            time_domain_gain=time_domain_gain, verbose=verbose)


class ORC_HD406_YDF(bc.step_index_active_fibre):
    """
    step_index_active_fibre with default parameters which provide fibre
    properties approximating the ORC's HD406 Yb-doped fibre specifications. No
    concrete data on this fibre is available to me, so I have matched the core
    diameter and index step to Corning HI1060.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    beat_length : float
        Polarization beat length in m
    doping_concentration : float
        Ion number density in m^-3.
    seed_repetition_rate : float
        Repetition rate of the seed laser pulses
    pump_points : int
        Number of points in the pump light grid.
    ASE_wl_lims : list
        Wavelength limits of the pump and ASE grid, [min_wl, max_wl] in m.
    boundary_conditions : dict
        Set the boundary conditions for resolving the evolution of the pump,
        signal, and ASE light in both directions through the fibre.
        The type of simulation -- i.e., single-pass or full boundary value
        solver -- is determined by this dictionary.
        See pyLaserPulse.base_components.step_index_active_fibre
    time_domain_gain : Boolean
        Time domain gain included if True.
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True
    """
    def __init__(self, grid, length, beat_length, doping_concentration,
                 seed_repetition_rate, pump_points, ASE_wl_lims,
                 boundary_conditions, time_domain_gain=False, n2=2.19e-20,
                 verbose=False):
        core_diam = 5.3e-6
        NA = 0.14
        fR = 0.18
        tol = 1e-5
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, doping_concentration,
            paths.fibres.cross_sections.Yb_Al_silica, seed_repetition_rate,
            pump_points, ASE_wl_lims,
            paths.materials.Sellmeier_coefficients.silica, boundary_conditions,
            lifetime=1.5e-3, time_domain_gain=time_domain_gain,
            verbose=verbose)


class CorActive_SCF_YB550_4_125_19(bc.step_index_active_fibre):
    """
    step_index_active_fibre with default parameters which provide fibre
    properties matching CorActive SCF-YB550-4/125-19.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    beat_length : float
        Polarization beat length in m
    seed_repetition_rate : float
        Repetition rate of the seed laser pulses
    pump_points : int
        Number of points in the pump light grid.
    ASE_wl_lims : list
        Wavelength limits of the pump and ASE grid, [min_wl, max_wl] in m.
    boundary_conditions : dict
        Set the boundary conditions for resolving the evolution of the pump,
        signal, and ASE light in both directions through the fibre.
        The type of simulation -- i.e., single-pass or full boundary value
        solver -- is determined by this dictionary.
        See pyLaserPulse.base_components.step_index_active_fibre
    time_domain_gain : Boolean
        Time domain gain included if True.
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True
    """
    def __init__(self, grid, length, beat_length, seed_repetition_rate,
                 pump_points, ASE_wl_lims, boundary_conditions,
                 time_domain_gain=False, n2=2.19e-20, verbose=False):
        core_diam = 4e-6
        NA = 0.19
        fR = 0.18
        tol = 1e-5
        doping_concentration = 1.45e26
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, doping_concentration,
            paths.fibres.cross_sections.Yb_Al_silica, seed_repetition_rate,
            pump_points, ASE_wl_lims,
            paths.materials.Sellmeier_coefficients.silica, boundary_conditions,
            lifetime=1.5e-3, time_domain_gain=time_domain_gain,
            verbose=verbose)


class OFS_R37003(bc.step_index_active_fibre):
    """
    step_index_active_fibre with default parameters which provide fibre
    properties matching OFS R37003 EDF.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    beat_length : float
        Polarization beat length in m
    seed_repetition_rate : float
        Repetition rate of the seed laser pulses
    pump_points : int
        Number of points in the pump light grid.
    ASE_wl_lims : list
        Wavelength limits of the pump and ASE grid, [min_wl, max_wl] in m.
    boundary_conditions : dict
        Set the boundary conditions for resolving the evolution of the pump,
        signal, and ASE light in both directions through the fibre.
        The type of simulation -- i.e., single-pass or full boundary value
        solver -- is determined by this dictionary.
        See pyLaserPulse.base_components.step_index_active_fibre
    time_domain_gain : Boolean
        Time domain gain included if True.
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.33e-20 m^2/W,
        which is the value for fused silica around 1550 nm.
    verbose : bool
        Print information to terminal if True
    """
    def __init__(self, grid, length, beat_length, seed_repetition_rate,
                 pump_points, ASE_wl_lims, boundary_conditions,
                 time_domain_gain=False, n2=2.33e-20, verbose=False):
        core_diam = 2.9e-6
        NA = 0.2963
        fR = 0.18
        tol = 1e-5
        doping_concentration = 4.75e24
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, doping_concentration,
            paths.fibres.cross_sections.Er_silica, seed_repetition_rate,
            pump_points, ASE_wl_lims,
            paths.materials.Sellmeier_coefficients.silica, boundary_conditions,
            lifetime=1.5e-3, time_domain_gain=time_domain_gain,
            verbose=verbose)


class OFS_EDF07_PM(bc.step_index_active_fibre):
    """
    step_index_active_fibre with default parameters which provide fibre
    properties matching OFS EDF07 PM.

    Low doping concentration Er-doped fibre (spec. ~7 dB/m @ 1530 nm).

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    seed_repetition_rate : float
        Repetition rate of the seed laser pulses
    pump_points : int
        Number of points in the pump light grid.
    ASE_wl_lims : list
        Wavelength limits of the pump and ASE grid, [min_wl, max_wl] in m.
    boundary_conditions : dict
        Set the boundary conditions for resolving the evolution of the pump,
        signal, and ASE light in both directions through the fibre.
        The type of simulation -- i.e., single-pass or full boundary value
        solver -- is determined by this dictionary.
        See pyLaserPulse.base_components.step_index_active_fibre
    time_domain_gain : Boolean
        Time domain gain included if True.
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    Dispersion values are NOT calculated from the fibre structure. Instead,
    Taylor coefficients derived from measured dispersion data are used with the
    override_dispersion_using_Taylor_coefficients method.
    """
    def __init__(self, grid, length, seed_repetition_rate, pump_points,
                 ASE_wl_lims, boundary_conditions, time_domain_gain=False,
                 n2=2.33e-20, verbose=False):
        core_diam = 4.5e-6
        NA = 0.21
        fR = 0.18
        tol = 1e-5
        beat_length = 5.2e-3  # dn = 3e-4 given on spec. sheet.
        doping_concentration = 3.85e24  # m^-3
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, doping_concentration,
            paths.fibres.cross_sections.Er_silica,
            seed_repetition_rate, pump_points, ASE_wl_lims,
            paths.materials.Sellmeier_coefficients.silica, boundary_conditions,
            lifetime=1.5e-3, time_domain_gain=time_domain_gain,
            verbose=verbose)

        # Estimated Taylor coefficients -- numpy polyfit to dispersion curve.
        # Approximate only.
        Taylors = [
            3.57136311e-026, 6.58171065e-041, -1.80618683e-055,
            2.44585290e-070, -1.27178335e-085, 1.02371874e-100,
            -8.15133802e-116, -3.11227103e-129, -1.50932274e-144,
            3.33607789e-158]
        self.override_dispersion_using_Taylor_coefficients(Taylors)


class OFS_EDF08_PM(bc.step_index_active_fibre):
    """
    step_index_active_fibre with default parameters which provide fibre
    properties matching an OFS fibre with part number EDF08-PM that has been
    reported in a few articles that I have read, but that I have not been able
    to find in the OFS catalogue (James Feehan <pylaserpulse@hotmail.com>).

    Low doping concentration Er-doped fibre
    (borrow the EDF07-PM spec. ~7 dB/m @ 1530 nm).

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    seed_repetition_rate : float
        Repetition rate of the seed laser pulses
    pump_points : int
        Number of points in the pump light grid.
    ASE_wl_lims : list
        Wavelength limits of the pump and ASE grid, [min_wl, max_wl] in m.
    boundary_conditions : dict
        Set the boundary conditions for resolving the evolution of the pump,
        signal, and ASE light in both directions through the fibre.
        The type of simulation -- i.e., single-pass or full boundary value
        solver -- is determined by this dictionary.
        See pyLaserPulse.base_components.step_index_active_fibre
    time_domain_gain : Boolean
        Time domain gain included if True.
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    Dispersion values are NOT calculated from the fibre structure. Instead,
    Taylor coefficients derived from measured dispersion data are used with the
    override_dispersion_using_Taylor_coefficients method.
    """
    def __init__(self, grid, length, seed_repetition_rate, pump_points,
                 ASE_wl_lims, boundary_conditions, time_domain_gain=False,
                 n2=2.33e-20, verbose=False):
        core_diam = 4.5e-6
        NA = 0.21
        fR = 0.18
        tol = 1e-5
        beat_length = 5.2e-3  # dn = 3e-4 given on spec. sheet.
        doping_concentration = 3.85e24  # m^-3
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, doping_concentration,
            paths.fibres.cross_sections.Er_silica,
            seed_repetition_rate, pump_points, ASE_wl_lims,
            paths.materials.Sellmeier_coefficients.silica, boundary_conditions,
            lifetime=1.5e-3, time_domain_gain=time_domain_gain,
            verbose=verbose)

        # Estimated Taylor coefficients -- numpy polyfit to dispersion curve.
        # Approximate only.
        Taylors = [
            2.04472032e-026, 1.04793944e-040, -2.25364515e-055,
            4.43083159e-070, -6.99350240e-084, -3.64616938e-098,
            1.10330252e-110, 3.92135618e-125, -5.82910283e-138,
            -1.21900693e-152]
        self.override_dispersion_using_Taylor_coefficients(Taylors)


class nLight_Er80_4_125_HD_PM(bc.step_index_active_fibre):
    """
    step_index_active_fibre with default parameters which provide fibre
    properties matching nLight Er80-4/125-HD-PM.

    High doping concentration Er-doped fibre (spec. 80+/-20 dB/m @ 1530 nm).

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    seed_repetition_rate : float
        Repetition rate of the seed laser pulses
    pump_points : int
        Number of points in the pump light grid.
    ASE_wl_lims : list
        Wavelength limits of the pump and ASE grid, [min_wl, max_wl] in m.
    boundary_conditions : dict
        Set the boundary conditions for resolving the evolution of the pump,
        signal, and ASE light in both directions through the fibre.
        The type of simulation -- i.e., single-pass or full boundary value
        solver -- is determined by this dictionary.
        See pyLaserPulse.base_components.step_index_active_fibre
    time_domain_gain : Boolean
        Time domain gain included if True.
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    Core diameter and NA were selected to match the specified dispersion value
    and MFD at 1550 nm. These are -22 ps/(nm km) and 6.5 +/- 1 um, respectively.
    The values given by the choice of core diameter and NA below are
    -19.4 ps/(nm km) and 5.6 um, respectively.
    """
    def __init__(self, grid, length, seed_repetition_rate, pump_points,
                 ASE_wl_lims, boundary_conditions, time_domain_gain=False,
                 n2=2.33e-20, verbose=False):
        core_diam = 3.45e-6
        NA = 0.22
        fR = 0.18
        tol = 1e-5
        beat_length = 1.5e-2  # dn = 3e-4 given on spec. sheet.
        doping_concentration = 6e25  # m^-3
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, doping_concentration,
            paths.fibres.cross_sections.Er_silica,
            seed_repetition_rate, pump_points, ASE_wl_lims,
            paths.materials.Sellmeier_coefficients.silica, boundary_conditions,
            lifetime=1.5e-3, time_domain_gain=time_domain_gain,
            verbose=verbose)


class Thorlabs_Liekki_M5_980_125(bc.step_index_active_fibre):
    """
    step_index_active_fibre with default parameters which provide fibre
    properties matching those of Thorlabs/Liekki M-5(980/125).

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    beat_length : float
        Polarization beat length in m
    seed_repetition_rate : float
        Repetition rate of the seed laser pulses
    pump_points : int
        Number of points in the pump light grid.
    ASE_wl_lims : list
        Wavelength limits of the pump and ASE grid, [min_wl, max_wl] in m.
    boundary_conditions : dict
        Set the boundary conditions for resolving the evolution of the pump,
        signal, and ASE light in both directions through the fibre.
        The type of simulation -- i.e., single-pass or full boundary value
        solver -- is determined by this dictionary.
        See pyLaserPulse.base_components.step_index_active_fibre
    time_domain_gain : Boolean
        Time domain gain included if True.
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.33e-20 m^2/W,
        which is the value for fused silica around 1550 nm.
    verbose : bool
        Print information to terminal if True
    """
    def __init__(self, grid, length, beat_length, seed_repetition_rate,
                 pump_points, ASE_wl_lims, boundary_conditions,
                 time_domain_gain=False, n2=2.33e-20, verbose=False):
        core_diam = 6e-6
        NA = 0.2247
        fR = 0.18
        tol = 1e-5
        doping_concentration = 3.5e24
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, doping_concentration,
            paths.fibres.cross_sections.Er_silica, seed_repetition_rate,
            pump_points, ASE_wl_lims,
            paths.materials.Sellmeier_coefficients.silica, boundary_conditions,
            lifetime=1.5e-3, time_domain_gain=time_domain_gain,
            verbose=verbose)


class NKT_DC_200_40_PZ_YB(bc.photonic_crystal_active_fibre):
    """
    photonic_crystal_active_fibre with default parameters which provide fibre
    properties approximating those of NKT DC-200/40-PZ-YB.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    beat_length : float
        Polarization beat length in m
    seed_repetition_rate : float
        Repetition rate of the seed laser pulses
    pump_ponts : int
        Number of points in the pump light grid
    ASE_wl_lims : list
        Wavelength limits of the pump and ASE grid, [min_wl, max_wl] in m.
    boundary_conditions : dict
        Set the boundary conditions for resolving the evolution of the pump,
        signal, and ASE light in both directions through the fibre.
        The type of simulation -- i.e., single-pass or full boundary value
        solver -- is determined by this dictionary.
        See pyLaserPulse.base_components.step_index_active_fibre
    time_domain_gain : Boolean
        Time domain gain included if True.
    cladding_pumping : bool
        Pump light is propagated in the cladding if true.
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    This fibre model is approximate only, and is not expected to produce highly
    accurate results.

    This fibre can be used for cladding or core pumping.
    """
    def __init__(self, grid, length, seed_repetition_rate, pump_points,
                 ASE_wl_lims, boundary_conditions, time_domain_gain=False,
                 cladding_pumping=True, n2=2.19e-20, verbose=False):
        fR = 0.18
        tol = 1e-5

        # Measured from images of fibre facet published by NKT
        hole_pitch = 10e-6
        hole_diam = 2.44e-6
        hole_diam_over_pitch = hole_diam / hole_pitch
        core_diam = 0.5 * 40e-6  # x.5 provides best match to published MFD

        # Doping concentration set by modelling small-signal absorption and
        # matching value to spec. sheet for cladding absorption at 976 nm.
        doping_concentration = 6.5e25  # gives nominal 12 dB/m

        beat_length = 1e-2  # dn >= 1e-4 given on spec. sheet.

        if cladding_pumping:
            # Air cladding, but specified pump NA of 0.55 - 0.65. Using ref.
            # index of 1.325 for the pump cladding gives the specified NA.
            self.cladding_pumping = {
                'pump_core_diam': 200e-6, 'pump_delta_n': 1.45 - 1.325,
                'pump_cladding_n': 1.325}
        else:
            self.cladding_pumping = {}

        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, hole_pitch,
            hole_diam_over_pitch, beat_length, n2, fR, tol,
            doping_concentration, paths.fibres.cross_sections.Yb_Al_silica,
            seed_repetition_rate, pump_points, ASE_wl_lims,
            paths.materials.Sellmeier_coefficients.silica, boundary_conditions,
            core_diam=core_diam, lifetime=1e-3,
            cladding_pumping=self.cladding_pumping,
            time_domain_gain=time_domain_gain, verbose=verbose)


class Nufern_EDFC_980_HP(bc.step_index_active_fibre):
    """
    step_index_active_fibre with default parameters which provide fibre
    properties matching those of Nufern EDFC-980-HP.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    beat_length : float
        Polarization beat length in m
    seed_repetition_rate : float
        Repetition rate of the seed laser pulses
    pump_points : int
        Number of points in the pump light grid.
    ASE_wl_lims : list
        Wavelength limits of the pump and ASE grid, [min_wl, max_wl] in m.
    boundary_conditions : dict
        Set the boundary conditions for resolving the evolution of the pump,
        signal, and ASE light in both directions through the fibre.
        The type of simulation -- i.e., single-pass or full boundary value
        solver -- is determined by this dictionary.
        See pyLaserPulse.base_components.step_index_active_fibre
    time_domain_gain : Boolean
        Time domain gain included if True.
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.33e-20 m^2/W,
        which is the value for fused silica around 1550 nm.
    verbose : bool
        Print information to terminal if True
    """
    def __init__(self, grid, length, beat_length, seed_repetition_rate,
                 pump_points, ASE_wl_lims, boundary_conditions,
                 time_domain_gain=False, n2=2.33e-20, verbose=False):
        core_diam = 3.2e-6
        NA = 0.22968
        fR = 0.18
        tol = 1e-5
        doping_concentration = 4.6e24
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA,
            beat_length, n2, fR, tol, doping_concentration,
            paths.fibres.cross_sections.Er_silica, seed_repetition_rate,
            pump_points, ASE_wl_lims,
            paths.materials.Sellmeier_coefficients.silica, boundary_conditions,
            lifetime=1.5e-3, time_domain_gain=time_domain_gain,
            verbose=verbose)


class Nufern_PLMA_YDF_10_125_M(bc.step_index_active_fibre):
    """
    step_index_active_fibre with default parameters which provide fibre
    properties matching those of Nufern PLMA GDF 10/125 M

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    seed_repetition_rate : float
        Repetition rate of the seed laser pulses
    pump_points : int
        Number of points in the pump light grid.
    ASE_wl_lims : list
        Wavelength limits of the pump and ASE grid, [min_wl, max_wl] in m.
    boundary_conditions : dict
        Set the boundary conditions for resolving the evolution of the pump,
        signal, and ASE light in both directions through the fibre.
        The type of simulation -- i.e., single-pass or full boundary value
        solver -- is determined by this dictionary.
        See pyLaserPulse.base_components.step_index_active_fibre
    time_domain_gain : Boolean
        Time domain gain included if True.
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True

   Notes
    -----
    This fibre can be used for caldding or core pumping.
    """
    def __init__(self, grid, length, seed_repetition_rate, pump_points,
                 ASE_wl_lims, boundary_conditions, time_domain_gain=False,
                 cladding_pumping=False, n2=2.19e-20, verbose=False):
        core_diam = 11e-6
        NA = 0.075
        fR = 0.18
        tol = 1e-5
        doping_concentration = 5.2e25  # 4.95 dB/m at 976 nm cladding pumped
        beat_length = 3.4e-3  # dn = 2.4e-3 given on spec. sheet
        if cladding_pumping:
            self.cladding_pumping = {
                'pump_core_diam': 125e-6, 'pump_delta_n': 1.4467 - 1.375,
                'pump_cladding_n': 1.375}
        else:
            self.cladding_pumping = {}
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, doping_concentration,
            paths.fibres.cross_sections.Yb_Al_silica, seed_repetition_rate,
            pump_points, ASE_wl_lims,
            paths.materials.Sellmeier_coefficients.silica, boundary_conditions,
            lifetime=1.5e-3, time_domain_gain=time_domain_gain,
            cladding_pumping=self.cladding_pumping, verbose=verbose)


class Nufern_PM_YDF_5_130_VIII(bc.step_index_active_fibre):
    """
    step_index_active_fibre with default parameters which provide fibre
    properties matching those of Nufern PM-YDF-5/130-VIII.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    seed_repetition_rate : float
        Repetition rate of the seed laser pulses
    pump_points : int
        Number of points in the pump light grid.
    ASE_wl_lims : list
        Wavelength limits of the pump and ASE grid, [min_wl, max_wl] in m.
    boundary_conditions : dict
        Set the boundary conditions for resolving the evolution of the pump,
        signal, and ASE light in both directions through the fibre.
        The type of simulation -- i.e., single-pass or full boundary value
        solver -- is determined by this dictionary.
        See pyLaserPulse.base_components.step_index_active_fibre
    time_domain_gain : Boolean
        Time domain gain included if True.
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    cladding_pumping : bool
        Pump light is propagated in the cladding if true.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    This fibre can be used for caldding or core pumping.
    """
    def __init__(self, grid, length, seed_repetition_rate, pump_points,
                 ASE_wl_lims, boundary_conditions, time_domain_gain=False,
                 cladding_pumping=False, n2=2.19e-20, verbose=False):
        core_diam = 5e-6
        NA = 0.12
        fR = 0.18
        tol = 1e-5
        doping_concentration = 9.75e25
        beat_length = 4.12e-3  # dn = 2.4e-3 given on spec. sheet
        if cladding_pumping:
            self.cladding_pumping = {
                'pump_core_diam': 130e-6, 'pump_delta_n': 1.45 - 1.375,
                'pump_cladding_n': 1.375}
        else:
            self.cladding_pumping = {}
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, doping_concentration,
            paths.fibres.cross_sections.Yb_Al_silica, seed_repetition_rate,
            pump_points, ASE_wl_lims,
            paths.materials.Sellmeier_coefficients.silica, boundary_conditions,
            lifetime=1.5e-3, time_domain_gain=time_domain_gain,
            cladding_pumping=self.cladding_pumping, verbose=verbose)


class Nufern_PLMA_YDF_25_250(bc.step_index_active_fibre):
    """
    step_index_active_fibre with default parameters which provide fibre
    properties matching those of Nufern PLMA-YDF-25/250-VIII.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    seed_repetition_rate : float
        Repetition rate of the seed laser pulses
    pump_points : int
        Number of points in the pump light grid.
    ASE_wl_lims : list
        Wavelength limits of the pump and ASE grid, [min_wl, max_wl] in m.
    boundary_conditions : dict
        Set the boundary conditions for resolving the evolution of the pump,
        signal, and ASE light in both directions through the fibre.
        The type of simulation -- i.e., single-pass or full boundary value
        solver -- is determined by this dictionary.
        See pyLaserPulse.base_components.step_index_active_fibre
    time_domain_gain : Boolean
        Time domain gain included if True.
    cladding_pumping : bool
        Pump light is propagated in the cladding if true.
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    This fibre can be used for cladding or core pumping.
    """
    def __init__(self, grid, length, seed_repetition_rate,
                 pump_points, ASE_wl_lims, boundary_conditions,
                 time_domain_gain=False, cladding_pumping=False,
                 n2=2.19e-20, verbose=False):
        core_diam = 25e-6
        NA = 0.065
        fR = 0.18
        tol = 1e-5
        beat_length = 4.12e-3  # dn = 2.4e-3 given on spec. sheet
        if cladding_pumping:
            self.cladding_pumping = {
                'pump_core_diam': 250e-6, 'pump_delta_n': 1.45 - 1.375,
                'pump_cladding_n': 1.375}
        else:
            self.cladding_pumping = {}
        doping_concentration = 2.975e25  # Gives nominal 5.25 dB/m at 976 nm
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, doping_concentration,
            paths.fibres.cross_sections.Yb_Al_silica, seed_repetition_rate,
            pump_points, ASE_wl_lims,
            paths.materials.Sellmeier_coefficients.silica, boundary_conditions,
            lifetime=1.5e-3, time_domain_gain=time_domain_gain,
            cladding_pumping=self.cladding_pumping, verbose=verbose)


class Nufern_PLMA_30_400(bc.step_index_active_fibre):
    """
    step_index_active_fibre with default parameters which provide fibre
    properties matching those of Nufern PLMA-YDF-30/400-VIII.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    seed_repetition_rate : float
        Repetition rate of the seed laser pulses
    pump_points : int
        Number of points in the pump light grid.
    ASE_wl_lims : list
        Wavelength limits of the pump and ASE grid, [min_wl, max_wl] in m.
    boundary_conditions : dict
        Set the boundary conditions for resolving the evolution of the pump,
        signal, and ASE light in both directions through the fibre.
        The type of simulation -- i.e., single-pass or full boundary value
        solver -- is determined by this dictionary.
        See pyLaserPulse.base_components.step_index_active_fibre
    time_domain_gain : Boolean
        Time domain gain included if True.
    cladding_pumping : bool
        Pump light is propagated in the cladding if true.
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    This fibre can be used for cladding or core pumping.
    """
    def __init__(self, grid, length, seed_repetition_rate,
                 pump_points, ASE_wl_lims, boundary_conditions,
                 time_domain_gain=False, cladding_pumping=False,
                 n2=2.19e-20, verbose=False):
        core_diam = 30e-6
        NA = 0.06
        fR = 0.18
        tol = 1e-5
        beat_length = 3.467e-3  # dn = 3e-4 given on spec. sheet
        if cladding_pumping:
            self.cladding_pumping = {
                'pump_core_diam': 400e-6, 'pump_delta_n': 1.45 - 1.375,
                'pump_cladding_n': 1.375}
        else:
            self.cladding_pumping = {}
        doping_concentration = 2.1e25  # Gives nominal 2.7 dB/m at 976 nm
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, doping_concentration,
            paths.fibres.cross_sections.Yb_Al_silica, seed_repetition_rate,
            pump_points, ASE_wl_lims,
            paths.materials.Sellmeier_coefficients.silica, boundary_conditions,
            lifetime=1.5e-3, time_domain_gain=time_domain_gain,
            cladding_pumping=self.cladding_pumping, verbose=verbose)


class Nufern_FUD_4288_LMA_YDF_48_400E(bc.step_index_active_fibre):
    """
    step_index_active_fibre with default parameters which provide fibre
    properties matching those of Nufern FUD-4288 LMA-YDF-48/400E.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    seed_repetition_rate : float
        Repetition rate of the seed laser pulses
    pump_points : int
        Number of points in the pump light grid.
    ASE_wl_lims : list
        Wavelength limits of the pump and ASE grid, [min_wl, max_wl] in m.
    boundary_conditions : dict
        Set the boundary conditions for resolving the evolution of the pump,
        signal, and ASE light in both directions through the fibre.
        The type of simulation -- i.e., single-pass or full boundary value
        solver -- is determined by this dictionary.
        See pyLaserPulse.base_components.step_index_active_fibre
    time_domain_gain : Boolean
        Time domain gain included if True.
    cladding_pumping : bool
        Pump light is propagated in the cladding if true.
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    This fibre can be used for cladding or core pumping.
    """
    def __init__(self, grid, length, seed_repetition_rate,
                 pump_points, ASE_wl_lims, boundary_conditions,
                 time_domain_gain=False, cladding_pumping=False,
                 n2=2.19e-20, verbose=False):
        core_diam = 48e-6
        NA = 0.05
        fR = 0.18
        tol = 1e-5
        beat_length = 0.1
        if cladding_pumping:
            self.cladding_pumping = {
                'pump_core_diam': 400e-6, 'pump_delta_n': 1.45 - 1.375,
                'pump_cladding_n': 1.375}
        else:
            self.cladding_pumping = {}
        doping_concentration = 2.125e25  # Gives nominal 2.7 dB/m at 976 nm
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, doping_concentration,
            paths.fibres.cross_sections.Yb_Al_silica, seed_repetition_rate,
            pump_points, ASE_wl_lims,
            paths.materials.Sellmeier_coefficients.silica, boundary_conditions,
            lifetime=1.5e-3, time_domain_gain=time_domain_gain,
            cladding_pumping=self.cladding_pumping, verbose=verbose)


class Nufern_PM_YSF_HI_HP(bc.step_index_active_fibre):
    """
    step_index_active_fibre with default parameters which provide fibre
    properties matching those of Nufern PLMA YSF 10/125.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    seed_repetition_rate : float
        Repetition rate of the seed laser pulses
    pump_points : int
        Number of points in the pump light grid.
    ASE_wl_lims : list
        Wavelength limits of the pump and ASE grid, [min_wl, max_wl] in m.
    boundary_conditions : dict
        Set the boundary conditions for resolving the evolution of the pump,
        signal, and ASE light in both directions through the fibre.
        The type of simulation -- i.e., single-pass or full boundary value
        solver -- is determined by this dictionary.
        See pyLaserPulse.base_components.step_index_active_fibre
    time_domain_gain : Boolean
        Time domain gain included if True.
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True
    """
    def __init__(self, grid, length, seed_repetition_rate, pump_points,
                 ASE_wl_lims, boundary_conditions, time_domain_gain=False,
                 n2=2.19e-20, verbose=False):
        core_diam = 6e-6
        NA = 0.11
        fR = 0.18
        tol = 1e-5
        beat_length = 2.7e-3  # dn = 2.8e-4 given on spec. sheet.
        doping_concentration = 3.586e25  # m^-3
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, doping_concentration,
            paths.fibres.cross_sections.Yb_Nufern_PM_YSF_HI_HP,
            seed_repetition_rate, pump_points, ASE_wl_lims,
            paths.materials.Sellmeier_coefficients.silica, boundary_conditions,
            lifetime=0.945e-3, time_domain_gain=time_domain_gain,
            verbose=verbose)


class Thorlabs_Liekki_Yb1200_6_125_DC(bc.step_index_active_fibre):
    """
    step_index_active_fibre with default parameters which provide fibre
    properties matching those of Liekki Yb1200-6/125DC.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    seed_repetition_rate : float
        Repetition rate of the seed laser pulses
    pump_points : int
        Number of points in the pump light grid.
    ASE_wl_lims : list
        Wavelength limits of the pump and ASE grid, [min_wl, max_wl] in m.
    boundary_conditions : dict
        Set the boundary conditions for resolving the evolution of the pump,
        signal, and ASE light in both directions through the fibre.
        The type of simulation -- i.e., single-pass or full boundary value
        solver -- is determined by this dictionary.
        See pyLaserPulse.base_components.step_index_active_fibre
    time_domain_gain : Boolean
        Time domain gain included if True.
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    cladding_pumping : bool
        Pump light is propagated in the cladding if true.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    This fibre can be used for caldding or core pumping.

    The octagonal cladding geometry is not modelled, so cladding absorption is
    appriximate only.
    """
    def __init__(self, grid, length, seed_repetition_rate, pump_points,
                 ASE_wl_lims, boundary_conditions, time_domain_gain=False,
                 cladding_pumping=False, n2=2.19e-20, verbose=False):
        core_diam = 5e-6
        NA = 0.12
        fR = 0.18
        tol = 1e-5
        beat_length = 1e-2  # dn >= 1e-4 given on spec. sheet.
        if cladding_pumping:
            self.cladding_pumping = {
                'pump_core_diam': 125e-6, 'pump_delta_n': 1.45 - 1.375,
                'pump_cladding_n': 1.375}
        else:
            self.cladding_pumping = {}
        doping_concentration = 1.25e26  # 2.4 dB/m cladding absorption at 976 nm
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, doping_concentration,
            paths.fibres.cross_sections.Yb_Al_silica, seed_repetition_rate,
            pump_points, ASE_wl_lims,
            paths.materials.Sellmeier_coefficients.silica, boundary_conditions,
            lifetime=1.5e-3, time_domain_gain=time_domain_gain,
            cladding_pumping=self.cladding_pumping, verbose=verbose)