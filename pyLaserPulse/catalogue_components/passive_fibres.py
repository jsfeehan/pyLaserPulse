#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 09:14:08 2023

@author: james feehan

Module of classes for branded passive optical fibre.
"""

from pyLaserPulse.data import paths
import pyLaserPulse.base_components as bc


class Corning_HI1060(bc.step_index_passive_fibre):
    """
    step_index_passive_fibre with default parameters which provide fibre
    properties matching Corning HI1060 specifications.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    beat_length : float
        Polarization beat length in m
    tol : float
        Tolerance for propagation integration error
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True
    """
    def __init__(
            self, grid, length, beat_length, tol, n2=2.19e-20, verbose=False):
        core_diam = 5.3e-6
        NA = 0.14
        fR = 0.18
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)


class Corning_HI1060_FLEX(bc.step_index_passive_fibre):
    """
    step_index_passive_fibre with default parameters which provide fibre
    properties matching Corning HI1060 FLEX specifications.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    beat_length : float
        Polarization beat length in m
    tol : float
        Tolerance for propagation integration error
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    NA = 0.19096 used instead of specified 0.21. 0.19096 gives better match
    to other specified (and more important) parameters, such as the MFD.

    """
    def __init__(
            self, grid, length, beat_length, tol, n2=2.19e-20, verbose=False):
        core_diam = 3.65e-6
        NA = 0.19096
        fR = 0.18
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)


class SMF_28(bc.step_index_passive_fibre):
    """
    step_index_passive_fibre with default parameters which provide fibre
    properties matching SMF-28 specifications.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    beat_length : float
        Polarization beat length in m
    tol : float
        Tolerance for propagation integration error
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.33e-20 m^2/W,
        which is the value for fused silica around 1550 nm.
    verbose : bool
        Print information to terminal if True
    """
    def __init__(
            self, grid, length, beat_length, tol, n2=2.33e-20, verbose=False):
        core_diam = 8.6e-6
        NA = 0.120354
        fR = 0.18
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)


class SMF_28e(bc.step_index_passive_fibre):
    """
    step_index_passive_fibre with default parameters which provide fibre
    properties matching SMF-28e specifications.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    beat_length : float
        Polarization beat length in m
    tol : float
        Tolerance for propagation integration error
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.33e-20 m^2/W,
        which is the value for fused silica around 1550 nm.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    NA = 0.1209533 chosen over specified value of 0.14 because this gives a
    better match to other (more important) specified values, such as the MFD.
    """
    def __init__(
            self, grid, length, beat_length, tol, n2=2.33e-20, verbose=False):
        core_diam = 9e-6
        NA = 0.1209533
        fR = 0.18
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)


class OFS_980(bc.step_index_passive_fibre):
    """
    step_index_passive_fibre with default parameters which provide fibre
    properties matching OFS-980 specifications.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    beat_length : float
        Polarization beat length in m
    tol : float
        Tolerance for propagation integration error
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.33e-20 m^2/W,
        which is the value for fused silica around 1550 nm.
    verbose : bool
        Print information to terminal if True
    """
    def __init__(
            self, grid, length, beat_length, tol, n2=2.33e-20, verbose=False):
        core_diam = 4.3e-6
        NA = 0.1681453
        fR = 0.18
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)


class Nufern_SM2000D(bc.step_index_passive_fibre):
    """
    step_index_passive_fibre with default parameters which provide fibre
    properties matching Nufern SM2000D dispersion-shifted SMF.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    beat_length : float
        Polarization beat length in m
    tol : float
        Tolerance for propagation integration error
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.33e-20 m^2/W,
        which is the value for fused silica around 1550 nm.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    Dispersion and MFD match the specfication sheet at 2 um, but only the
    MFD matches the specifications at 1550 nm. The dispersion at 1550 nm is
    -95.5 ps/(nm km) for this model, but specified to be -50 ps/(nm km).
    """
    def __init__(
            self, grid, length, beat_length, tol, n2=2.33e-20, verbose=False):
        core_diam = 2.1e-6
        NA = 0.369495227
        fR = 0.18
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)


class Nufern_FUD4258_UHNA(bc.step_index_passive_fibre):
    """
    Step-index ultra-high NA passive fibre based on Nufern's FUD-4258
    specifications.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    beat_length : float
        Polarization beat length in m
    tol : float
        Tolerance for propagation integration error
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.33e-20 m^2/W,
        which is the value for fused silica around 1550 nm.
    verbose : bool
        Print information to terminal if True
    """
    def __init__(
            self, grid, length, beat_length, tol, n2=2.33e-20, verbose=False):
        core_diam = 2.4e-6
        NA = 0.26
        fR = 0.18
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)


class Nufern_PLMA_GDF_10_125(bc.step_index_passive_fibre):
    """
    step_index_passive_fibre with default parameters which provide fibre
    properties matching Nufern PLMA GDF 10/125. This fibre is matched to
    Nufern PLMA YSF 10/125 (Nufern_PLMA_YSF_10_125).

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    tol : float
        Tolerance for propagation integration error
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True
    """
    def __init__(self, grid, length, tol, n2=2.19e-20, verbose=False):
        core_diam = 11e-6
        NA = 0.08853
        fR = 0.18
        beat_length = 3.4e-3
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)


class Nufern_PLMA_GDF_10_125_M(bc.step_index_passive_fibre):
    """
    step_index_passive_fibre with default parameters which provide fibre
    properties matching Nufern PLMA-GDF-10/125-M. This fibre is matched to
    Nufern PLMA-YDF-10/125-M (Nufern_PLMA_YDF_10_125_M).

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    tol : float
        Tolerance for propagation integration error
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True
    """
    def __init__(self, grid, length, tol, n2=2.19e-20, verbose=False):
        core_diam = 11e-6
        NA = 0.075
        fR = 0.18
        beat_length = 3.4e-3
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)


class Nufern_PLMA_GDF_25_250(bc.step_index_passive_fibre):
    """
    step_index_passive_fibre with default parameters which provide fibre
    properties matching those of Nufern PLMA-GDF-25/250-VIII.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    tol : float
        Tolerance for propagation integration error
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True
    """
    def __init__(self, grid, length, tol, n2=2.19e-20, verbose=False):
        core_diam = 25e-6
        NA = 0.065
        fR = 0.18
        tol = 1e-5
        beat_length = 4.12e-3  # dn = 2.4e-3 given on spec. sheet
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)


class SMF_780HP(bc.step_index_passive_fibre):
    """
    step_index_passive_fibre with default parameters which provide fibre
    properties matching 780 HP as listed on the Thorlabs website.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    beat_length : float
        Polarization beat length in m
    tol : float
        Tolerance for propagation integration error
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True
    """
    def __init__(self, grid, length, beat_length, tol, n2=2.19e-20,
                 verbose=False):
        core_diam = 4.4e-6
        NA = 0.13
        fR = 0.18
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)


class PM980_XP(bc.step_index_passive_fibre):
    """
    step_index_passive_fibre with default parameters which provide fibre
    properties matching PM980_XP as listed on the Thorlabs website.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    tol : float
        Tolerance for propagation integration error
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True
    """
    def __init__(self, grid, length, tol, n2=2.19e-20, verbose=False):
        core_diam = 5.5e-6
        NA = 0.12
        fR = 0.18
        beat_length = 2.7e-3  # Minimum specified value.
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)


class PM1550_XP(bc.step_index_passive_fibre):
    """
    step_index_passive_fibre with default parameters which provide fibre
    properties matching PM1550_XP as listed on the Thorlabs website.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    tol : float
        Tolerance for propagation integration error
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.33e-20 m^2/W,
        which is the value for fused silica around 1550 nm.
    verbose : bool
        Print information to terminal if True
    """
    def __init__(self, grid, length, tol, n2=2.33e-20, verbose=False):
        core_diam = 8.5e-6
        NA = 0.125
        fR = 0.18
        beat_length = 2.7e-3  # Specified as <5 mm at 1550 nm
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)


class Nufern_PM_GDF_5_130(bc.step_index_passive_fibre):
    """
    step_index_passive_fibre with default parameters which provide fibre
    properties matching Nufern PM GDF 5/130.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    tol : float
        Tolerance for propagation integration error
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True
    """
    def __init__(self, grid, length, tol, n2=2.19e-20, verbose=False):
        core_diam = 5.e-6
        NA = 0.12
        fR = 0.18
        beat_length = 4.12e-3  # Minimum specified value.
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)


class OFS_80414p2(bc.step_index_passive_fibre):
    """
    step_index_passive_fibre with default parameters which provide fibre
    properties matching OFS 80414p2 highly-nonlinear PM fibre. D ~ 2ps/(nm km)
    at 1550 nm.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    tol : float
        Tolerance for propagation integration error
    n2 : float
        Nonlinear index in m^2 / W.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    Using a step-index model for this fibre is likely to be inappropriate. The
    core diameter, NA, and n2 were chosen for the best match to the specified
    values for the effective mode area, dispersion at 1550 nm, and the nonlinear
    coefficient, and most likely are not representative (or even sensible)
    values for this fibre type.

    The specified values are:
    effective area: 12.5 square microns
    D(1550 nm): 2 ps/(nm km)
    nonlinear coefficient: 10.7 1/(W km)

    The model values are:
    effective area: 12.73 square microns
    D(1550 nm): 2.37 ps/(nm km)
    nonlinear coefficient: 10.83 1/(W km)
    """
    def __init__(self, grid, length, tol, n2=3.4e-20, verbose=False):
        core_diam = 3.65e-6
        NA = 0.315
        fR = 0.18
        beat_length = 1e80  # 5e-3  # Guess, no birefringence information given.
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, core_diam, NA, beat_length,
            n2, fR, tol, paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)


class NKT_SC_5_1040(bc.photonic_crystal_passive_fibre):
    """
    photonic_crystal_passive_fibre with default parameters whcih provide
    fibre properties matching NKT SC-5.0-1040.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    beat_length : float
        Polarization beat length in m
    tol : float
        Tolerance for propagation integration error
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    Non-PM, ~0.01 -- 0.011 1 / (W m) nonlinear coefficient, ZDW ~1040 nm.
    """
    def __init__(
            self, grid, length, beat_length, tol, n2=2.19e-20, verbose=False):
        hole_pitch = 3.5e-6
        hole_diam_over_pitch = 0.585
        fR = 0.18
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, hole_pitch,
            hole_diam_over_pitch, beat_length, n2, fR, tol,
            paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)


class NKT_SC_5_1040_PM(bc.photonic_crystal_passive_fibre):
    """
    photonic_crystal_passive_fibre with default parameters which provide
    fibre properties matching NKT SC-5.0-1040-PM.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    tol : float
        Tolerance for propagation integration error
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    PM, ~0.01 -- 0.011 1 / (W m) nonlinear coefficient, ZDW ~1040 nm.
    """
    def __init__(self, grid, length, tol, n2=2.19e-20, verbose=False):
        hole_pitch = 3.5e-6
        hole_diam_over_pitch = 0.585
        fR = 0.18
        beat_length = 6e-3  # dn = 1.7e-4 is given on spec. sheet.
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, hole_pitch,
            hole_diam_over_pitch, beat_length, n2, fR, tol,
            paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)


class NKT_NL_1050_NEG_1(bc.photonic_crystal_passive_fibre):
    """
    photonic_crystal_passive_fibre with default parameters which provide
    fibre properties matching NKT NL-1050-NEG-1 (discontinued).

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    tol : float
        Tolerance for propagation integration error
    beat_length : float
        Polarization beat length in m
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    ~19 1 / (W km) nonlinear coefficient, all-normal dispersion for
    pumping around 1 um. This nonlinear coefficient does not match the
    value specified by NKT, but the dispersion is a good match to
    experimental measurements taken from a length of this fibre
    (see A. Heidt, J. S. Feehan, et al., "Limits of coherent supercontinuum
    generation in normal dispersion fibres", J. Opt. Soc. Am. B 34(4),
    pp 764-775 (2017).
    """
    def __init__(
            self, grid, length, tol, beat_length, n2=2.19e-20, verbose=False):
        hole_pitch = 1.55e-6
        hole_diam_over_pitch = 0.37
        fR = 0.18
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, hole_pitch,
            hole_diam_over_pitch, beat_length, n2, fR, tol,
            paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)


class NKT_femtowhite_800(bc.photonic_crystal_passive_fibre):
    """
    photonic_crystal_passive_fibre with default parameters which provide
    fibre properties matching NKT femtoWHITE 800.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    beat_length : float
        Polarization beat length in m
    tol : float
        Tolerance for propagation integration error
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.3e-20 m^2/W. Lower than
        common values for fused silica at 800 nm, but matches the specified
        nonlinear parameter.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    The hole pitch (1.339e-6 m) and ratio of the hole diameter with respect to
    the pitch (0.635) were found using the Nelder Mead algorithm to minimize the
    difference between the calculated dispersion profile and that shown on the
    femtoWHITE 800 specification sheet. These values give a good match between
    600 nm to 1000 nm and accurately reproduce the first ZDW wavelength of
    ~750 nm, but DO NOT reproduce the second ZDW wavelength of ~1260 nm that is
    specified on the data sheet but not shown in the dispersion plot given by
    NKT.
    """
    def __init__(
            self, grid, length, tol, beat_length, n2=2.3e-20, verbose=False):
        hole_pitch = 1.339e-6
        hole_diam_over_pitch = 0.635
        fR = 0.18
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, hole_pitch,
            hole_diam_over_pitch, beat_length, n2, fR, tol,
            paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)


class NKT_DC_200_40_PZ_SI(bc.photonic_crystal_passive_fibre):
    """
    photonic_crystal_active_fibre with default parameters which provide fibre
    properties approximating those of NKT DC-200/40-PZ-SI.

    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    tol : float
        Tolerance for propagation integration error
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    This fibre model is approximate only, and is not expected to produce highly
    accurate results.
    """
    def __init__(self, grid, length, tol, n2=2.19e-20, verbose=False):
        fR = 0.18

        # Measured from images of fibre facet published by NKT
        hole_pitch = 10e-6
        hole_diam = 2.44e-6
        hole_diam_over_pitch = hole_diam / hole_pitch
        core_diam = 0.5 * 40e-6  # x.5 provides best match to published MFD
        core_radius = core_diam / 2

        beat_length = 1e-2  # dn >= 1e-4 given on spec. sheet.

        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, hole_pitch,
            hole_diam_over_pitch, beat_length, n2, fR, tol,
            paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)

        # Reset core_diam and core_radius.
        # Approximation -- May not be entirely appropriate to use the PCF model
        # for this fibre type (influence of stress rods on propagation
        # parameters is unknown, and more than one air hole removed to form the
        # large core).
        self.core_diam = core_diam
        self.core_radius = core_radius
        self.get_GNLSE_and_birefringence_parameters()  # effective MFD, etc.


class exail_IXF_SUP_5_125_1050_PM(bc.photonic_crystal_passive_fibre):
    """
    photonic_crystal_passive_fibre with default parameters which provide
    fibre properties matching Exail (Photonic Bretagne) IXD-SUP-5-125-1050-PM
    Parameters
    ----------
    grid : pyLaserPulse.grid.grid object
    length : float
        Fibre length.
    tol : float
        Tolerance for propagation integration error
    n2 : float
        Nonlinear index in m^2 / W. Default value is 2.19e-20 m^2/W,
        which is the value for fused silica around 1060 nm.
    verbose : bool
        Print information to terminal if True

    Notes
    -----
    PM, ~0.01 -- 0.011 1 / (W m) nonlinear coefficient, ZDW ~1050 nm.
    """
    def __init__(self, grid, length, tol, n2=2.19e-20, verbose=False):
        # Combination of hole pitch and relative diameter gives gamma ~0.01 W/m,
        # matching the specification sheet.
        # Loss taken to be standard silica Rayleigh scattering.
        hole_pitch = 3.2e-6
        hole_diam_over_pitch = 0.5
        fR = 0.18
        beat_length = 4.62e-3  # dn = 2.3e-4 is given on spec. sheet.
        super().__init__(
            grid, length, paths.materials.loss_spectra.silica,
            paths.materials.Raman_profiles.silica, hole_pitch,
            hole_diam_over_pitch, beat_length, n2, fR, tol,
            paths.materials.Sellmeier_coefficients.silica,
            verbose=verbose)

        # Taylor coefficients: found by curve fitting to data digitized from the
        # specification sheet. beta_0 and beta_1 omitted.
        T = [
            -2.1917239348038675e-27, 6.74037456981993e-41,
            -5.723254114718194e-56, 5.1904873586045684e-71,
            6.290787739109935e-86, 1.0874592863788497e-100,
            -5.3137028061790745e-115, -2.473403424245199e-130,
            8.186846622556787e-145, 1.3418999788736455e-161
        ]
        self.override_dispersion_using_Taylor_coefficients(T)