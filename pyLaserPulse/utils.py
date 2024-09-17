#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 00:35:01 2022

@author: james feehan

General functions.
"""


import numpy as np
import math
from scipy.interpolate import interp1d
import scipy.constants as const


fft = np.fft.fft
ifft = np.fft.ifft
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift


def check_dict_keys(key_list, d, param_name):
    """
    Check that all items in key_list are keys in dict d.

    Parameters
    ----------
    key_list : list
        keys (string) that are expected in dict d.
    d : dict
    param_name : string
        Name assigned to dictionary d in caller.

    Raises
    ------
    KeyError
        Raised if an item in key_list is not in dict d.
    """
    if not isinstance(d, dict):
        raise TypeError("%s must be a dictionary." % param_name)
    for item in key_list:
        try:
            d[item]
        except KeyError as err:
            msg = "Dictionary %s must contain key '%s'"
            raise KeyError(msg % (param_name, item)) from err


def find_nearest(search_value, container):
    """
    Find the value in 'container' which is closest to 'search_value'

    Parameters
    ----------
    search_value
        Value to look for in container
    container
        Iterable of numeric values

    Returns
    -------
    int
        Index in container of value nearest to search_value
    int, float, etc.
        Value in container closest to search_value
    """
    try:
        container = np.asarray(container)
    except Exception:
        pass
    index = np.argmin(np.abs(container - search_value))
    value = container[index]
    return index, value


def get_width(axis, data, meas_point=0.5):
    """
    Return the width of a distribution in array data.

    Parameters
    ----------
    axis : int
        Axis of data over which the width measurement is taken
    data : numpy array
        Distribution whose width is being measured
    meas_point : float
        Fraction of maximum value of data at which the width is measured.
        E.g., meas_point = 0.5 gives FWHM.

    Returns
    -------
    float, float64
        Width of the distribution in data measured at meas_point.
    """
    # must make sure that the measurement is done with the data centred
    # on axis origin.
    points = len(axis)
    half_points = int(points / 2)
    maximum = np.amax(data)
    idx_maximum = np.argmax(data)
    data = np.roll(data, -idx_maximum + half_points)

    idx_1 = find_nearest(meas_point * maximum, data[0:half_points])[0]
    idx_2 = find_nearest(
        meas_point * maximum, data[half_points::])[0] + half_points
    FWHM = np.abs(axis[idx_2] - axis[idx_1])
    return FWHM


def get_FWe2M(axis, data):
    """
    Return the full width at 1/e^2 of the maximum pulse duration.

    Parameters
    ----------
    axis : int
        Axis of data over which the width measurement is taken
    data : numpy array
        Distribution whose width is being measured.

    Returns
    -------
    float, float64
        Full-width at 1/e^2 of the distribution in data.
    """
    return get_width(axis, data, meas_point=np.exp(-2))


def swap_halves(arr, axis=-1):
    """
    Swap the halves of an even-element array over a given axis. E.g., over the
    0th axis of a 1-D array:

    Parameters
    ----------
    arr : numpy array
    axis : int

    Returns
    -------
    numpy array
        arr, but the halves are swapped:
        arr[half1:half2] --> arr[half2:half1]
    """
    pts = arr.shape[axis]
    half_1 = arr.take(indices=range(0, int(pts / 2)), axis=axis)
    half_2 = arr.take(indices=range(int(pts / 2), pts), axis=axis)
    return np.append(half_2, half_1, axis=axis)


def interpolate_data_from_file(filename, axis, axis_scale, data_scale,
                               interp_kind='linear', fill_value='extrapolate',
                               input_log=False, return_log=False):
    """
    Interpolate single-column data from a text file onto a grid axis.

    Parameters
    ----------
    filename : string
        Absolute path to the data file
    axis : numpy array
        New axis to interpolate onto
    axis_scale : float
        Scaling for the new axis
    data_scale : float
        Scaling of the data
    interp_kind
        Specifies the kind of interpolation as a string or as an integer
        specifying the order of the spline interpolator to use. See
        documentation for scipy.interpolate.interp1d.
    fill_value
        if an ndarray (or float), this value will be used to fill in for
        requested points outside of the data range. Default NaN if not provided
        If a two-element tuple, the first value is used as a fill value for
        x_new < x[0] and the second element is usd for x_new > x[-1].
        If 'extrapolate', then points outside the data range will be
        extrpolated. This is the default.
    input_log : bool
        Needs to be True if the data in the data file has units of dB.
    return_log : bool
        If True, returns -1 * np.log(10**(-1 * data / 10))

    Returns
    numpy array
        data from filename inteprolated onto axis.
    """
    try:
        data = np.loadtxt(filename, delimiter=",", dtype=np.float64)
    except ValueError:
        data = np.loadtxt(filename, delimiter="\t", dtype=np.float64)
    if input_log:
        # Convert from log_10 to linear scale
        data[:, 1] = 10**(data[:, 1] / 10)
    data[:, 1] = data_scale * data[:, 1]

    # Make sure all data values appear in order
    sorted_indices = np.argsort(data[:, 0])
    data[:, 0] = axis_scale * data[sorted_indices, 0]
    data[:, 1] = data[sorted_indices, 1]

    # Interpolate,
    data_func = interp1d(data[:, 0], data[:, 1], kind=interp_kind,
                         fill_value=fill_value, bounds_error=False)
    data = data_func(axis)
    if return_log:
        data = -1 * np.log(10**(-1 * data / 10))
    return data


def load_Raman(filename, time_window, dt):
    """
    Load parallel and perpendicular Raman contributions from text files and
    interpolate onto the time-domain grid.

    Parameters
    ----------
    filename : string
        Absolute path to the Raman data file.
    time_window : numpy array
        pyLaserPulse.grid.grid.time_window
    dt : float
        Resolution of time_window (pyLaserPulse.grid.grid.dt)

    Returns
    numpy array
        Raman data from filename interpolated onto time_window.
    """
    data = np.loadtxt(filename, delimiter='\t', skiprows=4)
    scaling = np.loadtxt(filename, skiprows=1, max_rows=1)
    time = data[:, 0] * 1e-15
    Raman = data[:, 1:3]
    time_1 = np.linspace(0, np.amax(time), len(time))
    time_2 = np.linspace(0, np.amax(time), int(np.amax(time) / dt))
    interp_Raman = interp1d(time_1, Raman, axis=0, kind='cubic')(time_2)
    Raman = np.zeros((len(time_window), 2))
    Raman[0:len(interp_Raman), :] = interp_Raman
    Raman = fft(Raman / (np.sum(Raman, axis=0)), axis=0) * (2 * np.pi)**.5 / dt

    # Scale // and _|_ components
    Raman[:, 1] = (1 / scaling) * Raman[:, 1]
    return Raman


def load_cross_sections(filename, delimiter, axis, axis_scale,
                        interp_kind='linear'):
    """
    Load active dopant cross-section data.

    Parameters
    ----------
    filename : string
        Absolute path to the cross section data file
    delimiter : string
        Delimiter used in the cross section data file
    axis : numpy array
        New wavelength axis to interpolate the data onto.
        See pyLaserPulse.grid.grid.lambda_window
    axis_scale : float
        Scaling for the interpolation axis
    interp_kind
        See documentation for scipy.interpolate.interp1d

    Returns
    -------
    numpy array
        absorption cross section data interpolated onto axis
    numpy array
        emission cross section data interpolated onto axis
    list
        [min_wavelength, max_wavelength]

    Notes
    -----
    Expects three-column text file of data formatted as follows:
        Wavelength delimiter Emission delimiter Absorption
    """
    data = np.loadtxt(filename, delimiter=delimiter)
    sorted_indices = np.argsort(data[:, 0])
    data = data[sorted_indices, :]
    data[:, 0] *= axis_scale
    min_wl = data[:, 0].min()
    max_wl = data[:, 0].max()
    absorption_func = interp1d(data[:, 0], data[:, 2], kind=interp_kind,
                               fill_value='extrapolate')
    emission_func = interp1d(data[:, 0], data[:, 1], kind=interp_kind,
                             fill_value='extrapolate')
    absorption = absorption_func(axis)
    emission = emission_func(axis)

    # Data with a fine wavelength grid and high dynamic range can result in
    # bad values after interpolation. Fix these.
    absorption[absorption < 0] = 0
    emission[emission < 0] = 0
    return absorption, emission, [min_wl, max_wl]


def load_target_power_spectral_density(energy, repetition_rate, filename, axis,
                                       d_axis, axis_scale, background,
                                       PSD_scale, interp_kind='linear',
                                       fill_value=0, input_log=False):
    """
    Load a file containing some target power spectral density.
    use PSD_scale to convert from W/m to mW/nm, etc.

    Parameters
    ----------
    energy : float
        Energy in Joules
    repetition_rate : float
    filename : string
        Absolute path to the data file containing the power spectral density
    axis : numpy array
        Axis onto which the data in filename is interpolated.
        If the data is given as a function of wavelength in the data file, use
        pyLaserPulse.grid.grid.lambda_window
        If the data is given as a function of angular frequency in the data
        file, use pyLaserPulse.grid.grid.omega_window.
    d_axis : numpy array
        Resolution of axis
    axis_scale : float
        Scaling for axis
    background : float
        Background value to subtract
    PSD_scale : float
        Scaling for the power spectral density
    interp_kind : See scipy.interpolate.interp1d documentation.
    fill_value : See scipy.interpolate.interp1d documentation. Default 0
    input_log : bool
        Needs to be True if the data in filename is logarithmic.

    Returns
    -------
    numpy array
        Target power spectral density interpolated onto axis.
    """
    target = interpolate_data_from_file(filename, axis, axis_scale, 1,
                                        interp_kind, fill_value,
                                        input_log=input_log)
    target -= background
    target[target < 0] = 0
    target *= repetition_rate * energy / np.sum(target)
    target *= PSD_scale / d_axis
    return target


def get_ESD_and_PSD(lambda_window, spectrum, repetition_rate):
    """
    Calculate the energy spectral density and the power spectral density
    from real-valued spectrum. The PSD is normalized to mW/nm.

    Parameters
    ----------
    lambda_window : numpy array
        Wavelength grid in m. See pyLaserPulse.grid.grid.lambda_window
    spectrum : numpy array
        Spectral data
    repetition_rate : float
        Repetition rate of the pulse source.

    Returns
    -------
    numpy array
        Energy spectral density in J/m
    numpy array
        Power spectral density in W/nm
    """
    energy_spectral_density = \
        spectrum * 2 * np.pi * const.c / lambda_window**2
    power_spectral_density = \
        energy_spectral_density * repetition_rate * 1e-6
    return energy_spectral_density, power_spectral_density


def Sellmeier(lambda_window, f):
    """
    Calculate the refractive index as a function of wavelength for fused silica

    Parameters
    ----------
    lambda_window : numpy array
        Wavelength grid in m. See pyLaserPulse.grid.grid.lambda_window
    f : string
        Absolute path to file containing Sellmeier coefficients.

    Returns
    -------
    numpy array
        Refractive index as a function of wavelength.
    """
    lw = 1e6 * lambda_window
    coeffs = np.loadtxt(f, skiprows=1)
    n_sq = 1
    for B, C in iter(coeffs):
        n_sq += B * lw**2 / (lw**2 - C**2)
    return np.sqrt(n_sq)


def fft_convolve(arr1, arr2):
    """
    Use the convolution theorem to do faster convolutions.

    Parameters
    ----------
    arr1 : numpy array
    arr2 : numpy array

    Returns
    -------
    numpy array
        The convolution of arr1 and arr2

    Notes
    -----
    arr1 and arr2 must have the same shape
    """
    if arr1.shape != arr2.shape:
        raise ValueError("arr1 and arr2 must have the same shape.")
    conv = ifft(fft(arr1) * fft(arr2))
    return conv


def PCF_propagation_parameters_K_Saitoh(
        lambda_window, grid_midpoint, omega_window, a, b, c, d, hole_pitch,
        hole_diam_over_pitch, core_radius, Sellmeier_file):
    """
    Calculate V, mode_ref_index, D, beta_2 for hexagonal-lattice PCF.

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
    a : numpy array
        4x4 array. See reference in notes (K. Saitoh)
    b : numpy array
        4x4 array. See reference in notes (K. Saitoh)
    c : numpy array
        4x4 array. See reference in notes (K. Saitoh)
    d : numpy array
        4x4 array. See reference in notes (K. Saitoh)
    hole_pitch : float
        Separation of neighbouring air holes in the hexagonal-lattice PCF
        structure.
    hole_diam_over_pitch : float
        Ratio of the air hole diameter to the hole pitch.
    core_radius : float
        Appriximate core radius in m
    Sellmeier_file : string
        Absolute path to the Sellmeier coefficients.
        See pyLaserPulse.data.paths.materials.Sellmeier_coefficients.

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

    Notes
    -----
    K. Saitoh et al., "Empirical relations for simple design of
    photonic crystal fibres",Opt. Express 13(1), 267--274 (2005).
    """
    material_ref_index = Sellmeier(
        lambda_window, Sellmeier_file)
    n_central = material_ref_index[grid_midpoint]

    A = np.zeros((4), dtype=float)
    B = np.zeros((4), dtype=float)
    for i in range(np.shape(a)[1]):
        A[i] = \
            a[0, i] + a[1, i] * (hole_diam_over_pitch)**b[0, i] \
            + a[2, i] * (hole_diam_over_pitch)**b[1, i] \
            + a[3, i] * (hole_diam_over_pitch)**b[2, i]
        B[i] = \
            c[0, i] + c[1, i] * (hole_diam_over_pitch)**d[0, i] \
            + c[2, i] * (hole_diam_over_pitch)**d[1, i] \
            + c[3, i] * (hole_diam_over_pitch)**d[2, i]

    V = A[0] + A[1] / (
        1 + A[2] * np.exp(
            A[3] * lambda_window / hole_pitch))
    W = B[0] + B[1] / (
        1 + B[2] * np.exp(
            B[3] * lambda_window / hole_pitch))

    n_FSM = np.sqrt(n_central**2
                    - (lambda_window * V
                        / (2 * np.pi * core_radius))**2)
    ref_index = np.sqrt((lambda_window * W
                         / (2 * np.pi * core_radius))**2 + n_FSM**2)

    k = 2 * np.pi * material_ref_index / lambda_window
    v_group = np.gradient(omega_window, k, edge_order=2)
    beta = 1 / v_group
    beta2_MAT = np.gradient(beta, omega_window, edge_order=2)
    D_MAT = -2 * np.pi * const.c * beta2_MAT / lambda_window**2

    decimate = 1
    max_points = 512
    if grid_midpoint > max_points:  # i.e., grid size is > 1024
        # Required because very fine grids can result in noisy gradient
        # calculations
        decimate = int(grid_midpoint / max_points)

    tmp = np.gradient(ref_index[::decimate], lambda_window[::decimate],
                      edge_order=2)
    tmp = np.gradient(tmp, lambda_window[::decimate], edge_order=2)
    D_WG = -1 * (lambda_window[::decimate] / const.c) * tmp
    D = D_WG + D_MAT[::decimate]
    beta_2 = -1 * lambda_window[::decimate]**2 * D / (2 * np.pi * const.c)

    if decimate > 1:  # Interpolate D and beta_2 onto original grid
        # kind='linear' produces artefacts. No difference seen between
        # kind='quadratic' and kind='cubic'.
        f = interp1d(lambda_window[::decimate], D, kind='quadratic',
                     fill_value='extrapolate')
        D = f(lambda_window)
        f = interp1d(lambda_window[::decimate], beta_2, kind='quadratic',
                     fill_value='extrapolate')
        beta_2 = f(lambda_window)
    return V, ref_index, D, beta_2


def get_Taylor_coeffs_from_beta2(beta_2, grid):
    """
    Calculate the Taylor coefficients which describe the propagation constant
    calculated using, e.g., Gloge, Saitoh (depending on what is being
    simulated).

    Parameters
    ----------
    beta_2 : numpy array
        Dispersion curve in s^2 / m
    grid : pyLaserPulse.grid.grid object

    Returns
    -------
    beta : numpy array
        Complex part of the propagation constant.

    Notes
    -----
    Second-order gradient of beta with respect to omega will give the dispersion
    curve to within the accuracy of the Taylor expansion.

    This function could be used for retrieving the Taylor coefficients for,
    e.g., grating compressors, but analytic formulae should be used instead
    where available.
    """
    idx_max = grid.points - 1
    idx_min = 0
    lim = 2 * grid.lambda_c
    if grid.lambda_max > lim:
        # Get min and max indices for Taylor coefficient calculations.
        # Only required for very large frequency grid spans.
        # From testing, seems that 2x central wavelength is a good upper limit.
        # Recall indexing for grid.lambda_window is reversed.
        idx_min = find_nearest(lim, grid.lambda_window)[0]
        idx_max = find_nearest(-1 * grid.omega[idx_min], grid.omega)[0]

    # Truncate to 11th order. In testing, >11th order could't be found reliably.
    tc = np.polyfit(
        grid.omega[idx_min:idx_max], beta_2[idx_min:idx_max], 9)[::-1]
    Taylors = np.zeros((len(tc) + 2))
    Taylors[2::] = tc
    print(Taylors)

    beta = np.zeros_like(grid.omega, dtype=np.complex128)
    for i, tc in enumerate(Taylors):
        beta += 1j * tc * grid.omega**i / math.factorial(i)
    return Taylors, beta