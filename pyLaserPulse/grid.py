#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:50:35 2020

@author: james feehan

Grid class
"""


import os
import numpy as np
import scipy.constants as const

import pyLaserPulse.utils as utils
import pyLaserPulse.exceptions as exc


class grid:
    """
    Time-frequency grid class.
    """
    def __init__(self, lambda_c, lambda_lims, t_range, verbose=True):
        """
        Parameters
        ----------
        lambda_c : float
           Central wavelength in m.
        lambda_lims : tuple of floats
           Wavelength limits of the plotting window in m. Must contain the
           full expected span of the final simulated spectrum.
        t_range : float
           Span of the time window in s.
        verbose : bool
            Prints information about the grid if True

        Attributes
        ----------
        points : int
            Number of points in the time-frequency grid. Integer power of 2.
        midpoint : int
            int(points / 2) -- The central point of the time-frequency grid.
        sim_idx : list
            Indices of the grid which contain data relevant to the simulation.
            Indices not included in this list should be outside of the user-
            defined frequency span.
        sim_idx_shift : list
            Indices of the FFTshifted grid which contain data relevant to the
            simulation. Indicies not included in this list should be outside
            of the user-defined frequency span.
        sim_idx_mask : array
            Masking array which can be used to isolate the array indices
            relevant to the simulation.
        sim_idx_midpoint : int
            Midpoint of sim_idx
        crop_points : int
            Number of points relevant to the simulation in the frequency
            domain.
        lambda_c : float
            The central wavelength of the frequency grid in m.
        lambda_min : float
            The minimum wavelength of the frequency grid in m.
        lambda_max : float
            The maximum wavelength of the frequency grid in m.
        lambda_lims : tuple of floats
            Wavelength limits of the plotting window in m, containing the full
            expected span of the final simulated spectrum as defined by the
            user. See notes below.
        omega_c : float
            The central angular frequency of the angular frequency grid in
            rad Hz.
        f_range : float
            The range (or span) of the frequency grid in Hz
        f_range_crop : float
            The range (or span) of the cropped frequency grid in Hz.
        df : float
            The resolution of the frequency grid in Hz
        f_c : float
            The central frequency of the frequency grid in Hz.
        dOmega : float
            The resolution of the angular frequency grid in rad Hz
        t_range : float
            The range (or span) of the time grid in s. See notes below.
        dt : float
            The resolution of the time grid in s
        time_window : numpy array
            The time grid in s
        omega : numpy array
            The angular frequency grid in rad Hz centred at 0 rad Hz
        omega_crop : numpy array
            omega[sim_idx_mask]
        omega_window : numpy array
            The angular frequency grid in rad Hz
        omega_window_shift : numpy array
            The FFTshifted angular frequency grid in rad Hx
        omega_window_crop : numpy array
            omega_window[sim_idx_mask]
        omega_window_crop_shift : numpy array
            omega_window_shift[sim_idx_shift]
        f_window : numpy array
            The frequency grid in Hz
        f_window_crop : numpy array
            f_window[sim_idx_mask]
        lambda_window : numpy array
            The wavelength window in m
        lambda_window_crop : numpy array
            lambda_window[sim_idx_mask]
        d_wl : numpy array
            The resolution of the wavelength window. See notes; the wavelength
            window is not evenly spaced.
        d_wl_crop : numpy array
            d_wl[sim_idx_crop]
        energy_window : numpy array
            The energy window in J, given by Planck's constant * f_window
        energy_window_crop : numpy array
            energy_window[sim_idx_crop]
        energy_window_shift : numpy array
            The FFTshifted energy window in J.
        energy_window_shift_crop : numpy array
            energy_window_shift[sim_idx_shift]
        wl_lims : tuple of floats
            Plotting limits in m, calculated based on lambda_lims and the
            extent of lambda_window.
        gobbler : numpy array
            Apodization window.
        FFT_scale : float
            Scaling multiplier or divider for FFTs and IFFts, respectively.
            FFT_scale = sqrt(2 * pi) / dt
        verbose : bool

        Notes
        -----
        t_range is rewritten to a larger value if needed; the frequency span
        takes priority.

        The minimum value of lambda_lims can be equal or less than that
        specified by the user.
        """
        if lambda_c < min(lambda_lims) or lambda_c > max(lambda_lims):
            msg = "lambda_c must be within the range defined by lambda_lims"
            raise exc.GridDefinitionIncorrectError(msg)
        self.verbose = verbose

        self.lambda_c = lambda_c
        self.f_c = const.c / self.lambda_c
        self.omega_c = 2 * np.pi * self.f_c

        self.lambda_lims = lambda_lims
        self.lambda_max = max(lambda_lims)
        self.lambda_min = min(lambda_lims)
        self.t_range = t_range

        fmin = const.c / self.lambda_max
        fmax = const.c / self.lambda_min

        fmax_window = 1.1 * fmax  # make 10% larger for windowing

        f_lims = [fmin, fmax_window]
        omega_min = 2 * np.pi * fmin
        omega_max_window = 2 * np.pi * fmax_window

        f_lims = [f_lim - const.c / self.lambda_c for f_lim in f_lims]
        self.f_range = max(f_lims) - min(f_lims)
        self.dt = 1 / (2 * np.amax(f_lims))
        self.FFT_scale = ((2 * np.pi)**0.5) / self.dt

        self.points = int(2**(np.ceil(np.log2(t_range / self.dt))))
        self.midpoint = int(self.points / 2)
        self.t_range = self.dt * self.points

        axis = np.linspace(0, self.points - 1, self.points)
        self.time_window = (axis - self.points / 2) * self.dt

        self.dOmega = 2 * np.pi / self.t_range
        self.omega = (axis - self.points / 2) * self.dOmega
        self.omega_window = self.omega + self.omega_c
        self.omega_window_shift = np.fft.fftshift(self.omega_window)

        # Indices of grid containing simulation data for visuals, etc.
        self.sim_idx = np.squeeze(np.where(
                (self.omega_window > omega_min) &
                (self.omega_window < omega_max_window)))
        self.sim_idx_shift = np.squeeze(np.where(
                (self.omega_window_shift > omega_min) &
                (self.omega_window_shift < omega_max_window)))
        self.sim_idx_mask = np.zeros((self.points), dtype=bool)
        self.sim_idx_mask[self.sim_idx] = True

        # Define the cropped angular frequency windows
        self.omega_window_crop = self.omega_window[self.sim_idx]
        self.omega_window_crop_shift = \
            self.omega_window_shift[self.sim_idx]
        self.omega_crop = self.omega[self.sim_idx]
        self.sim_idx_midpoint = np.argmin(np.abs(self.omega_crop - 0))
        self.crop_points = len(self.sim_idx)

        # Define the full and cropped frequency windows
        self.df = self.dOmega / (2 * np.pi)
        self.f_window = self.omega_window / (2 * np.pi)
        self.f_window_crop = self.omega_window_crop / (2 * np.pi)
        self.f_range = self.f_window.max() - self.f_window.min()
        self.f_range_crop = self.f_window_crop.max() - self.f_window_crop.min()

        # Define the full and cropped wavelength windows
        self.lambda_window = const.c / self.f_window
        self.lambda_window_crop = const.c / self.f_window_crop
        self.d_wl = np.gradient(-1 * self.lambda_window)
        self.d_wl_crop = np.gradient(-1 * self.lambda_window_crop)

        # Define the full and cropped energy windows
        self.energy_window = const.h * self.f_window
        self.energy_window_shift = np.fft.fftshift(self.energy_window)
        self.energy_window_crop = const.h * self.f_window_crop
        self.energy_window_shift_crop = np.fft.fftshift(
                self.energy_window_crop)

        # Define a Planck-taper apodization window -- self.gobbler -- based
        # on the limits of the calculated frequency grid OR the user-defined
        # limits, depending on which is the most stringent. Equal tapering at
        # both edges of the window.
        eps = 0.01
        self.gobbler = np.zeros((self.points))
        _mask = np.zeros((self.points), dtype=bool)
        _mask[(axis >= 1) & (axis < eps*self.points)] = True
        _riser = 1 / (
                1 + np.exp((eps * self.points / axis[_mask])
                           - (eps * self.points / (
                               eps * self.points - axis[_mask])))
                )
        _faller = _riser[::-1]
        self.gobbler[np.roll(_mask, self.sim_idx.min())] = _riser
        self.gobbler[
            self.sim_idx.max() - len(_faller) - 1:
            self.sim_idx.max() - 1] = _faller
        self.gobbler[self.sim_idx.min() + len(_riser):
                     self.sim_idx.max() - len(_faller)] = 1
        self.gobbler = np.fft.fftshift(self.gobbler)

        # Sort wavelength limits from total grid OR cropped grid
        # Used for plots.
        self.wl_lims = \
            [self.lambda_window_crop.min() if
             self.lambda_window_crop.min() > self.lambda_window.min()
             else self.lambda_window.min(),
             self.lambda_window_crop.max() if
             self.lambda_window_crop.max() < self.lambda_window.max()
             else self.lambda_window.max()]

        if self.verbose:
            infostring = '\nGrid parameters:'
            infostring += '\n' + '-' * len(infostring)
            infostring += '\n\tGrid points: %d' % self.points
            infostring += ('\n\tTemporal resolution: %.3f fs'
                           % (1e15 * self.dt))
            infostring += '\n\tTime range: %.3f ps' % (1e12 * self.t_range)
            infostring += ('\n\tFrequency resolution: %.3f THz'
                           % (1e-12 * self.df))
            infostring += ('\n\tCentral frequency: %.3f THz'
                           % (1e-12 * self.omega_c / (2 * np.pi)))

            infostring += ('\n\tSimulation')
            infostring += ('\n\t\tFrequency limits: '
                           + '%.3f THz to %.3f THz'
                           % (1e-12 * self.f_window.min(),
                              1e-12 * self.f_window.max()))
            infostring += ('\n\t\tFrequency range: %.3f THz'
                           % (1e-12 * (self.f_window.max()
                                       - self.f_window.min())))
            infostring += ('\n\t\tWavelength limits: '
                           + '%.3f nm to %.3f nm'
                           % (1e9 * self.lambda_window.min(),
                              1e9 * self.lambda_window.max()))

            infostring += ('\n\tVisuals')
            infostring += ('\n\t\tFrequency limits: '
                           + '%.3f THz to %.3f THz'
                           % (1e-12 * omega_min / (2 * np.pi),
                              1e-12 * omega_max_window / (2 * np.pi)))
            infostring += ('\n\t\tWavelength limits: '
                           + '%.3f nm to %.3f nm'
                           % (2e9 * np.pi * const.c / omega_max_window,
                              2e9 * np.pi * const.c / omega_min))
            print(infostring)

    def save(self, directory):
        """
        Save the grid information to a file in directory.

        Parameters
        ----------
        directory : string
            directory to which the data will be saved.

        Notes
        -----
        Only information required to recreate the grid object is saved.

        Data is saved using the numpy.savez method. Data can be accessed using
        the numpy.load method, which will return a dictionary with the
        following keys:
        points : Number of data points in the time-frequency grid
        lambda_c : Central wavelength in m
        lambda_max : Maximum grid wavelength in m.
        """
        np.savez(directory + 'grid.npz',
                 lambda_c=self.lambda_c,
                 lambda_lims=self.lambda_lims,
                 t_range=self.t_range)


class grid_from_pyLaserPulse_simulation(grid):
    """
    Time-frequency grid class.
    """
    def __init__(self, data_directory):
        if not data_directory.endswith(os.sep):
            grid_data = np.load(data_directory + os.sep + 'grid.npz')
        else:
            grid_data = np.load(data_directory + 'grid.npz')
        super().__init__(
            grid_data['lambda_c'],
            grid_data['lambda_lims'],
            grid_data['t_range'])


class _legacy_grid:
    """
    Time-frequency grid class.

    THIS GRID IS NO LONGER IN USE FOR SIMULATIONS, AND SHOULD ONLY BE USED
    TO LOAD DATA FROM SIMULATIONS RUN USING A VERSION OF pyLaserPulse WHICH
    WAS COMMITTED BEFORE JUNE 2026.

    DATA CREATED BEFORE THIS DATE CANNOT BE USED IN pyLaserPulse SIMULATIONS
    AND PLOTTING MUST BE DONE MANUALLY ALSO.
    """

    def __init__(self, points, lambda_c, lambda_max):
        """
        Parameters
        ----------
        points : int
            Number of points in the time-frequency grid. Must be an integer
            power of 2.
        lambda_c : float
            Central wavelength of the frequency grid in m.
        lambda_max : float
            Maximum wavelength of the frequency grid in m.

        Attributes
        ----------
        points : int
            Number of points in the time-frequency grid. Integer power of 2.
        midpoint : int
            int(points / 2) -- The central point of the time-frequency grid.
        lambda_c : float
            The central wavelength of the frequency grid in m
        lambda_max : float
            The maximum wavelength of the frequency grid in m.
        omega_c : float
            The central angular frequency of the angular frequency grid in
            rad Hz.
        f_range : float
            The range (or span) of the frequency grid in Hz
        df : float
            The resolution of the frequency grid in Hz
        f_c : float
            The central frequency of the frequency grid in Hz.
        dOmega : float
            The resolution of the angular frequency grid in rad Hz
        t_range : float
            The range (or span) of the time grid in s
        dt : float
            The resolution of the time grid in s
        time_window : numpy array
            The time grid in s
        omega : numpy array
            The angular frequency grid in rad Hz centred at 0 rad Hz
        omega_window : numpy array
            The angular frequency grid in rad Hz
        omega_window_shift : numpy array
            The FFTshifted angular frequency grid in rad Hx
        f_window : numpy array
            The frequency grid in Hz
        lambda_window : numpy array
            The wavelength window in m
        d_wl : numpy array
            The resolution of the wavelength window. See notes; the wavelength
            window is not evenly spaced.
        energy_window : numpy array
            The energy window in J, given by Planck's constant * f_window
        energy_window_shift : numpy array
            The FFTshifted energy window in J.
        FFT_scale : float
            Scaling multiplier or divider for FFTs and IFFts, respectively.
            FFT_scale = sqrt(2 * pi) / dt

        Notes
        -----
        The grids are calculated as follows:

        f_range = 2 (c / lambda_c - c / lambda_max), where c = 299792457 m/s
        df = f_range / points
        t_range = 1 / df
        dt = t_range / points
        time_window = dt * linspace(-1*midpoint, midpoint-1, points)
        dOmega = 2 * pi * df
        omega_window = dOmega * linspace(-1*midpoint, midpoint-1, points)
        lambda_window = 2 * pi * c / omega_window, where c = 299792458 m/s
        """
        self.points = points
        self.midpoint = int(self.points / 2)
        axis = np.linspace(-1 * self.midpoint,
                           (self.points - 1) / 2, self.points)
        self.lambda_c = lambda_c
        self.lambda_max = lambda_max
        self.omega_c = 2 * np.pi * const.c / self.lambda_c
        self.f_c = self.omega_c / (2 * np.pi)

        self.f_range = 2 * (const.c / self.lambda_c -
                            const.c / self.lambda_max)
        self.df = self.f_range / self.points
        self.dOmega = 2 * np.pi * self.df
        self.t_range = 1 / self.df
        self.dt = self.t_range / self.points
        self.time_window = self.dt * axis
        self.omega = self.dOmega * axis
        self.omega_window = self.omega_c + self.omega
        self.omega_window_shift = utils.fftshift(self.omega_window)
        self.f_window = self.omega_window / (2 * np.pi)
        self.lambda_window = const.c / self.f_window
        self.lambda_min = self.lambda_window.min()
        self.d_wl = np.gradient(-1 * self.lambda_window)
        self.energy_window = const.h * self.f_window
        self.energy_window_shift = utils.fftshift(self.energy_window)

        self.FFT_scale = ((2 * np.pi)**0.5) / self.dt

    def save(self, directory):
        """
        Save the grid information to a file in directory.

        Parameters
        ----------
        directory : string
            directory to which the data will be saved.

        Notes
        -----
        Only information required to recreate the grid object is saved.

        Data is saved using the numpy.savez method. Data can be accessed using
        the numpy.load method, which will return a dictionary with the
        following keys:
        points : Number of data points in the time-frequency grid
        lambda_c : Central wavelength in m
        lambda_max : Maximum grid wavelength in m.
        """
        np.savez(directory + 'grid.npz',
                 points=self.points,
                 lambda_c=self.lambda_c,
                 lambda_max=self.lambda_max)


class _legacy_grid_from_pyLaserPulse_simulation(grid):
    """
    Time-frequency grid class.

    THIS GRID IS NO LONGER IN USE FOR SIMULATIONS, AND SHOULD ONLY BE USED
    TO LOAD DATA FROM SIMULATIONS RUN USING A VERSION OF pyLaserPulse WHICH
    WAS COMMITTED BEFORE JUNE 2026.

    DATA CREATED BEFORE THIS DATE CANNOT BE USED IN pyLaserPulse SIMULATIONS
    AND PLOTTING MUST BE DONE MANUALLY ALSO.
    """

    def __init__(self, data_directory):
        """
        Parameters
        ----------
        data_direcotry : str
            Absolute path of directory containing grid.npz data file produced
            by a previous pyLaserPulse simulation.

        Attributes
        ----------
        points : int
            Number of points in the time-frequency grid. Integer power of 2.
        midpoint : int
            int(points / 2) -- The central point of the time-frequency grid.
        lambda_c : float
            The central wavelength of the frequency grid in m
        lambda_max : float
            The maximum wavelength of the frequency grid in m.
        omega_c : float
            The central angular frequency of the angular frequency grid in
            rad Hz.
        f_range : float
            The range (or span) of the frequency grid in Hz
        df : float
            The resolution of the frequency grid in Hz
        dOmega : float
            The resolution of the angular frequency grid in rad Hz
        t_range : float
            The range (or span) of the time grid in s
        dt : float
            The resolution of the time grid in s
        time_window : numpy array
            The time grid in s
        omega : numpy array
            The angular frequency grid in rad Hz centred at 0 rad Hz
        omega_window : numpy array
            The angular frequency grid in rad Hz
        omega_window_shift : numpy array
            The FFTshifted angular frequency grid in rad Hx
        f_window : numpy array
            The frequency grid in Hz
        lambda_window : numpy array
            The wavelength window in m
        d_wl : numpy array
            The resolution of the wavelength window. See notes; the wavelength
            window is not evenly spaced.
        energy_window : numpy array
            The energy window in J, given by Planck's constant * f_window
        energy_window_shift : numpy array
            The FFTshifted energy window in J.
        FFT_scale : float
            Scaling multiplier or divider for FFTs and IFFts, respectively.
            FFT_scale = sqrt(2 * pi) / dt

        Notes
        -----
        The grids are calculated as follows:

        f_range = 2 (c / lambda_c - c / lambda_max), where c = 299792457 m/s
        df = f_range / points
        t_range = 1 / df
        dt = t_range / points
        time_window = dt * linspace(-1*midpoint, midpoint-1, points)
        dOmega = 2 * pi * df
        omega_window = dOmega * linspace(-1*midpoint, midpoint-1, points)
        lambda_window = 2 * pi * c / omega_window, where c = 299792458 m/s
        """
        if not data_directory.endswith(os.sep):
            grid_data = np.load(data_directory + os.sep + 'grid.npz')
        else:
            grid_data = np.load(data_directory + 'grid.npz')
        super().__init__(
            grid_data['points'],
            grid_data['lambda_c'],
            grid_data['lambda_max'])


if __name__ == "__main__":
    g = grid(1050e-9, (101e-9, 1760e-9), 15e-12)
    print(g.lambda_min, g.lambda_max)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(g.lambda_window, np.fft.ifftshift(g.gobbler))
    ax.set_xlim([g.lambda_window_crop.min(), g.lambda_window_crop.max()])
    plt.show()
