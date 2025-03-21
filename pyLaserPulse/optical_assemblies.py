#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 11:30:45 2022

@author: james feehan

Module containing classes of optical assemblies. These are to be inherited and
used as templates for amplifiers, lasers, etc.
"""

import os
import numpy as np
from scipy.interpolate import interp1d
from abc import ABC
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm, Normalize

import pyLaserPulse.utils as utils
import pyLaserPulse.base_components as bc
import pyLaserPulse.coupling as coupling


class assembly(ABC):
    """
    Class for use as a template for fibre assemblies.
    James Feehan, 19/3/2022. Revised ~5/2022 and 18/7/2022.
    """

    def __init__(
            self, grid, components, name, wrap=False, high_res_sampling=None,
            plot=False, data_directory=None, verbose=True):
        """
        Parameters
        ----------
        grid : pyLaserPulse grid object.
        components : list of component objects.
        name : str.
            String identifier for the assembly object.
        wrap : bool. Include loss between first and last components (required,
            e.g., for a laser cavity).
        high_res_sampling : Nonetype or int
            if sampling : int
                Number of samples to take along the propagation axis
            if not sampling : None (default)
        plot : bool.
            Create plot information.
        data_directory : Nonetype or string
            if saving data : string
                Directory to which data will be saved
            if not saving data : None
        verbose : bool
            Print information about the simulation at runtime
        """
        self.grid = grid
        # Adds coupling loss between components, effectively 'building' the
        # laser from the components list passed as an argument to the
        # __init__ method.
        self.make_full_components_list(components, wrap)

        # self.sampling = high_res_sampling
        self.sampling = False
        if high_res_sampling:
            self.sampling = True
        self.plot = plot
        self.num_samples = high_res_sampling
        self.name = name
        if self.name is None and self.plot:
            raise ValueError(
                "optical_assemblies.fibre_assembly.name cannot be None if "
                "optical_assemblies.fibre_assembly.plot evaluates to True.")
        if self.name is not None and self.plot:
            self.plot_dict = {}

        self.save_data = False
        self.data_directory = data_directory
        if self.data_directory is not None:
            self.directory = self.data_directory + '/' + self.name + '/'
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            self.save_data = True
        self.verbose = verbose

        # Override verbosity of all components if self.verbose
        if self.verbose:
            for c in self.components:
                try:
                    c.make_verbose()
                except AttributeError:  # e.g., coupling classes.
                    pass
        else:
            for c in self.components:
                try:
                    c.make_silent()
                except AttributeError:  # e.g., coupling classes.
                    pass

    def make_full_components_list(self, comps, wrap):
        """
        Cycle through components and introduce coupling losses between them
        depending on their type and properties.

        Python equivalent of building the oscillator from the components list
        (hence why this method is a member of the sm_fibre_laser class).

        Parameters
        ----------
        comps : list
            List of component objects
        wrap : bool. Include loss between first and last components (required,
            e.g., for a laser cavity).

        Notes
        -----
        Neglects bulk components because loss is defined using the transmission
        window member variable. Adds Fresnel losses to coupling between fibre
        and free space (and vice versa), and an additional loss for coupling
        from free space to fibre.
        """
        self.components = []
        self.components.append(comps[0])

        # Coupling losses between each component from first to last:
        for i in range(1, len(comps)):
            names_1 = [comps[i - 1].__class__.__name__,
                       comps[i - 1].__class__.__base__.__name__]
            names_2 = [comps[i].__class__.__name__,
                       comps[i].__class__.__base__.__name__]

            # Note ordering of if/elif/else is important for proper handling
            # of free-space components, fibre, and fibre components.

            # fibre to fibre (includes fibre components):
            if (any('fibre' in names for names in names_1)
                    and any('fibre' in names for names in names_2)):
                c = coupling.fibreToFibreCoupling(
                    self.grid, comps[i - 1], comps[i], names_1, names_2)
                self.components.append(c)
                self.components.append(comps[i])

            # Fibre/free space and free space/fibre are separated because
            # an additional coupling loss for free space/fibre that accounts
            # for MFD mistmatch will be available in a future version.

            # fibre/fibre component to free space:
            elif (any('fibre' in names for names in names_1)
                    and any('component' in names for names in names_2)):
                c = coupling.freeSpaceToFibreCoupling(self.grid, comps[i - 1])
                self.components.append(c)
                self.components.append(comps[i])

            # free space to fibre/fibre component:
            elif (any('component' in names for names in names_1)
                    and any('fibre' in names for names in names_2)):
                c = coupling.freeSpaceToFibreCoupling(self.grid, comps[i])
                self.components.append(c)
                self.components.append(comps[i])

            # Free space component to free space component handled by loss
            # member variables and transmission window.
            else:
                self.components.append(comps[i])

        if wrap:
            # Special case: Coupling loss between last and first component:
            names_1 = [comps[-1].__class__.__name__,
                       comps[-1].__class__.__base__.__name__]
            names_2 = [comps[0].__class__.__name__,
                       comps[0].__class__.__base__.__name__]

            # fibre/fibre component to fibre/fibre component:
            if (any('fibre' in names for names in names_1)
                    and any('fibre' in names for names in names_2)):
                c = coupling.fibreToFibreCoupling(
                    self.grid, comps[-1], comps[0], names_1, names_2)
                self.components.append(c)

            # Fibre/free space and free space/fibre are separated because
            # an additional coupling loss for free space/fibre that accounts
            # for MFD mistmatch will be available in a future version.

            # fibre/fibre component to free space:
            elif (any('fibre' in names for names in names_1)
                    and any('component' in names for names in names_2)):
                c = coupling.freeSpaceToFibreCoupling(self.grid, comps[-1])
                self.components.append(c)

            # free space to fibre/fibre component:
            elif (any('component' in names for names in names_1)
                    and any('fibre' in names for names in names_2)):
                c = coupling.freeSpaceToFibreCoupling(self.grid, comps[0])
                self.components.append(c)

    def update_pulse_class(self, pulse, field):
        """
        Called when returning simulate() method.

        Parameters
        ----------
        pulse : pyLaserPulse.pulse.pulse object
        field : numpy array.
            Field to use for calculations and plots. This will generally either
            be pulse.field, or pulse.output.
        """
        if isinstance(field, list):
            field = np.asarray(field[0])
        pulse.get_chirp(self.grid, field)
        pulse.get_ESD_and_PSD(self.grid, field)
        pulse.get_energy_and_average_power(self.grid, field)

        pulse.energy_spectral_density = np.asarray(
            pulse.energy_spectral_density)
        pulse.power_spectral_density = np.asarray(
            pulse.power_spectral_density)
        pulse.pulse_energy = np.asarray(pulse.pulse_energy)
        return pulse

    @classmethod
    def _simulate(cls, func):
        """
        Simulation decorator. Just prints some nice info.

        Parameters
        ----------
        func : simulation method of the derived class
        """

        def wrapper(self, pulse):
            if self.verbose:
                infostring = '\nSimulating    %s' % self.name
                infostring += '\n' + '-' * len(infostring)
                print(infostring)
            return func(self, pulse)
        return wrapper

    def plot_pulse(self, pulse):
        """
        Plot the pulse time domain distributions.

        Parameters
        ----------
        pulse : pyLaserPulse.pulse.pulse object
        """
        handles = []
        colours = ['mediumpurple', 'seagreen']
        alphas = [1, 0.55]
        axis_str = [r'$x$', r'$y$']
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_title('Time domain distributions')
        for i, col in enumerate(colours):
            h1, = ax.semilogy(
                self.grid.time_window * 1e12, 1e-3 * np.abs(pulse.field[i, :])**2,
                c=col, lw=2, label='P(T) ' + axis_str[i], alpha=alphas[i])
            handles.append(h1)
        ax.set_xlabel(r'$T = t - z/v_{g}$, ps')
        ax.set_ylabel(r'P(T), kW')
        ax.set_xlim([1e12 * self.grid.time_window.min(),
                     1e12 * self.grid.time_window.max()])
        fmt = []
        self.plot_dict[self.name + ': time domain distribution'] = (ax, fmt)

        linestyles = ['-', '-.']
        alphas = [1, 0.55]
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_title('Pulse chirps')
        for i, col in enumerate(colours):
            h1, = ax.plot(
                self.grid.time_window * 1e12, 1e9 * pulse.chirp[i, :], c=col, lw=2,
                label=r'$\lambda (T)$ ' + axis_str[i], ls=linestyles[i],
                alpha=alphas[i])
            handles.append(h1)
        ax.set_xlabel(r'$T = t - z/v_{g}$, ps')
        ax.set_ylabel(r'$\lambda (T)$, nm')
        ax.set_xlim([1e12 * self.grid.time_window.min(),
                     1e12 * self.grid.time_window.max()])
        fmt = []
        self.plot_dict[self.name + ': pulse chirps'] = (ax, fmt)

    def plot_B_integral(self, pulse):
        """
        Convert pulse.high_res_field_samples into the B-integral and plot.

        Parameters
        ----------
        pulse : pyLaserPulse.pulse.pulse object
        """
        B = np.asarray(pulse.high_res_B_integral_samples).T

        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_title('Net B integral up to the output of %s' % self.name)
        ax.semilogy(
            np.cumsum(pulse.high_res_field_sample_points), B,
            c='mediumpurple', lw=2)
        ax.set_xlabel('z, m', fontsize=13)
        ax.set_ylabel('B integral, rad.', fontsize=13)
        ax.set_xlim([0, np.sum(pulse.high_res_field_sample_points)])
        ax.set_ylim([B[np.nonzero(B)].min(), B.max()])
        fmt = ['axes.fill_between(ax.lines[0].get_data()[0], 0, '
               + 'ax.lines[0].get_data()[1], color="mediumpurple", '
               + 'alpha=0.33)']
        self.plot_dict[
            self.name + ': B integral'] = (ax, fmt)

    def plot_energy_and_average_power(self, pulse):
        """
        Calculate the pulse energy and average power at each propagation step
        using pulse.high_res_field_samples.

        Parameters
        ----------
        pulse : pyLaserPulse.pulse.pulse object
        """
        energy_samples = np.asarray(np.abs(pulse.high_res_field_samples)**2).T
        energy_samples = np.sum(energy_samples, axis=1)
        energy_samples = np.sum(energy_samples, axis=0) * self.grid.dt
        power_samples = energy_samples * pulse.high_res_rep_rate_samples

        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_title('Signal energy up to the output of %s' % self.name)
        ax.semilogy(np.cumsum(pulse.high_res_field_sample_points),
                    1e9 * energy_samples, c='slategray', lw=2)
        ax.set_xlabel('z, m', fontsize=13)
        ax.set_ylabel('Energy, nJ', fontsize=13)
        ax.set_xlim([0, np.sum(pulse.high_res_field_sample_points)])
        ax.set_ylim([1e9 * energy_samples.min(), 1e9 * energy_samples.max()])
        fmt = ['axes.fill_between(ax.lines[0].get_data()[0], 0, '
               + 'ax.lines[0].get_data()[1], color="slategray", '
               + 'alpha=0.33)']
        self.plot_dict[
            self.name + ': pulse energy'] = (ax, fmt)

        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_title('Signal power up to the output of %s' % self.name)
        ax.semilogy(np.cumsum(pulse.high_res_field_sample_points),
                    power_samples, c='cornflowerblue', lw=2)
        ax.set_xlabel('z, m', fontsize=13)
        ax.set_ylabel('Power, W', fontsize=13)
        ax.set_xlim([0, np.sum(pulse.high_res_field_sample_points)])
        ax.set_ylim([power_samples.min(), power_samples.max()])
        fmt = ['axes.fill_between(ax.lines[0].get_data()[0], 0, '
               + 'ax.lines[0].get_data()[1], color="cornflowerblue", '
               + 'alpha=0.33)']
        self.plot_dict[
            self.name + ': average power'] = (ax, fmt)

    def plot_pulse_samples(self, pulse):
        """
        plot pulse power spectral density and time-domain distribution as a
        function of propagation distance.

        Parameters
        ----------
        pulse : pyLaserPulse.pulse.pulse object
        """
        time_samples = np.sum(np.abs(pulse.high_res_field_samples)**2, axis=1)
        pulse.get_ESD_and_PSD_from_high_res_field_samples(self.grid)

        Y = np.cumsum(pulse.high_res_field_sample_points)
        if self.grid.points >= 512:
            d = int(self.grid.points / 512)

        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_title(r'$P(T)$ vs z up to the output of %s' % self.name)
        vmax = time_samples.max()
        vmin = np.min(np.max(time_samples, axis=0))
        p = ax.pcolormesh(
            1e12 * self.grid.time_window[::d], Y, time_samples[:, ::d],
            cmap='cubehelix_r', norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_xlabel(r'$T = t - z/v_{g}$, ps')
        ax.set_ylabel(r'Propagated distance, m')

        ax.p = p  # MUST be used for pcolormesh
        ax.colorbar_label = 'Power, W'
        fmt = []
        self.plot_dict[self.name + ': Pulse vs z'] = (ax, fmt)

        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_title('PSD vs ' + r'$z$' + ' up to the output of %s \n'
                     % self.name
                     + ' (neglects light outside of grid.time_window)')
        vmax = pulse.high_res_PSD_samples.max()
        vmin = np.min(np.max(pulse.high_res_PSD_samples, axis=0))
        p = ax.pcolormesh(
            1e9 * self.grid.lambda_window[::d], Y,
            pulse.high_res_PSD_samples[:, ::d], cmap='cubehelix_r',
            norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_xlabel(r'$\lambda$, nm')
        ax.set_ylabel(r'Propagated distance, m')

        ax.p = p  # MUST be used for pcolormesh
        ax.colorbar_label = 'PSD, dBm/nm'
        fmt = []
        self.plot_dict[self.name + ': PSD vs z'] = (ax, fmt)

    def plot_spectra(self, pulse):
        """
        Plot the pulse power spectral densities at the output of the assembly.

        Parameters
        ----------
        pulse : pyLaserPulse.pulse.pulse object
        """
        min_y = 1e-6
        handles = []
        integrated_PSD = np.sum(pulse.power_spectral_density, axis=0)

        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_title('PSDs')
        h1, = ax.semilogy(self.grid.lambda_window * 1e9, integrated_PSD, c='k',
                          ls='--', lw=2, label='Net PSD')
        h2, = ax.semilogy(
            self.grid.lambda_window * 1e9, pulse.power_spectral_density[0, :],
            c='mediumseagreen', ls='-', lw=2, label=r'x')
        h3, = ax.semilogy(
            self.grid.lambda_window * 1e9, pulse.power_spectral_density[1, :],
            c='darkorange', ls='-', lw=2, label=r'y')
        handles.extend([h1, h2, h3])
        ax.set_xlabel('Wavelength, nm', fontsize=13)
        ax.set_ylabel('Power spectral density, mW/nm', fontsize=13)
        max_y = integrated_PSD.max()
        ax.set_ylim([min_y, 4 * max_y])
        ax.set_xlim([1e9 * self.grid.lambda_window.min(),
                     1e9 * self.grid.lambda_window.max()])

        fmt = []
        self.plot_dict[self.name + ': gain fibre, net PSDs'] = (ax, fmt)

    @classmethod
    def saver(cls, method):
        """
        Decorator for saving data.

        Parameters
        ----------
        method : save method of any derived class

        Notes
        -----
        This method saves the data belonging to the pulse object.

        All derived classes should save dta using this decorated method.
        If no additional functionality is required by the method func in
        the derived class, use the following syntax:

            @assembly.saver
            def save(self):
                pass

        If additional functionality is required, use the following syntax:

            @assembly.saver
            def save(self):
                self.another_method()
        """

        def wrapper(self, pulse, grid):
            pulse.save(self.directory)
            grid.save(self.directory)
            return method(self)
        return wrapper


class passive_assembly(assembly):
    """
    Class for use as a passive SM fibre assembly template.
    """

    def __init__(
            self, grid, components, name, wrap=False, high_res_sampling=None,
            plot=False, data_directory=None, verbose=True):
        super().__init__(
            grid, components, name, wrap=wrap,
            high_res_sampling=high_res_sampling,
            plot=plot, data_directory=data_directory, verbose=verbose)

    @assembly._simulate
    def simulate(self, pulse):
        """
        Simulate an SM fibre amplifier.

        pulse: pulse class. Starting field used in simulations.

        Returns the pulse class.
        """
        pulse.get_ESD_and_PSD(self.grid, pulse.field)
        self.input_pulse_PSD = pulse.power_spectral_density

        # Only used if self.sampling, but also only needs to be set
        # once so keep it out of the loop.
        if self.sampling:
            pulse.num_samples = self.num_samples

        if pulse.save_high_res_samples and pulse.save_dir is None:
            raise Exception("pulse.save_dir cannot be NoneType if "
                            "pulse.save_high_res_samples == True")

        # Handle high-resolution field sampling
        if self.sampling:  # Turn on
            pulse.high_res_samples = True
            component_locations = [0]  # 0 for start of 1st component

        # Propagate through each component
        for val in self.components:
            if self.verbose:
                print('\n' + val.__class__.__name__)
            pulse = val.propagate(pulse)

            # If sampling is active, retrieve component locations and skip
            # coupling transmission objects inserted into self.components
            # by make_full_components_list.
            if self.sampling:
                loc = np.sum(pulse.high_res_field_sample_points)
                if loc != component_locations[-1]:
                    component_locations.append(loc)

            # Handle NaN solutions
            if np.any(np.isnan(pulse.field)):
                pulse.output.append(np.ones_like(pulse.field) * np.nan)
                return self.update_pulse_class(pulse, pulse.field)
            else:
                pulse.add_OPPM_noise(self.grid, True)
                pulse.get_photon_spectrum(self.grid, pulse.field)

        # # Save high-resolution field samples if sampling is active and if
        # # pulse.save_high_res_samples == True. This parameter must be set
        # # in the pulse object outside of the sm_fibre_amplifier class along
        # # with the directory that the data is saved under.
        # if self.sampling and pulse.save_high_res_samples:
        #     pulse.save_field(self.filename,
        #                      component_locations=component_locations)

        if self.sampling:  # Turn off
            pulse.high_res_samples = False

        self.update_pulse_class(pulse, pulse.field)

        if self.plot:
            self.plot_spectra(pulse)
            self.plot_pulse(pulse)
            if self.sampling:
                self.plot_B_integral(pulse)
                self.plot_pulse_samples(pulse)
                self.plot_energy_and_average_power(pulse)

        if self.save_data:
            self.save(pulse, self.grid)

        return pulse

    @assembly.saver
    def save(self):
        """
        Save all data to self.directory.
        """
        pass


class sm_fibre_laser:
    """
    Class for use as a single-mode mode-locked fibre laser template.
    """
    def __init__(self, grid, amplifiers, round_trips, name,
                 round_trip_output_samples=10, verbose=False):
        """
        grid : pyLaserPulse.grid.grid object
        amplifiers : list
            List of pyLaserPulse.optical_assemblies fibre amplifier objects.
        round_trips : int
            Number of round trips to simulate
        name : str
            String identifier for the assembly object
        round_trip_samples : int
            Number of round trips in which the output field is sampled.
            Sampling is done from roundtrip number
            round_trips-round_trip_samples to round_trips. If
            round_trip_samples > round_trips, round_trip_samples = round_trips.
        verbose : bool
            Print information about the simulation at runtime
        """
        self.name = name
        self.grid = grid
        self.round_trips = round_trips
        self.amplifiers = amplifiers
        self.verbose = verbose
        self.round_trip_output_samples = round_trip_output_samples

        # Set active_fibre.oscillator = True -- Replenish pump each round trip
        # Set verbosity of active fibre to verbosity of optical assembly.
        for amp in amplifiers:
            for c in amp.components:
                if (isinstance(c, bc.step_index_active_fibre)
                        or isinstance(c, bc.photonic_crystal_active_fibre)):
                    c.oscillator = True
                    c.verbose = self.verbose

    def simulate(self, pulse):
        """
        Simulate an SM fibre laser.

        Parameters
        ----------
        pulse : pyLaserPulse.pulse.pulse object

        Returns
        -------
        pyLaserPulse.pulse.pulse
        """
        for rt in range(self.round_trips):
            pulse.roundtrip_reset()
            for j, amp in enumerate(self.amplifiers):
                # No need for OPPM addition here (unlike other optical assembly
                # classes) because this is handled by the amplifier objects.
                if rt == 0 and j == 0:
                    # 0th round trip co_core_ASE from quantum noise only.
                    pulse = amp.simulate(pulse)
                else:
                    amp.add_co_core_ASE(
                            self.amplifiers[j-1].co_core_ASE_ESD_output)
                    pulse = amp.simulate(pulse)

                # Handle NaN solutions
                if np.any(np.isnan(pulse.field)):
                    pulse.output.append(np.ones_like(pulse.field) * np.nan)
                    return self.update_pulse_class(pulse, pulse.output)

            # Handle output field sampling
            if rt >= self.round_trips - self.round_trip_output_samples:
                pulse.output_samples.append(pulse.output)

        self.update_pulse_class(pulse, pulse.output)
        return pulse

    def update_pulse_class(self, pulse, field):
        """
        Called when returning simulate() method.

        Parameters
        ----------
        pulse : pyLaserPulse.pulse.pulse object
        field : numpy array.
            Field to use for calculations and plots. This will generally either
            be pulse.field, or pulse.output.
        """
        if isinstance(field, list):
            field = np.asarray(field[0])
        pulse.get_ESD_and_PSD(self.grid, field)
        pulse.get_energy_and_average_power(self.grid, field)

        pulse.energy_spectral_density = np.asarray(
            pulse.energy_spectral_density)
        pulse.power_spectral_density = np.asarray(
            pulse.power_spectral_density)
        pulse.pulse_energy = np.asarray(pulse.pulse_energy)
        return pulse

# class sm_fibre_laser(assembly):
#     """
#     Class for use as a SM fibre laser template.
#     James Feehan, 24/12/2020.
#     """

#     def __init__(self, grid, components, round_trips, name,
#                  round_trip_output_samples=10, high_res_sampling=False,
#                  high_res_sampling_limits=[0, 1],
#                  high_res_sample_interval=10e-2, plot=False,
#                  data_directory=None, verbose=True):
#         """
#         components_list: list of component objects. Must appear in order.
#             Recommended that components_list[0] is gain.
#             Coupling losses between components are added automatically.
#         round_trips: int. Number of cavity round trips to simulate.
#         round_trips: int. Number of round trips to simulate.
#         name : str.
#             String identifier for the assembly object.
#         round_trip_samples: int. Number of round trips in which the output
#             field is sampled. Sampling is done from roundtrip number
#             round_trips-round_trip_samples to round_trips. If
#             round_trip_samples > round_trips, round_trip_samples = round_trips.
#         high_res_sampling: bool. If True, high-resolution sampling of the
#             intracavity field. This is slow.
#         high_res_sampling_limits: list, int, len=2. Start and stop round trips
#             for the high-resolution field sampling. Ignored if
#             high_res_sampling==False.
#         high_res_sample_interval: float. Distance separating high-resolution
#             field samples. Default is 10 cm.
#         """
#         super().__init__(grid, components, name, wrap=True, plot=plot,
#                          data_directory=data_directory, verbose=verbose)
#         self.round_trips = round_trips
#         self.round_trip_output_samples = round_trip_output_samples
#         if round_trip_output_samples > self.round_trips:
#             self.round_trip_output_samples = self.round_trips
#         self.high_res_sampling = high_res_sampling
#         self.high_res_sampling_limits = high_res_sampling_limits
#         if self.high_res_sampling_limits[0] < 0:
#             self.high_res_sampling_limits[0] = 0
#         if self.high_res_sampling_limits[1] > self.round_trips:
#             self.high_res_sampling_limits[1] = self.round_trips
#         self.high_res_sample_interval = high_res_sample_interval

#         # Set active_fibre.oscillator = True -- Replenish pump each round trip
#         # Set verbosity of active fibre to verbosity of optical assembly.
#         for c in self.components:
#             if (isinstance(c, bc.step_index_active_fibre)
#                     or isinstance(c, bc.photonic_crystal_active_fibre)):
#                 c.oscillator = True
#                 c.verbose = self.verbose

#     @assembly._simulate
#     def simulate(self, pulse):
#         """
#         Simulate an SM fibre laser.

#         pulse: pulse class. Starting field used in simulations.

#         Returns the pulse class.
#         """
#         # Only used if self.high_res_sampling, but also only needs to be set
#         # once so keep it out of the loop.
#         pulse.high_res_sample_interval = self.high_res_sample_interval

#         if pulse.save_high_res_samples and pulse.save_dir is None:
#             raise Exception("pulse.save_dir cannot be NoneType if "
#                             "pulse.save_high_res_samples == True")

#         for i in range(self.round_trips):
#             print(i)
#             pulse.roundtrip_reset()  # Reset single-use member variables

#             # Handle high-resolution field sampling
#             if (self.high_res_sampling
#                     and i == self.high_res_sampling_limits[0]):  # Turn on
#                 pulse.high_res_samples = True
#                 component_locations = [0]  # 0 for start of 1st component
#             if (self.high_res_sampling
#                     and i == self.high_res_sampling_limits[1]):  # Turn off
#                 pulse.high_res_samples = False

#             # Propagate through each component
#             for val in self.components:
#                 pulse = val.propagate(pulse)

#                 # If sampling is active, retrieve component locations and skip
#                 # coupling transmission objects inserted into self.components
#                 # by make_full_components_list.
#                 if (self.high_res_sampling
#                         and i == self.high_res_sampling_limits[0]):
#                     loc = np.sum(pulse.high_res_field_sample_points)
#                     if loc != component_locations[-1]:
#                         component_locations.append(loc)

#                 # Handle NaN solutions
#                 if np.any(np.isnan(pulse.field)):
#                     pulse.output.append(np.ones_like(pulse.field) * np.nan)
#                     return self.update_pulse_class(pulse, pulse.output)
#                 else:
#                     pulse.add_OPPM_noise(self.grid, True)
#                     pulse.get_photon_spectrum(self.grid, pulse.field)

#             # Handle output field sampling
#             if i >= self.round_trips - self.round_trip_output_samples:
#                 pulse.output_samples.append(pulse.output)

#             # Save high-resolution field samples if sampling is active and if
#             # pulse.save_high_res_samples == True. This parameter must be set
#             # in the pulse object outside of the sm_fibre_laser class along
#             # with the directory that the data is saved under.
#             if (self.high_res_sampling and pulse.save_high_res_samples
#                     and self.high_res_sampling_limits[0] <= i
#                     <= self.high_res_sampling_limits[1]):
#                 pulse.save_field(
#                     str(i), component_locations=component_locations)

#         self.update_pulse_class(pulse, pulse.output)

#         if self.plot:
#             self.plot_spectra(pulse)
#             self.plot_pulse(pulse)
#             if self.sampling:
#                 self.plot_B_integral(pulse)
#                 self.plot_pulse_samples(pulse)
#                 self.plot_energy_and_average_power(pulse)

#         return pulse


class sm_fibre_amplifier(assembly):
    """
    Class for use as a template for other SM fibre amplifiers.

    Only supports simulations where the boundary value conditions for the
    active fibre are solved.

    James Feehan, 19/3/2022
    """

    def __init__(self, grid, components, co_ASE=None, high_res_sampling=None,
                 plot=False, name=None, data_directory=None, verbose=True):
        """
        grid: grid class.
        components: list of component objects. Must appear in order.
            Recommended that components_list[0] is gain.
            Coupling losses between components are added automatically.
            An active fibre must be present in components and
            boundary_value_solver needs to be enabled.
        co_ASE: numpy array (or None).
            ASE from a previous amplifier that co-propagates in the core with
            the signal.
        high_res_sampling : Nonetype or int
            if sampling : int
                Number of samples to take along the propagation axis
            if not sampling : None (default)
        plot: bool.
            Create plot information.
            True if high_res_sampling.
        name: str.
            String identifier for the amplifier object. Cannot be None if
            plot == True
        data_directory : str or Nonetype
            Save the data to data_directory if not None
        verbose : bool
            Print information about the simulation at runtime
        """
        super().__init__(
            grid, components, wrap=False, high_res_sampling=high_res_sampling,
            plot=plot, name=name, data_directory=data_directory,
            verbose=verbose)

        # Copy gain fibre (some methods also useful here)
        for c in self.components:
            if (isinstance(c, bc.step_index_active_fibre)
                    or isinstance(c, bc.photonic_crystal_active_fibre)):
                self.gain_fibre = c
        if not self.gain_fibre.boundary_value_solver:
            raise ValueError(
                "Full ASE is required for sm_fibre_amplifier simulations.")

        if self.sampling:
            if (self.gain_fibre.num_steps
                    < self.gain_fibre.L / self.num_samples):
                self.num_samples = self.gain_fibre.num_steps / 10
                if self.num_samples < 2:
                    self.num_samples = 2

        if co_ASE is not None:
            self.add_co_core_ASE(co_ASE)

    def add_co_core_ASE(self, co_ASE):
        """
        Add co-propagating ASE to the relevant gain fibre pump/ASE channels.

        Parameters
        ----------
        co_ASE: numpy array (or None).
            ASE from a previous amplifier that co-propagates in the core with
            the signal.
        """
        if self.gain_fibre.cladding_pumping:
            co_ASE = self.scale_co_core_ASE(
                co_ASE, self.gain_fibre.co_core_ASE.omega_window)
            self.gain_fibre.co_core_ASE.spectrum += co_ASE
        else:
            co_ASE = self.scale_co_core_ASE(
                co_ASE, self.gain_fibre.pump.omega_window)
            self.gain_fibre.pump.spectrum += co_ASE

    def scale_co_core_ASE(self, spectrum, omega_axis):
        """
        Core ASE light at the input of the amplifier that is generated by
        previous amplifiers needs to be scaled by the components which come
        before the gain fibre of this amplifier.

        Only required if co_ASE is not None at constructor.
        Input analogue to self.make_co_core_ASE_output_spectra.

        spectrum: numpy array. Either self.gain_fibre.co_core_ASE.spectrum OR
            self.gain_fibre.pump.spectrum AFTER co_ASE has been added. Which
            spectrum is used depends on self.gain_fibre.cladding_pumping
            (former if True).
        original_omega_window. numpy array.
            Either self.gain_fibre.co_core_ASE.omega_window OR
            self.gain_fibre.pump.omega_window, depending on whether
            self.gain_fibre.cladding_pumping is True (former) or
            False (latter).

        James Feehan, 14/5/2022
        """
        before_g = True
        for c in self.components:
            # Switch before check to make sure gain not applied
            # Exception handling as not all components have a transmission
            # function.
            if (isinstance(c, bc.step_index_active_fibre)
                    or isinstance(c, bc.photonic_crystal_active_fibre)):
                before_g = False
            if before_g:
                try:
                    spectrum = c.propagate_spectrum(spectrum, omega_axis)
                except AttributeError:
                    pass
        return spectrum

    @assembly._simulate
    def simulate(self, pulse):
        """
        Simulate an SM fibre amplifier.

        pulse: pulse class. Starting field used in simulations.

        Returns the pulse class.
        """
        # infostring = '\nSimulating    %s' % self.name
        # infostring += '\n' + '-'*len(infostring)
        # print(infostring)

        pulse.get_ESD_and_PSD(self.grid, pulse.field)
        self.input_pulse_PSD = pulse.power_spectral_density

        # Only used if self.sampling, but also only needs to be set
        # once so keep it out of the loop.
        if self.sampling:
            pulse.high_res_sample_interval = \
                self.gain_fibre.L / self.num_samples
            pulse.num_samples = self.num_samples

        if pulse.save_high_res_samples and pulse.save_dir is None:
            raise Exception("pulse.save_dir cannot be NoneType if "
                            "pulse.save_high_res_samples == True")

        # Handle high-resolution field sampling
        if self.sampling:  # Turn on
            pulse.high_res_samples = True
            component_locations = [0]  # 0 for start of 1st component

        # Propagate through each component
        for val in self.components:
            if self.verbose:
                print('\n' + val.__class__.__name__)
            pulse = val.propagate(pulse)

            # Sample pulse PSD immediately after gain fibre
            if (isinstance(val, bc.step_index_active_fibre)
                    or isinstance(val, bc.photonic_crystal_active_fibre)):
                pulse.get_ESD_and_PSD(self.grid, pulse.field)
                self.pulse_PSD_after_gain = pulse.power_spectral_density

            # If sampling is active, retrieve component locations and skip
            # coupling transmission objects inserted into self.components
            # by make_full_components_list.
            if self.sampling:
                loc = np.sum(pulse.high_res_field_sample_points)
                if loc != component_locations[-1]:
                    component_locations.append(loc)

            # Handle NaN solutions
            if np.any(np.isnan(pulse.field)):
                pulse.output.append(np.ones_like(pulse.field) * np.nan)
                return self.update_pulse_class(pulse, pulse.field)
            else:
                pulse.add_OPPM_noise(self.grid, True)
                pulse.get_photon_spectrum(self.grid, pulse.field)

        # Save high-resolution field samples if sampling is active and if
        # pulse.save_high_res_samples == True. This parameter must be set
        # in the pulse object outside of the sm_fibre_amplifier class along
        # with the directory that the data is saved under.
        if self.sampling and pulse.save_high_res_samples:
            pulse.save_field(self.filename,
                             component_locations=component_locations)

        if self.sampling:  # Turn off
            pulse.high_res_samples = False

        self.update_pulse_class(pulse, pulse.field)

        # Make ASE array passed to next amplifier:
        self.make_co_core_ASE_output_spectra()
        if self.plot:
            self.make_display_spectra(pulse)
            self.plot_spectra(pulse)
            self.plot_pulse(pulse)
            self.plot_inversion()
            self.plot_integration_error()
            if self.sampling:
                self.plot_B_integral(pulse)
                self.plot_change_in_B_over_gain_fibre(pulse)
                self.plot_gain_fibre_spectral_samples(
                    pulse.repetition_rate)
                self.plot_pulse_samples(pulse)
                self.plot_energy_and_average_power(pulse)
                self.plot_pump_powers_over_gain_fibre()
        if self.save_data:
            self.save(pulse, self.grid)
        return pulse

    def make_co_core_ASE_output_spectra(self):
        """
        Make arrays containing the co-propagating core ASE PSDs after they
        have been modified by components that come after the gain fibre.

        These arrays are passed to subsequent amplifiers as the starting
        conditions for co-propagating ASE.

        This method creates self.co_core_ASE_ESD_output.
        This method neglects cladding modes (assumed to be isolated effectively
        by spatial constraints).
        """
        if self.gain_fibre.cladding_pumping:
            self.co_core_ASE_ESD_output \
                = self.gain_fibre.co_core_ASE.propagated_spectrum.copy()
            omega_axis = self.gain_fibre.co_core_ASE.omega_window.copy()
        else:
            self.co_core_ASE_ESD_output \
                = self.gain_fibre.pump.propagated_spectrum.copy()
            omega_axis = self.gain_fibre.pump.omega_window.copy()

        before_g = True
        for c in self.components:
            if not before_g:
                try:
                    self.co_core_ASE_ESD_output = \
                        c.propagate_spectrum(
                            self.co_core_ASE_ESD_output, omega_axis)
                except AttributeError:
                    pass
            if (isinstance(c, bc.step_index_active_fibre)
                    or isinstance(c, bc.photonic_crystal_active_fibre)):
                before_g = False

    def make_display_spectra(self, pulse):
        """
        Organise spectra for plots.

        pulse: pulse class. Starting field used in simulations.

        James Feehan, 12/5/2022
        """
        if self.gain_fibre.cladding_pumping:
            # Sum all contributions from the CORE with the same propagation
            # direction immediately after gain.
            # Cladding modes, self.gain_fibre.pump.propagated_PSD and
            # self.gain_fibre.counter_pump.propagated_PSD, are NOT added here.
            f = interp1d(
                self.gain_fibre.co_core_ASE.lambda_window,
                self.gain_fibre.co_core_ASE.propagated_PSD, axis=1,
                fill_value='extrapolate')
            self.co_core_ASE_PSD = f(self.grid.lambda_window)
            self.net_forwards_PSD = self.pulse_PSD_after_gain \
                + self.co_core_ASE_PSD
            f = interp1d(
                self.gain_fibre.counter_core_ASE.lambda_window,
                self.gain_fibre.counter_core_ASE.propagated_PSD, axis=1,
                fill_value='extrapolate')
            self.counter_core_ASE_PSD = f(self.grid.lambda_window)
            self.net_backwards_PSD = \
                self.gain_fibre.counter_pulse.power_spectral_density \
                + self.counter_core_ASE_PSD
            f = interp1d(
                self.gain_fibre.pump.lambda_window,
                self.gain_fibre.pump.propagated_PSD, axis=1,
                fill_value='extrapolate')
            self.forwards_cladding_PSD = f(self.grid.lambda_window)
            f = interp1d(
                self.gain_fibre.counter_pump.lambda_window,
                self.gain_fibre.counter_pump.propagated_PSD, axis=1,
                fill_value='extrapolate')
            self.backwards_cladding_PSD = f(self.grid.lambda_window)

            # Get net output PSD (i.e., not just from gain fibre)
            # Requires new rep. rate if a pulse picker has been simulated.
            _, self.co_core_ASE_PSD_output = \
                self.gain_fibre.co_core_ASE.get_ESD_and_PSD(
                    self.co_core_ASE_ESD_output, pulse.repetition_rate)
            f = interp1d(
                self.gain_fibre.co_core_ASE.lambda_window,
                self.co_core_ASE_PSD_output, axis=1, fill_value='extrapolate')
            self.net_amplifier_output = \
                pulse.power_spectral_density + f(self.grid.lambda_window)
        else:
            # Sum all contributions with the same direction.
            f = interp1d(
                self.gain_fibre.pump.lambda_window,
                self.gain_fibre.pump.propagated_PSD, axis=1,
                fill_value='extrapolate')
            self.core_co_pump_PSD = f(self.grid.lambda_window)
            self.net_forwards_PSD = self.pulse_PSD_after_gain \
                + self.core_co_pump_PSD
            f = interp1d(
                self.gain_fibre.counter_pump.lambda_window,
                self.gain_fibre.counter_pump.propagated_PSD, axis=1,
                fill_value='extrapolate')
            self.core_counter_pump_PSD = f(self.grid.lambda_window)
            self.net_backwards_PSD = \
                self.gain_fibre.counter_pulse.power_spectral_density \
                + self.core_counter_pump_PSD

            # Get the net output PSD (i.e., not just from the gain fibre).
            # Requires new rep. rate if a pulse picker has been simulated.
            _, self.co_core_ASE_PSD_output = \
                self.gain_fibre.pump.get_ESD_and_PSD(
                    self.co_core_ASE_ESD_output, pulse.repetition_rate)
            f = interp1d(
                self.gain_fibre.pump.lambda_window,
                self.co_core_ASE_PSD_output, axis=1, fill_value='extrapolate')
            self.net_amplifier_output = \
                pulse.power_spectral_density + f(self.grid.lambda_window)

    def plot_spectra(self, pulse):
        """
        Make the spectral plots and display them
        """
        # summed spectra -- gain fibre only
        min_y = 1e-6
        handles = []
        fig = Figure()  # figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.set_title('Gain fibre: net PSDs')
        h1, = ax.semilogy(
            self.grid.lambda_window * 1e9, np.sum(self.input_pulse_PSD, axis=0),
            c='mediumseagreen', alpha=0.75, label='Input')
        h2, = ax.semilogy(
            self.grid.lambda_window * 1e9, np.sum(self.net_forwards_PSD, axis=0),
            c='darkorange', alpha=0.75, label='Co')
        h3, = ax.semilogy(
            self.grid.lambda_window * 1e9,
            np.sum(self.net_backwards_PSD, axis=0),
            c='mediumpurple', alpha=0.75, label='Counter')
        handles.extend([h1, h2, h3])
        if self.gain_fibre.cladding_pumping:
            h4, = ax.semilogy(
                self.grid.lambda_window * 1e9,
                np.sum(self.forwards_cladding_PSD, axis=0), c='darkslateblue',
                alpha=0.75, label='Co clad')
            h5, = ax.semilogy(
                self.grid.lambda_window * 1e9,
                np.sum(self.backwards_cladding_PSD, axis=0), c='steelblue',
                alpha=0.75, label='Counter clad')
            handles.extend([h4, h5])
        ax.set_xlabel('Wavelength, nm', fontsize=13)
        ax.set_ylabel('Power spectral density, mW/nm', fontsize=13)
        max_y = max(self.input_pulse_PSD.max(), self.net_forwards_PSD.max(),
                    self.net_backwards_PSD.max())
        if self.gain_fibre.cladding_pumping:
            max_y = max(max_y, self.forwards_cladding_PSD.max(),
                        self.backwards_cladding_PSD.max())
        ax.set_ylim([min_y, 4 * max_y])
        ax.set_xlim([1e9 * self.gain_fibre.wl_lims[0],
                     1e9 * self.gain_fibre.wl_lims[1]])

        fmt = []
        self.plot_dict[self.name + ': gain fibre, net PSDs'] = (ax, fmt)

        # Individual spectra -- gain fibre only
        handles = []
        axis_str = [r'$x$', r'$y$']
        linestyles = ['-', '-.']  # for polarization axes
        alphas = [0.55, 1]
        fig = Figure()  # figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.set_title('Gain fibre: individual PSDs')
        for i, ls in enumerate(linestyles):
            h1, = ax.semilogy(
                self.grid.lambda_window * 1e9, self.pulse_PSD_after_gain[i, :],
                ls=ls, c='mediumseagreen', alpha=alphas[i], lw=2,
                label='Pulse, ' + axis_str[i])
            handles.append(h1)
            if self.gain_fibre.cladding_pumping:
                h2, = ax.semilogy(
                    self.gain_fibre.pump.lambda_window * 1e9,
                    self.gain_fibre.pump.propagated_PSD[i, :],
                    ls=ls, c='darkslateblue', alpha=alphas[i], lw=2,
                    label='Co, clad, ' + axis_str[i])
                h3, = ax.semilogy(
                    self.gain_fibre.pump.lambda_window * 1e9,
                    self.gain_fibre.counter_pump.propagated_PSD[i, :],
                    ls=ls, c='steelblue', alpha=alphas[i], lw=2,
                    label='Counter, clad, ' + axis_str[i])
                h4, = ax.semilogy(
                    self.gain_fibre.co_core_ASE.lambda_window * 1e9,
                    self.gain_fibre.co_core_ASE.propagated_PSD[i, :],
                    c='darkorange', alpha=alphas[i], lw=2,
                    label='Co ASE, ' + axis_str[i])
                h5, = ax.semilogy(
                    self.gain_fibre.counter_core_ASE.lambda_window * 1e9,
                    self.gain_fibre.counter_core_ASE.propagated_PSD[i, :],
                    ls=ls, c='mediumpurple', alpha=alphas[i], lw=2,
                    label='Counter ASE, ' + axis_str[i])
                handles.extend([h2, h3, h4, h5])
            else:
                h6, = ax.semilogy(
                    self.gain_fibre.pump.lambda_window * 1e9,
                    self.gain_fibre.pump.propagated_PSD[i, :],
                    ls=ls, c='darkslateblue', alpha=alphas[i], lw=2,
                    label='Co pump, ' + axis_str[i])
                h7, = ax.semilogy(
                    self.gain_fibre.counter_pump.lambda_window * 1e9,
                    self.gain_fibre.counter_pump.propagated_PSD[i, :],
                    ls=ls, c='steelblue', alpha=alphas[i], lw=2,
                    label='Counter pump, ' + axis_str[i])
                handles.extend([h6, h7])
        ax.set_xlabel('Wavelength, nm', fontsize=13)
        ax.set_ylabel('Power spectral density, mW/nm', fontsize=13)
        max_y = self.pulse_PSD_after_gain.max()
        if self.gain_fibre.cladding_pumping:
            max_y = max(max_y, self.forwards_cladding_PSD.max(),
                        self.backwards_cladding_PSD.max(),
                        self.co_core_ASE_PSD.max(),
                        self.counter_core_ASE_PSD.max())
        else:
            max_y = max(max_y, self.core_co_pump_PSD.max(),
                        self.core_counter_pump_PSD.max())
        ax.set_ylim([min_y, 2 * max_y])
        xmin = min(self.gain_fibre.wl_lims[0],
                   self.gain_fibre.pump.lambda_lims[0])
        xmax = max(self.gain_fibre.wl_lims[1],
                   self.gain_fibre.pump.lambda_lims[1])
        ax.set_xlim([1e9 * xmin, 1e9 * xmax])
        fmt = []
        self.plot_dict[self.name + ': gain fibre, individual PSDs'] = (ax, fmt)

        # Net amplifier output -- co-propagating only
        fig = Figure()  # figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.set_title('Net amplifier output')
        for i, ls in enumerate(linestyles):
            ax.semilogy(
                self.grid.lambda_window * 1e9, pulse.power_spectral_density[i, :],
                ls=ls, c='mediumseagreen', alpha=alphas[i], lw=2,
                label='Pulse, ' + axis_str[i])
            if self.gain_fibre.cladding_pumping:
                ax.semilogy(
                    self.gain_fibre.co_core_ASE.lambda_window * 1e9,
                    self.co_core_ASE_PSD_output[i, :], ls=ls, c='darkslateblue',
                    alpha=alphas[i], lw=2, label='Co core, ' + axis_str[i])
            else:
                ax.semilogy(
                    self.gain_fibre.pump.lambda_window * 1e9,
                    self.co_core_ASE_PSD_output[i, :], ls=ls, c='darkslateblue',
                    alpha=alphas[i], lw=2, label='Co core, ' + axis_str[i])
            ax.semilogy(
                self.grid.lambda_window * 1e9, self.net_amplifier_output[i, :],
                ls=ls, c='darkorange', alpha=alphas[i], lw=2,
                label='Net, ' + axis_str[i])
        ax.set_xlabel('Wavelength, nm', fontsize=13)
        ax.set_ylabel('Power spectral density, mW/nm', fontsize=13)
        ax.set_ylim([min_y, 2 * self.net_amplifier_output.max()])
        xmin = min(self.gain_fibre.wl_lims[0],
                   self.gain_fibre.pump.lambda_lims[0])
        xmax = max(self.gain_fibre.wl_lims[1],
                   self.gain_fibre.pump.lambda_lims[1])
        ax.set_xlim([1e9 * xmin, 1e9 * xmax])
        fmt = []
        self.plot_dict[self.name + ': net amplifier output'] = (ax, fmt)

    def plot_inversion(self):
        """
        Plot the inversion as a function of fibre length.
        """
        z = np.linspace(
            0, self.gain_fibre.L, len(self.gain_fibre.inversion_vs_distance))
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_title('Inversion profile')
        ax.plot(z, self.gain_fibre.inversion_vs_distance, c='indianred')
        ax.set_xlim([0, self.gain_fibre.L])
        ax.set_ylim([0, 100])
        ax.set_xlabel('z, m', fontsize=13)
        ax.set_ylabel('Population inversion, %', fontsize=13)

        fmt = ['axes.fill_between(ax.lines[0].get_data()[0], 0, '
               + 'ax.lines[0].get_data()[1], color="indianred", alpha=0.33)']
        self.plot_dict[
            self.name + ': gain fibre, inversion profile'] = (ax, fmt)

    def plot_integration_error(self):
        """
        Plot the error for the ESD and full-field propagation.
        """
        ln = len(self.gain_fibre.boundary_value_solver_ESD_optimization_loss)
        ln_field = len(
            self.gain_fibre.boundary_value_solver_field_optimization_loss)
        mx = max(
            self.gain_fibre.boundary_value_solver_ESD_optimization_loss.max(),
            self.gain_fibre.boundary_value_solver_field_optimization_loss.max()
        )
        mn = min(
            self.gain_fibre.boundary_value_solver_ESD_optimization_loss.min(),
            self.gain_fibre.boundary_value_solver_field_optimization_loss.min()
        )

        ESD_err_ax = np.arange(0, ln, 1)
        field_err_ax = np.arange(ESD_err_ax.max() + 1,
                                 ESD_err_ax.max() + ln_field + 1, 1)
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_title('Full ASE optimization loss')
        h1, = ax.semilogy(
            ESD_err_ax,
            self.gain_fibre.boundary_value_solver_ESD_optimization_loss, 'o',
            ms=6, c='cornflowerblue', markeredgecolor='k', label='ESD')
        h2, = ax.semilogy(
            field_err_ax,
            self.gain_fibre.boundary_value_solver_field_optimization_loss,
            'o', ms=6, c='darkorange', markeredgecolor='k', label='Field')
        ax.plot([ESD_err_ax.max(), ESD_err_ax.max()], [0, 1.1 * mx], ls='--',
                c='k', alpha=0.75)
        ax.set_xlabel('Iteration', fontsize=13)
        ax.set_ylabel('Error', fontsize=13)
        ax.fill_between([-0.5, ESD_err_ax.max()], 0, 1.1 * mx,
                        color='cornflowerblue', alpha=0.2, edgecolor=None)
        ax.fill_between([ESD_err_ax.max(), field_err_ax.max() + 0.5], 0,
                        1.1 * mx, color='darkorange', alpha=0.2,
                        edgecolor=None)
        ax.set_ylim([mn, 1.1 * mx])
        ax.set_xlim([-0.5, field_err_ax.max() + 0.5])

        ESD_err_ax_max_str = str(ESD_err_ax.max())
        field_err_ax_max_str = str(field_err_ax.max() + 0.5)
        max_str = str(1.1 * mx)
        fmt = [
            "axes.fill_between([-0.5," + ESD_err_ax_max_str + "] , 0, " +
            max_str + ", color='cornflowerblue', alpha=0.2, edgecolor=None)",
            "axes.fill_between([" + ESD_err_ax_max_str + ", "
            + field_err_ax_max_str + "], 0, " + max_str
            + ", color='darkorange', alpha=0.2, edgecolor=None)",
            "self.fb1, = plotWidget.canvas.axes.fill(" +
            "np.NaN, np.NaN, 'cornflowerblue', alpha=0.2, label='ESD')",
            "self.fb2, = plotWidget.canvas.axes.fill(" +
            "np.NaN, np.NaN, 'darkorange', alpha=0.2, label='Field')",
            "axes.legend([(self.fb1, ax.get_legend_handles_labels()[0][0])," +
            "(self.fb2, ax.get_legend_handles_labels()[0][1])]," +
            " ['ESD', 'Field'])"]
        self.plot_dict[
            self.name + ': full ASE optimization loss'] = (ax, fmt)

    def plot_gain_fibre_spectral_samples(self, rep_rate):
        """
        Plot self.gain_fibre.spectral_samples as heatmaps.
        """
        samples = np.sum(self.gain_fibre.spectral_samples, axis=1)
        self.net_co_PSD_samples = np.zeros((len(self.gain_fibre.dz_samples),
                                            self.grid.points))

        # Don't want zeros her - numpy complains.
        self.net_counter_PSD_samples = 1e-80 * np.ones_like(
            self.net_co_PSD_samples)
        for k in self.gain_fibre.stacks.slices.keys():
            f = interp1d(
                self.gain_fibre.stacks.lambda_window[
                    self.gain_fibre.stacks.slices[k]],
                samples[:, self.gain_fibre.stacks.slices[k]], axis=1,
                fill_value='extrapolate')
            s = f(self.grid.lambda_window)
            s[s < 0] = np.amin(np.abs(s))
            if k.startswith('co_'):
                self.net_co_PSD_samples += utils.get_ESD_and_PSD(
                    self.grid.lambda_window, s, rep_rate)[1]
            elif k.startswith('counter_'):
                self.net_counter_PSD_samples += utils.get_ESD_and_PSD(
                    self.grid.lambda_window, s, rep_rate)[1]

        d = 1
        if self.grid.points >= 512:
            d = int(self.grid.points / 512)

        Y = np.cumsum(self.gain_fibre.dz_samples)

        vmax = np.round(
            10 * np.log10(
                max(self.net_co_PSD_samples.max(),
                    self.net_counter_PSD_samples.max())),
            decimals=1)
        norm = Normalize(vmin=-40, vmax=vmax)
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_title('Co light, gain fibre')
        p = ax.pcolormesh(
            1e9 * self.grid.lambda_window[::d], Y,
            10 * np.log10(self.net_co_PSD_samples[:, ::d]),
            cmap='cubehelix_r', norm=norm)
        ax.set_xlabel('Wavelength, nm')
        ax.set_ylabel('Position along the gain fibre, m')

        ax.p = p  # MUST be used for pcolormesh
        ax.colorbar_label = 'PSD, dBm/nm'
        fmt = []
        self.plot_dict[
            self.name + ': Co-PSD vs z, gain fibre'] = \
            (ax, fmt)

        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_title('Counter light, gain fibre')
        p = ax.pcolormesh(
            1e9 * self.grid.lambda_window[::d], Y[::-1],
            10 * np.log10(self.net_counter_PSD_samples[::-1, ::d]),
            cmap='cubehelix_r', norm=norm)
        ax.set_xlabel('Wavelength, nm')
        ax.set_ylabel('Position along the gain fibre, m')

        ax.p = p
        ax.colorbar_label = 'PSD, dBm/nm'
        fmt = []
        self.plot_dict[
            self.name + ': Counter-propagating spectrum vs z, gain fibre'] = \
            (ax, fmt)

    def plot_change_in_B_over_gain_fibre(self, pulse):
        """
        Same as assembly but just for the gain fibre.
        """
        B = np.asarray(self.gain_fibre.B_samples).T

        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_title('Increase in B integral over the gain fibre')
        ax.semilogy(
            np.cumsum(self.gain_fibre.dz_samples), B, c='mediumaquamarine',
            lw=2)
        ax.set_xlabel('z, m', fontsize=13)
        ax.set_ylabel('B integral, rad.', fontsize=13)
        ax.set_xlim([0, self.gain_fibre.L])
        ax.set_ylim([B.min(), B.max()])
        fmt = ['axes.fill_between(ax.lines[0].get_data()[0], 0, '
               + 'ax.lines[0].get_data()[1], color="mediumaquamarine", '
               + 'alpha=0.33)']
        self.plot_dict[
            self.name + ': B integral over gain fibre'] = (ax, fmt)

    def plot_pump_powers_over_gain_fibre(self):
        """
        Plot the power in the co- and counter-propagating pump and ASE channels
        over the length of the gain fibre.
        """
        co_power = []
        num_samples = self.gain_fibre.pump.high_res_samples.shape[0]
        for i in range(num_samples):
            co_power.append(
                np.sum(self.gain_fibre.pump.high_res_samples[i, :, :]
                       * self.gain_fibre.pump.d_wl * 1e6))
        co_power = np.asarray(co_power)
        max_P = np.amax(co_power)
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_title('Power in the pump & ASE channels\n'
                     'over the gain fibre')
        ax.plot(np.cumsum(self.gain_fibre.dz_samples), co_power, c='seagreen')
        ax.set_xlabel('z, m', fontsize=13)
        ax.set_ylabel('Power, W', fontsize=13)
        ax.yaxis.label.set_color('seagreen')
        ax.set_xlim([0, self.gain_fibre.L])
        fmt = ['axes.fill_between(ax.lines[0].get_data()[0], 0, '
               + 'ax.lines[0].get_data()[1], color="seagreen", '
               + 'alpha=0.33)']
        legend_fmt = "axes.legend(['Co'"
        if self.gain_fibre.boundary_value_solver:
            counter_power = []
            num_samples = self.gain_fibre.counter_pump.high_res_samples.shape[0]
            for i in range(num_samples):
                counter_power.append(np.sum(
                    self.gain_fibre.counter_pump.high_res_samples[i, :, :]
                    * self.gain_fibre.counter_pump.d_wl * 1e6))
            counter_power = np.asarray(counter_power)
            ax.plot(np.cumsum(self.gain_fibre.dz_samples), counter_power, c='darkorchid')
            legend_fmt += ", 'Counter'"
            fmt.append('axes.fill_between(ax.lines[1].get_data()[0], 0, '
               + 'ax.lines[1].get_data()[1], color="darkorchid", '
               + 'alpha=0.33)')
            if np.amax(counter_power) > np.amax(co_power):
                max_P = np.amax(counter_power)
        ax.set_ylim([0, 1.1 * max_P])
        legend_fmt += "])"
        fmt.append(legend_fmt)
        self.plot_dict[
            self.name + ': Pump and ASE power over gain fibre'] = (ax, fmt)

    @assembly.saver
    def save(self):
        """
        Save all data to self.directory.

        Notes
        -----
        The data is saved using the numpy.savez method and can be accessed
        using the numpy.load method, which returns a dictionary containing all
        of the data saved by this method. E.g.,:
        amplifier_data = numpy.load('/path/gain_fibre.npz')
        net_co_PSD_samples = amplifier_data['net_co_PSD_samples']

        Dictionary keys:
        sample_points : locations along the fibre length at which the data is
            sampled.
        net_co_PSD_samples : Total forwards-propagating PSD vs distance.
        net_counter_PSD_samples : Total backwards-propagating PSD vs distance.
        boundary_value_solver_ESD_optimization_loss : convergence error of
            the boundary value solver when solving for energy spectral density
            propagation only.
        boundary_value_solver_field_optimization_loss : convergence error of
            the boundary value solver when solving for full-field propagation.
        inversion_vs_distance : population inversion vs propagation distance.
        net_co_PSD : Total forwards-propagating power spectral density.
        net_counter_PSD : Total backwards-propagating power spectral density.

        if gain_fibre.cladding_pumping == True, the following dictionary keys
        are added:
        co_cladding_PSD : Total power spectral density of the forwards-
            propagating cladding light
        counter_cladding_PSD : Total power spectral density of the backwards-
            propagating cladding light.
        """
        z = np.linspace(0, self.gain_fibre.L,
                        len(self.gain_fibre.inversion_vs_distance))
        savez_dict = dict()
        if self.gain_fibre.cladding_pumping:
            savez_dict = {
                'sample_points': z,
                'net_co_PSD_samples': self.net_co_PSD_samples,
                'net_counter_PSD_samples': self.net_counter_PSD_samples,
                'boundary_value_solver_ESD_optimization_loss': self.gain_fibre.boundary_value_solver_ESD_optimization_loss,
                'boundary_value_solver_field_optimization_loss': self.gain_fibre.boundary_value_solver_field_optimization_loss,
                'inversion_vs_distance': self.gain_fibre.inversion_vs_distance,
                'net_co_PSD': self.net_forwards_PSD,
                'net_counter_PSD': self.net_backwards_PSD,
                'co_cladding_PSD': self.forwards_cladding_PSD,
                'counter_cladding_PSD': self.backwards_cladding_PSD,
                'co_core_ASE_ESD_output': self.co_core_ASE_ESD_output,
                'pump_points': self.gain_fibre.pump.points,
                'pump_wl_lims': self.gain_fibre.pump.lambda_lims,
                'pump_lambda_window': self.gain_fibre.pump.lambda_window,
                'co_pump_PSD_samples': np.sum(self.gain_fibre.pump.high_res_samples, axis=1)[1::, :],
                'counter_pump_PSD_samples': np.sum(self.gain_fibre.counter_pump.high_res_samples, axis=1)[1::, :]}
        else:
            savez_dict = {
                'sample_points': z,
                'net_co_PSD_samples': self.net_co_PSD_samples,
                'net_counter_PSD_samples': self.net_counter_PSD_samples,
                'boundary_value_solver_ESD_optimization_loss': self.gain_fibre.boundary_value_solver_ESD_optimization_loss,
                'boundary_value_solver_field_optimization_loss': self.gain_fibre.boundary_value_solver_field_optimization_loss,
                'inversion_vs_distance': self.gain_fibre.inversion_vs_distance,
                'net_co_PSD': self.net_forwards_PSD,
                'net_counter_PSD': self.net_backwards_PSD,
                'co_core_ASE_ESD_output': self.co_core_ASE_ESD_output,
                'pump_points': self.gain_fibre.pump.points,
                'pump_wl_lims': self.gain_fibre.pump.lambda_lims,
                'pump_lambda_window': self.gain_fibre.pump.lambda_window,
                'co_pump_PSD_samples': np.sum(self.gain_fibre.pump.high_res_samples, axis=1)[1::, :],
                'counter_pump_PSD_samples': np.sum(self.gain_fibre.counter_pump.high_res_samples, axis=1)[1::, :]}
        np.savez(self.directory + "optical_assembly.npz", **savez_dict)
 
