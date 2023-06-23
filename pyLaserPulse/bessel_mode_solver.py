#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import numbers
from scipy.special import jv, kn
from scipy.optimize import root
import matplotlib.pyplot as plt


class bessel_mode_solver:
    """
    Find modes supported by a step index fibre.

    Based loosely on code in pyMMF and D. Marcuse, "Light Transmission Optics",
    Van Nostrand Reinhold, New York, 1972.
    """

    def __init__(self, core_rad, clad_rad, n_co, n_cl, wl, tol=1e-9):
        """
        Parameters
        ----------
        core_rad : float
            Radius of the core in m
        clad_rad : float
            Radius of the cladding in m
        n_co : float
            Refractive index of the core
        n_cl : float
            Refractive index of the cladding
        wl : float
            Wavelength of the light in m
        """
        self.k = 2 * np.pi / wl
        self.core_rad = core_rad
        self.clad_rad = clad_rad
        self.n_co = n_co
        self.n_cl = n_cl
        self.V = self.k * self.core_rad * np.sqrt(self.n_co**2 - self.n_cl**2)
        self.m = 0
        self.number = 0
        self.tol = tol

    def LP_eigenvalue_equation(self, u):
        """
        Find my roots for guided modes!

        Parameters
        ----------
        u : numpy array
            u = r_core * sqrt(n_core^2 * k^2 - beta^2)

        Returns
        -------
        numpy array
            (J_v(u) / (u J_v-1(u))) + (K_v(w) / (w K_v-1(w)))

        Notes
        -----
        LP modes can be found by finding the roots to this function.
        See D. Marcuse, "Light Transmission Optics", Van Nostrand Reinhold,
        New York, 1972.
        """
        w = np.sqrt(self.V**2 - u**2)
        term_1 = jv(self.m, u) / (u * jv(self.m - 1, u))
        term_2 = kn(self.m, w) / (w * kn(self.m - 1, w))
        return term_1 + term_2

    def root_func(self, u):
        """
        Wrapper for using scipy.optimize.root to find the roots of
        self.LP_eigenvalue_equation.

        Parameters
        ----------
        u : numpy array
            u = r_core * sqrt(n_core^2 * k^2 - beta^2)

        Returns
        -------
        scipy OptimizeResult object
            Solution (i.e., the roots of self.LP_eigenvalue_equation).
        """
        return root(self.LP_eigenvalue_equation, u, tol=self.tol)

    def solve(self, max_modes=500):
        """
        Find the LP modes of an ideal step-index fibre.

        Parameters
        ----------
        max_modes : int
            Maximum number of modes to find
        """
        beta_list = []
        u_list = []
        w_list = []
        m_list = []
        l_list = []
        self.num_modes = 0
        roots = [0]
        interval = np.arange(0, self.V, self.V * 1e-4)

        with np.errstate(all='ignore'):
            # Suppress numpy RuntimeWarning for zero-division error in
            # Bessel mode solver. Beyond my control...
            while len(roots) and self.num_modes < max_modes:
                guesses = np.argwhere(np.abs(np.diff(np.sign(
                    self.LP_eigenvalue_equation(interval)))))
                sols = map(self.root_func, interval[guesses])
                roots = [s.x for s in sols if s.success]
                roots = np.unique([np.round(r / self.tol) * self.tol
                                   for r in roots if
                                   (r > 0 and r < self.V)]).tolist()
                roots_num = len(roots)

                if roots_num:
                    degeneracy = 1 if self.m == 0 else 2
                    beta_list.extend(
                        [np.sqrt((self.k * self.n_co)**2
                                 - (r / self.core_rad)**2) for r in roots]
                        * degeneracy)
                    u_list.extend(roots * degeneracy)
                    w_list.extend([np.sqrt(self.V**2 - r**2) for r in roots]
                                  * degeneracy)
                    l_list.extend(
                        [x + 1 for x in range(roots_num)] * degeneracy)
                    m_list.extend([self.m] * roots_num * degeneracy)
                    self.num_modes += roots_num * degeneracy
                self.m += 1
        self.beta_arr = np.asarray(beta_list)
        self.u_arr = np.asarray(u_list)
        self.w_arr = np.asarray(w_list)
        self.m_arr = np.asarray(m_list)
        self.l_arr = np.asarray(l_list)

    def make_modes(self, r, num_modes=1):
        """
        Return an array containing the mode shapes.

        Parameters
        ----------
        r : numpy array
            radial (polar) axis.
        num_modes : int.
            Number of mode profiles to calculate (starts from LP01)
            if num_modes > self.num_modes, default to self.num_modes
        """
        if num_modes > self.num_modes:
            num_modes = self.num_modes
        num_modes = int(num_modes)
        idx = np.linspace(0, num_modes - 1, num_modes, dtype=int)
        m = self.m_arr[idx]
        u = self.u_arr[idx]
        w = self.w_arr[idx]
        r_tiled = r[:, None].repeat(num_modes, axis=1)
        R = r_tiled / self.core_rad  # normalized radius
        self.modes = jv(m, u * R)
        idx = r >= self.core_rad
        self.modes[idx, :] = (jv(m, u) / kn(m, w)) * kn(m, w * R[idx, :])

    def get_amplitude_distribution(self, std=None):
        """
        Calculate an incoherent sum of all modes.

        Parameters
        ----------
        std
            If None, then the modes are assumed to contain equal energy. If
            numeric, it is used as the standard deviation for a normal
            distribution which scales the mode energy (favouring low-order
            modes).
        """
        self.amplitude_distribution = np.abs(self.modes)
        self.amplitude_distribution /= np.sum(
            self.amplitude_distribution, axis=0)
        n_modes = self.modes.shape[0]
        if std is not None:
            if not isinstance(std, numbers.Number):
                raise ValueError("std in bessel_mode_solver.incoherent_sum"
                                 + " Must be a number.")
            else:
                mode_idx = np.linspace(0, n_modes - 1, n_modes)
                scale = np.exp(-mode_idx**2 / std**2)
                self.amplitude_distribution = self.amplitude_distribution.T
                self.amplitude_distribution *= scale
                self.amplitude_distribution = self.amplitude_distribution.T
        self.amplitude_distribution = np.sum(
            self.amplitude_distribution, axis=1)
        self.amplitude_distribution /= np.sum(self.amplitude_distribution)


if __name__ == "__main__":
    modes = 4000
    sol = bessel_mode_solver(125e-6, 160e-6, 1.45, 1.378, 976e-9)
    sol.solve(max_modes=modes)

    r = np.linspace(0, sol.clad_rad, 2**11)
    sol.make_modes(r, num_modes=modes)
    sol.get_amplitude_distribution()
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(r * 1e6, sol.modes[:, 0:4000], c='seagreen', alpha=0.15)
    ax.plot(r * 1e6,
            sol.amplitude_distribution / sol.amplitude_distribution.max(),
            c='k')
    ax.set_ylabel('Amplitude, arb.')
    ax.set_xlabel(r'radius, $\mu$m')
    plt.show()
