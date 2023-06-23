#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jun 24 15:02:23 2022

@author: James Feehan

Get absolute paths for data files independently of the user profile, OS, etc.
"""


import os


main_dir = os.path.dirname(os.path.abspath(__file__))


class _components:
    class _loss_spectra:
        def __init__(self):
            local_dir = main_dir + '\\components\\loss_spectra\\'
            self.Andover_155FS10_25_bandpass = local_dir + \
                'Andover_155FS10_25_bandpass.dat'
            self.fused_WDM_976_1030 = local_dir + \
                'fused_WDM_976_1030.dat'
            self.Opneti_95_5_PM_fibre_tap_fast_axis_blocked = local_dir + \
                'Opneti_95_5_PM_fibre_tap_fast_axis_blocked.dat'
            self.Opneti_high_power_isolator_HPMIS_1030_1_250_5 = local_dir + \
                'Opneti_high_power_isolator_HPMIS_1030_1_250_5.dat'
            self.Opneti_microoptic_976_1030_wdm = local_dir + \
                'Opneti_microoptic_976_1030_wdm.dat'
            self.Opneti_microoptic_isolator_PMIS_S_P_1030_F = local_dir + \
                'Opneti_microoptic_isolator_PMIS_S_P_1030_F.dat'
            self.Thorlabs_IO_J_1030 = local_dir + 'Thorlabs_IO_J_1030.dat'

    def __init__(self):
        self.loss_spectra = self._loss_spectra()


class _materials:
    class _loss_spectra:
        def __init__(self):
            local_dir = main_dir + '\\materials\\loss_spectra\\'
            self.silica = local_dir + 'silica.dat'

    class _Raman_profiles:
        def __init__(self):
            local_dir = main_dir + '\\materials\\Raman_profiles\\'
            self.silica = local_dir + 'silica.dat'

    class _reflectivities:
        def __init__(self):
            local_dir = main_dir + '\\materials\\reflectivity_spectra\\'
            self.aluminium = local_dir + 'aluminium.dat'
            self.gold = local_dir + 'gold.dat'
            self.silver = local_dir + 'silver.dat'

    class _Sellmeier_coefficients:
        def __init__(self):
            local_dir = main_dir + '\\materials\\Sellmeier_coefficients\\'
            self.silica = local_dir + 'silica.dat'

    def __init__(self):
        self.loss_spectra = self._loss_spectra()
        self.Raman_profiles = self._Raman_profiles()
        self.reflectivities = self._reflectivities()
        self.Sellmeier_coefficients = self._Sellmeier_coefficients()


class _fibres:
    class _cross_sections:
        def __init__(self):
            local_dir = main_dir + '\\fibres\\cross_sections\\'
            self.Er_silica = local_dir + 'Er_silica.dat'
            self.Tm_silica = local_dir + 'Tm_silica.dat'
            self.Yb_Al_silica = local_dir + 'Yb_Al_silica.dat'
            self.Yb_Nufern_PM_YSF_HI_HP = local_dir + \
                'Yb_Nufern_PM_YSF_HI_HP.dat'

    def __init__(self):
        self.cross_sections = self._cross_sections()


class _single_plot_window:
    def __init__(self):
        # need to remove 'data' from main_dir
        local_dir = main_dir[0:-4] + '\\single_plot_window\\'
        self.icon = local_dir + 'icon.png'


fibres = _fibres()
materials = _materials()
components = _components()
single_plot_window = _single_plot_window()
