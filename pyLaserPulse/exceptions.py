#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 16:59:13 2022

@author: james feehan

Custom exceptions: for when things go wrong.
"""


class PropagationMethodNotConvergingError(Exception):
    """Raised when an iterative propagation method does not converge"""
    pass


class BoundaryConditionError(Exception):
    """
    Raised when the boundary conditions in active_fibre_base are inadequately
    defined
    """
    pass


class NanFieldError(Exception):
    """Raised when the pulse field has NaN values"""
    pass


class GridDefinitionIncorrectError(Exception):
    """Raised when the grid definition is inappropriate"""
    pass


class PulseDecimationError(Exception):
    """Raised when pulse.interpolate_onto_another_grid is used for decimation"""


class GridCentresDonNotMatch(Exception):
    """
    Raised when pulse.interpolate_onto_another_grid is used, but the
    grid central wavelengths don't match
    """


class GridCentreNotInMaclaurinSeriesDomain(Exception):
    """
    Raised when utils.Maclaurin_coefficients is called and the series limits
    are specified differently to the sim_idx indices, but do not contain the
    grid midpoint index.
    """
