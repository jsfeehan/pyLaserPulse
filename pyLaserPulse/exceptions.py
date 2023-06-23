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
