#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Library for investigating the low-T/|W| instability.
#

__all__ = "calc_densfluct",


import yt
import numpy as np
from scipy.interpolate import interp1d

from ..io import gamer_io


def calc_densfluct(fn, radius, center, resolution = 1024, nbin = 128):
    """
    Compute the density fluctuation on the XY-plane, assuming the rotation axis is z-axis.

    Parameters
    ----------
    fn: string
        Path to the HDF5 snapshot.
    radius: float
        Half-width of the domain, in cm.
    center: array-like of float
        Coordinate of reference center, in cm.
    resolution: integer, optional
        Number of pixel in each direction.
    nbin: integer, optional
        Number of bins for constructing the azimuthally averaged density.
    """
    # use a larger range than the specified one for the averaged density profile
    rmax = yt.YTArray(np.sqrt(2) * radius, "cm")

    # obtain the density distribution on the XY-plane via the fixed-resolution buffer in yt
    gamer_obj = gamer_io()
    coord_x, coord_y, dens = gamer_obj.yt_slice(fn, field = "density", width = 2.0 * rmax,
                                                direction = "z", center = center, resolution = resolution)

    # compute the 1D averaged density profile
    coord_r  = np.sqrt(coord_x**2 + coord_y**2)
    bin_edge = np.linspace(0.0, rmax.v, nbin+1)

    bin_cen  = list()
    bin_dens = list()
    for idx in range(nbin):
        bin_rL = bin_edge[idx  ]
        bin_rR = bin_edge[idx+1]
        bin_rC = 0.5 * (bin_rL + bin_rR)

        mask = (coord_r > bin_rL) & (coord_r < bin_rR)

        if np.any(mask):
            bin_cen.append(bin_rC)
            bin_dens.append(dens[mask].mean())

    # compute the density fluctuation
    interp_func = interp1d(bin_cen, bin_dens,
                           bounds_error = False,
                           fill_value = (bin_dens[0], bin_dens[-1]))

    dens_ave   = interp_func(coord_r)
    dens_fluct = (dens - dens_ave) / dens_ave

    return coord_x, coord_y, dens_fluct
