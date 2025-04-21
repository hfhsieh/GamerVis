#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Library for estimating the accretion rate at the speicified radius using yt.
#
#    We currently assumes the PNS center locates at the box center.
#

__all__ = ("calc_massenc",
           "calc_accretion_shell",
           "calc_accretion_profile")


from math import pi

import yt
from scipy.interpolate import interp1d

from ..io import gamer_io


def calc_massenc(fn_list, radius, center):
    """
    Compute the enclosed mass within the specified radius
    for estimating the mass accretion rate via post-processing.

    Parameters
    ----------
    fn_list: array-like of string
        Filename of HDF5 snapshots to be processed.
    radius: float
        Radius at where the enclosed mass is computed, in cm.
    center: array-like of float
        Coordinate of reference center, in cm.
    """
    assert len(fn_list) == len(center), "Inconsistent size between fn_list and cneter."

    data = list()

    for fn, c in zip(fn_list, center):
        ds = yt.load(fn)
        gamer_io._yt_addfield_sph_pns(None, ds, center = c)

        ad     = ds.all_data()
        shpere = ad.include_below(("pns_sph_radius"), radius)
        mass   = shpere["cell_mass"].sum().v  # in gram

        data.append(float(mass))

    return data


def calc_accretion_shell(fn, radius, width, center):
    """
    Estimate the accretion rate via the mass flow within [radius - width, radius + width].

    Parameters
    ----------
    fn: string
        Path to the HDF5 snapshot.
    radius: float
        Radius at where the accretion rate is estimated, in cm.
    width: float
        Half-width of the shell, in cm.
    center: array-like of float
        Coordinate of reference center, in cm.
    """
    ds = yt.load(fn)
    gamer_io._yt_addfield_sph_pns(None, ds, center = center)

    ad    =    ds.all_data()
    shell =    ad.include_above(("pns_sph_radius"), radius - width)
    shell = shell.include_below(("pns_sph_radius"), radius + width)

    mass     = shell["cell_mass"].v
    vrad     = shell["pns_sph_vradius"].v
    acc_rate = -sum(mass * vrad) / (2.0 * width)  # arbitrarily normalize by 2.0 * width

    return float(acc_rate)


def calc_accretion_profile(fn, radius, center, logscale = False, nbin = 64):
    """
    Construct the spherically averaged profile using yt.
    Then, interpolate the profiles to obtain the density and radial velocity
    at the specified radius for estimating the accretion rate.

    Parameters
    ----------
    fn: string
        Path to the HDF5 snapshot.
    radius: float
        Radius at where the accretion rate is estimated, in cm.
    center: array-like of float
        Coordinate of reference center, in cm.
    logscale: boolean, optional
        Indicates whether to apply a logarithmic scale to the bin edges.
    nbin: integer, optional
        Number of bins.
    """
    # compute the profile over a larger range for interpolation
    rmax_scaling = 1.5
    rmax         = yt.YTArray(rmax_scaling * radius, "cm")

    # profile setting
    field_coord = "pns_sph_radius"
    fields      = [("gas", "density"), ("gas", "pns_sph_vradius")]
    kwargs_prof = {"units"       : {field_coord: "km"},
                   "logs"        : {field_coord: logscale},
                   "extrema"     : {field_coord: (None, rmax)},
                   "weight_field": ("gas", "cell_mass"),
                   "n_bins"      : nbin }

    # construct the spherically averaged profile using get_sphave_profile() in the gamer_io class
    gamer_obj = gamer_io()

    profile = gamer_obj.get_sphave_profile(fn, fields, rmax, center = center,
                                           field_coord = field_coord, **kwargs_prof)
    rad, dens, vrad = profile.T

    # compute the accretion rate at the specified radius
    interp_func_dens = interp1d(rad, dens)
    interp_func_vrad = interp1d(rad, vrad)

    dens_target = interp_func_dens(radius)
    vrad_target = interp_func_vrad(radius)

    acc_rate = -4.0 * pi * radius**2 * dens_target * vrad_target

    return acc_rate
