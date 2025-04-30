#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Library for computing various time scales.
#

__all__ = ("calc_timescale_heating",
           "calc_timescale_advection")


from math import pi

import yt
from scipy.interpolate import interp1d

from ..io import gamer_io


def calc_timescale_heating(fn):
    """
    Compute the heating timescale, which is defined as

        tau_heat = E / \dot{Q}

    where

        E       is the energy of the matter      in the gain region
        \dot{Q} is the net neutrino heating rate in the gain region.

    The definition of E varies across the literature:
        E is the binding energy : E = \eps + 0.5 * v^2 (+ \Phi) in Buras et al. 2006, A&A, 457, 281
                 internal energy: E = \eps                      in Murphy & Burrows 2008, ApJ, 688, 1159

    Here we adopt the definition in Murphy & Burrows (2008).

    Parameters
    ----------
    fn: string
        Path to the HDF5 snapshot.
    """
    # setting
    gamer_obj = gamer_io()
    gamer_obj.get_unitsys(fn)

    gain_region = gamer_obj.yt_cut_gainregion(fn)
    energyshift = gamer_obj.get_energyshift(fn)
    unitsys     = gamer_obj.unit

    # compute the specific internal energy
    try:
        sEint_CGS = gain_region["specific_thermal_energy"].in_cgs()
    except:
        # in case specific_thermal_energy is unavailable
        dens = gain_region["Dens"]
        momx = gain_region["MomX"]
        momy = gain_region["MomY"]
        momz = gain_region["MomZ"]
        engy = gain_region["Engy"]

        sEint = (engy - 0.5 * (momx**2 + momy**2 + momz**2) / dens) / dens
        sEint_CGS = sEint.in_cgs()

    # note that the energy shift must be subtracted
    Eth_CGS = (sEint_CGS.v - energyshift) * gain_region["cell_mass"].in_cgs()

    # compute the net heating rate in the gain region
    # --> note that we need to convert the unit manually since dEdt_Nu is dimensionless
    unit_dEdt = unitsys["V"]**2 * unitsys["D"] / unitsys["T"]
    dEdt_CGS  = gain_region["dEdt_Nu"] * unit_dEdt \
              * gain_region["cell_volume"].in_cgs()

    # compute the neutrino heating timescale
    Eth_total_CGS  = Eth_CGS.sum().to_value()
    dEdt_total_CGS = dEdt_CGS.sum().to_value()

    tau_heating = Eth_total_CGS / dEdt_total_CGS

    return Eth_total_CGS, dEdt_total_CGS, tau_heating


def calc_timescale_advection(fn, radius, center, logscale = False, nbin = 128):
    """
    Compute the advection timescale defined as

        tau_advect = M / \dot{M}

    for multi-dimension simulations, where

        M       is the mass enclosed in the gain region
        \dot{M} is the total mass flux in and out of the gain region.

    Note that this formula assumed steady flow implicitly, and the advection timescale
    could be underestimated as mode explodes (see Murphy & Burrows 2008, ApJ, 688, 1159).

    Regarding the aspherically shock expansion, the \dot{M} is better estimated at various
    radii, such as mean/max shock radii and the neutrino sphere.

    Parameters
    ----------
    fn: string
        Path to the HDF5 snapshot.
    radius: array-like of float
        Radius at where the mass flux is evaluated, in cm.
    center: array-like of float
        Coordinate of reference center, in cm.
    logscale: boolean, optional
        Indicates whether to apply a logarithmic scale to the bin edges.
    nbin: integer, optional
        Number of bins.
    """
    gamer_obj = gamer_io()

    # compute the total mass in the gain region
    gain_region    = gamer_obj.yt_cut_gainregion(fn)
    mass_CGS       = gain_region["cell_mass"].in_cgs()
    mass_total_CGS = mass_CGS.sum().to_value()

    # get the spherical averaged profile
    # --> compute the profile over a larger range for interpolation
    field_coord  = "pns_spherical_radius"
    rmax_scaling = 1.5
    rmax         = rmax_scaling * max(radius)

    fields      = [("gas", "density"), ("gas", "pns_velocity_spherical_radius")]
    kwargs_prof = {"units"       : {field_coord: "km"},
                   "logs"        : {field_coord: logscale},
                   "weight_field": ("gas", "cell_mass"),
                   "n_bins"      : nbin }

    profile = gamer_obj.get_sphave_profile(fn, fields, rmax, center = center,
                                           field_coord = field_coord, **kwargs_prof)
    rad, dens, vrad = profile.T

    # compute the mass flux at various radii
    M_dot = list()

    interp_func_dens = interp1d(rad, dens)
    interp_func_vrad = interp1d(rad, vrad)

    for r in radius:
        dens_target = interp_func_dens(r)
        vr_target   = interp_func_vrad(r)

        M_dot_target = -4.0 * r**2 * pi * dens_target * vr_target
        M_dot.append(M_dot_target)

    # compute the advection timescale
    tau_advect = [mass_total_CGS / dM  for dM in M_dot]

    return mass_total_CGS, *M_dot, *tau_advect
