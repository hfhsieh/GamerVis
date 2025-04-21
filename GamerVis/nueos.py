#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Class for invoking the nuclear EoS solver.
#

__all__ = "nueos",


from numpy import nan
from NuclearEoS import NuclearEoS


class nueos():
    def __init__(self, nuctable):
        """
        Parameters
        ----------
        nuctable: string
            Path to the nuclear EoS table.
        """
        self.nuctable  = nuctable
        self.solver    = NuclearEoS(nuctable)
        self.engyshift = self.solver.energy_shift

    def get_thermo_temp(self, dens, ye, temp, target_field, **kwargs):
        """
        Invoke the solver in temperature mode to obtain the target thermodynamic quantity.

        Parameters
        ----------
        dens: float
            Mass density on a linear scale, in g/cm3.
        ye: float
            Electron fraction on a linear scale.
        temp: float
            Temperature on a linear scale, in Kelvin.
        target_field: string or array-like of string, optional
            Name of target thermodynamic quantity.
        """
        try:
            values = self.solver.get_thermovar(dens = dens, ye = ye, temp = temp,
                                               mode = "temp", target_field = target_field)
            if isinstance(target_field, str):
                return values[target_field]
            else:
                return value

        except ValueError:
            # in case the input values are invalid
            return nan

    def get_thermo_engy(self, dens, ye, engy, target_field, **kwargs):
        """
        Invoke the solver in energy mode to obtain the target thermodynamic quantity.

        Parameters
        ----------
        dens: float
            Mass density on a linear scale, in g/cm^3.
        ye: float
            Electron fraction on a linear scale.
        engy: float
            Internal energy on a linear scale, in cm^2/s^2.
        target_field: string or array-like of string, optional
            Name of target thermodynamic quantity.
        """
        # note that the internal energy (or thermal energy) stored in the GAMER snapshots
        # includes the energy shift
        engy -= self.engyshift

        try:
            values = self.solver.get_thermovar(dens = dens, ye = ye, engy = engy,
                                               mode = "engy", target_field = target_field)
            if isinstance(target_field, str):
                return values[target_field]
            else:
                return value

        except ValueError:
            # in case the input values are invalid
            return nan

    def get_thermo_entr(self, dens, ye, entr, target_field, **kwargs):
        """
        Invoke the solver in entropy mode to obtain the target thermodynamic quantity.

        Parameters
        ----------
        dens: float
            Mass density on a linear scale, in g/cm3.
        ye: float
            Electron fraction on a linear scale.
        entr: float
            Entropy on a linear scale, in kB/baryon.
        target_field: string or array-like of string, optional
            Name of target thermodynamic quantity.
        """
        try:
            values = self.solver.get_thermovar(dens = dens, ye = ye, entr = entr,
                                               mode = "entr", target_field = target_field)
            if isinstance(target_field, str):
                return values[target_field]
            else:
                return value

        except ValueError:
            # in case the input values are invalid
            return nan
