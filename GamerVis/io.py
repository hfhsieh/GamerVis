#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Class for handling GAMER data for CCSN simulations.
#

__all__ = ("gamer_ascii",
           "gamer_hdf5",
           "gamer_io"    )


import os
import re
from glob import glob

import yt
import h5py
import numpy as np
from scipy.interpolate import interp1d

from .fmt import *
from .helper import *
from .constant import *


class gamer_ascii():
    """
    Functions for handling ASCII files.
    """
    def __init__(self, rundir = "."):
        """
        Parameters
        ----------
        rundir: string, optional
            Path to the simulation data.
        """
        self.rundir = rundir

    def _loader_centquant(self):
        """
        Load the data in Record__CentralQuant.
        """
        fn = os.path.join(self.rundir, "Record__CentralQuant")

        # check whether the leakage scheme is adopted, as the output format differs
        if self.get_param("NEUTRINO_SCHEME") == "LEAKAGE":
            fmt_data = fmt_centquant_leakage
        else:
            fmt_data = fmt_centquant_lightbulb

        data = np.genfromtxt(fn, dtype = fmt_data)

        # remove duplicate data
        _, idx = np.unique(data["step"], return_index = True)

        return data[idx]

    def _loader_quadmom(self):
        """
        Load the data in Record__QuadMom_2nd.
        """
        fn   = os.path.join(self.rundir, "Record__QuadMom_2nd")
        data = np.genfromtxt(fn, dtype = fmt_quadmom)

        # remove duplicate data
        _, idx = np.unique(data["step"], return_index = True)

        return data[idx]

    def _get_param_ascii(self, keyword, kind = "note"):
        """
        Retrieve the value corresponding to the specified keyword in Record__Note/Input__TestProb,
        assuming it is a single-column entry.

        Parameters
        ----------
        keyword: string
            Name of the runtime parameter as used in Input__Parameter. Case-sensitive.
        kind: string, optional
            Kind of target file to search.
            --> "note"    : Record__Note.
                "testprob": Input__TestProb.
        """
        assert kind in ["note", "testprob"], "Unsupported kind: {}".format(kind)

        if kind == "note":
            fn      = os.path.join(self.rundir, "Record__Note")
            keyword = keyword.upper()  # convert to upper case
        else:
            fn = os.path.join(self.rundir, "Input__TestProb")

        assert os.path.isfile(fn), "File {} does not exist!".format(fn)

        with open(fn, "r") as f:
            value = ""

            # search the file for the keyword
            for line in f:
                if keyword in line:
                    value = line.strip().split()[1]

            assert value, "Keyword {} not found!".format(keyword)

        return convert_datatype(value)


class gamer_hdf5():
    """
    Functions for handling HDF5 files using h5py and yt.
    """
    Pattern_YTField  = r"obj\[(.*?)\]"

    def __init__(self, rundir = "."):
        """
        Parameters
        ----------
        rundir: string, optional
            Path to the simulation data.
        """
        self.rundir = rundir

    def _h5py_get_phystime(self, fn):
        """
        Retrieve the physical time in the specified HDF5 snapshot, in code unit.
        """
        with h5py.File(fn, "r") as h5obj:
            phys_time = h5obj["Info"]["KeyInfo"]["Time"]

        return phys_time[0].tolist()

    def _h5py_get_unitsystem(self, fn):
        """
        Retrieve the unit system in the specified HDF5 snapshot.
        """
        units = dict()

        with h5py.File(fn, "r") as h5obj:
            units["L"] = h5obj["Info"]["InputPara"]["Unit_L"]  # length
            units["M"] = h5obj["Info"]["InputPara"]["Unit_M"]  # mass
            units["T"] = h5obj["Info"]["InputPara"]["Unit_T"]  # time
            units["V"] = h5obj["Info"]["InputPara"]["Unit_V"]  # velocity
            units["D"] = h5obj["Info"]["InputPara"]["Unit_D"]  # density
            units["E"] = h5obj["Info"]["InputPara"]["Unit_E"]  # energy
            units["P"] = h5obj["Info"]["InputPara"]["Unit_P"]  # pressure or energy density

            if h5obj["Info"]["Makefile"]["Magnetohydrodynamics"]:
                units["B"] = h5obj["Info"]["InputPara"]["Unit_B"]  # magnetic field

        # convert to float
        for key, value in units.items():
            units[key] = value.tolist()

        return units

    def _yt_get_param_hdf5(self, fn, keyword):
        """
        Retrieve the runtime parameter recorded in the given HDF5 snapshot.

        Parameters
        ----------
        fn: string
            Path to the HDF5 snapshot.
        keyword: string
            Name of the runtime parameter as used in Input__Parameter. Case-sensitive.
        """
        ds    = yt.load(fn)
        value = ds.parameters[keyword]

        return convert_datatype_numpy(value)

    def _yt_addfield_ye(self, ds):
        """
        Add the electron fraction field with name of ("gas", "ye").
        """
        def _ye(field, data):
            return yt.YTArray(data["Ye"].v / data["Dens"].v, "dimensionless")

        ds.add_field(("gas", "ye"), function = _ye,
                     units = "dimensionless", sampling_type = "cell")

    def _yt_addfield_cyl_pns(self, ds, center):
        """
        Add cylindrical coordinates and the corresponding angular velocity
        relative to the specified reference center.
        """
        Is_BoxCenter = np.all(ds.domain_center.in_cgs() == center)

        def _pns_cyl_radius(field, data):
            if Is_BoxCenter:
                return data["cylindrical_radius"]
            else:
                x = data["x"] - center[0]
                y = data["y"] - center[1]

                return np.sqrt(x * x + y * y)

        def _pns_cyl_theta(field, data):
            if Is_BoxCenter:
                return data["cylindrical_theta"]
            else:
                x = data["x"] - center[0]
                y = data["y"] - center[1]

                return np.arctan2(y, x)

        def _pns_cyl_z(field, data):
            if Is_BoxCenter:
                return data["cylindrical_z"]
            else:
                return data["z"] - center[2]

        def _pns_cyl_vradius(field, data):
            if Is_BoxCenter:
                return data["velocity_cylindrical_radius"]
            else:
                theta = data["pns_cylindrical_theta"]
                vx    = data["velocity_x"]
                vy    = data["velocity_y"]

                return np.cos(theta) * vx + np.sin(theta) * vy

        def _pns_cyl_vtheta(field, data):
            if Is_BoxCenter:
                return data["velocity_cylindrical_theta"]
            else:
                theta = data["pns_cylindrical_theta"]
                vx    = data["velocity_x"]
                vy    = data["velocity_y"]

                return -np.sin(theta) * vx + np.cos(theta) * vy

        def _pns_cyl_vz(field, data):
            return data[("gas", "velocity_cylindrical_z")]

        def _pns_cyl_omega(field, data):
            if Is_BoxCenter:
                return data["velocity_cylindrical_theta"]     / data["cylindrical_radius"]
            else:
                return data["pns_velocity_cylindrical_theta"] / data["pns_cylindrical_radius"]

        ds.add_field(("gas", "pns_cylindrical_radius"),          function = _pns_cyl_radius,
                     units = "cm",            display_name = r"Cylindrical Radius",              sampling_type = "cell")
        ds.add_field(("gas", "pns_cylindrical_theta"),           function = _pns_cyl_theta,
                     units = "dimensionless", display_name = r"Cylindrical Theta",               sampling_type = "cell")
        ds.add_field(("gas", "pns_cylindrical_z"),               function = _pns_cyl_z,
                     units = "cm",            display_name = r"Cylindrical Z",                   sampling_type = "cell")
        ds.add_field(("gas", "pns_velocity_cylindrical_radius"), function = _pns_cyl_vradius,
                     units = "cm/s",          display_name = r"Cylindrical Radial Velocity$",    sampling_type = "cell")
        ds.add_field(("gas", "pns_velocity_cylindrical_theta"),  function = _pns_cyl_vtheta,
                     units = "cm/s",          display_name = r"Cylindrical Azimuthal Velocity$", sampling_type = "cell")
        ds.add_field(("gas", "pns_velocity_cylindrical_z"),      function = _pns_cyl_vz,
                     units = "cm/s",          display_name = r"Cylindrical Vertical Velocity$",  sampling_type = "cell")
        ds.add_field(("gas", "pns_oemga_cylindrical"),           function = _pns_cyl_omega,
                     units = "1/s",           display_name = r"$\Omega$",                        sampling_type = "cell")

    def _yt_addfield_sph_pns(self, ds, center):
        """
        Add spherical coordinates and velocities relative to the specified reference center.
        """
        Is_BoxCenter = np.all(ds.domain_center.in_cgs() == center)

        def _pns_sph_radius(field, data):
            if Is_BoxCenter:
                return data["spherical_radius"]
            else:
                x = data["x"] - center[0]
                y = data["y"] - center[1]
                z = data["z"] - center[2]

                return np.sqrt(x * x + y * y + z * z)

        def _pns_sph_theta(field, data):
            if Is_BoxCenter:
                return data["spherical_theta"]
            else:
                r = data["pns_spherical_radius"]
                z = data["z"] - center[2]

                return np.arccos(z / r)

        def _pns_sph_phi(field, data):
            if Is_BoxCenter:
                return data["spherical_phi"]
            else:
                x = data["x"] - center[0]
                y = data["y"] - center[1]

                return np.arctan2(y, x)

        def _pns_sph_vradius(field, data):
            if Is_BoxCenter:
                return data["velocity_spherical_radius"]
            else:
                x  = data["x"] - center[0]
                y  = data["y"] - center[1]
                z  = data["z"] - center[2]
                vx = data["velocity_x"]
                vy = data["velocity_y"]
                vz = data["velocity_z"]
                r  = np.sqrt(x * x + y * y + z * z)

                return (x * vx + y * vy + z * vz) / r

        def _pns_sph_vtheta(field, data):
            if Is_BoxCenter:
                return data["velocity_spherical_theta"]
            else:
                x   = data["x"] - center[0]
                y   = data["y"] - center[1]
                z   = data["z"] - center[2]
                vx  = data["velocity_x"]
                vy  = data["velocity_y"]
                vz  = data["velocity_z"]
                rho = np.sqrt(x * x + y * y)
                r   = np.sqrt(rho * rho + z * z)

                return (z * (x * vx + y * vy) - (x * x + y * y) * vz) / (r * rho)

        def _pns_sph_vphi(field, data):
            if Is_BoxCenter:
                return data["velocity_spherical_phi"]
            else:
                x  = data["x"] - center[0]
                y  = data["y"] - center[1]
                vx = data["velocity_x"]
                vy = data["velocity_y"]

                return (-y * vx + x * vy) / np.sqrt(x * x + y * y)

        ds.add_field(("gas", "pns_spherical_radius"),          function = _pns_sph_radius,
                     units = "cm",            display_name = r"Spherical Radius",             sampling_type = "cell")
        ds.add_field(("gas", "pns_spherical_theta"),           function = _pns_sph_theta,
                     units = "dimensionless", display_name = r"Spherical Theta",              sampling_type = "cell")
        ds.add_field(("gas", "pns_spherical_phi"),             function = _pns_sph_phi,
                     units = "dimensionless", display_name = r"Spherical Phi",                sampling_type = "cell")
        ds.add_field(("gas", "pns_velocity_spherical_radius"), function = _pns_sph_vradius,
                     units = "cm/s",          display_name = r"Spherical Radial Velocity",    sampling_type = "cell")
        ds.add_field(("gas", "pns_velocity_spherical_theta"),  function = _pns_sph_vtheta,
                     units = "1/s",           display_name = r"Spherical Polar Velocity",     sampling_type = "cell")
        ds.add_field(("gas", "pns_velocity_spherical_phi"),    function = _pns_sph_vphi,
                     units = "1/s",           display_name = r"Spherical Azimuthal Velocity", sampling_type = "cell")

    def _yt_addfield_mri_N2(self, ds, eos):
        """
        Add the field representing the square of Brunt-Vaisala frequency (N2)
        along with its dependencies for MRI analysis.

        Parameters
        ----------
        ds: class instance
            yt object.
        eos: class instance
            Nuclear EoS solver.
        """
        # adiabatic index, dlnP / dln\rho at constant entropy
        # --> adopt the values in the nuclear EoS table
        def _gamma1(field, data):
            # note that eint contains the energy shift
            dens = data[("gas", "density")].in_cgs()
            ye   = data[("gas", "ye")]
            eint = data[("gas", "specific_thermal_energy")].in_cgs()

            gamma1 = [eos.get_thermo_engy(d.v, y.v, e.v, "gamma")
                      for d, y, e in zip(dens.ravel(), ye.ravel(), eint.ravel())]
            gamma1 = np.reshape(gamma1, dens.shape)

            return yt.YTArray(gamma1, "dimensionless")

        # function generator for adding field defined along each direction
        def _vecG_generator(direction):
            def _vecG(field, data):
                key = "gas", "pressure_gradient_{}".format(direction)

                return data[key] / data[("gas", "density")]

            return _vecG

        def _vecB_generator(direction):
            def _vecB(field, data):
                key_dens = "gas", "density_gradient_{}".format(direction)
                key_pres = "gas", "pressure_gradient_{}".format(direction)

                dens_grad = data[key_dens] / data[("gas", "density")]
                pres_grad = data[key_pres] / data[("gas", "pressure")]
                gamma1    = data[("gas", "gamma1")]

                return dens_grad - pres_grad / gamma1

            return _vecB

        # add the required fields
        self.yt_check_field(ds, "ye")

        ds.add_field(("gas", "gamma1"), function = _gamma1, units = "dimensionless",
                     display_name = r"$\Gamma_1$", sampling_type = "local")

        for direction in "xyz":
            _vecG_func = _vecG_generator(direction)
            _vecB_func = _vecB_generator(direction)

            field_name   = "gas", "vecG_{}".format(direction)
            display_name = "vecG{}".format(direction)
            ds.add_field(field_name, function = _vecG_func, units = "cm/s**2",
                         display_name = display_name, sampling_type = "cell")

            field_name   = "gas", "vecB_{}".format(direction)
            display_name = "vecB{}".format(direction)
            ds.add_field(field_name, function = _vecB_func, units = "1/cm",
                         display_name = display_name, sampling_type = "cell")

        # now add the N2 field
        def _mri_N2(field, data):
            return sum(data[("gas", b)] * data[("gas", g)]
                       for b, g in zip(("vecB_x", "vecB_y", "vecB_z"),
                                       ("vecG_x", "vecG_y", "vecG_z")) )

        ds.add_field(("gas", "mri_N2"), function = _mri_N2, units = "1/s**2",
                     display_name = r"$N^2$", sampling_type = "cell")

    def _yt_addfield_mri_O2(self, ds, center):
        """
        Add the field representing the square of angular frequency
        along with its dependencies for MRI analysis.

        Parameters
        ----------
        ds: class instance
            yt object.
        center: array-like of float
            Coordinate of reference center, in cm.
        """
        self.yt_check_field(ds, "pns_cylindrical_radius", center = center)

        def _pns_cyl_omega2(field, data):
            return data["pns_oemga_cylindrical"]**2

        ds.add_field(("gas", "mri_O2"), function = _pns_cyl_omega2, units = "1/s**2",
                     display_name = r"$\Omega^2$", sampling_type = "cell")

    def _yt_addfield_mri_Rvarpi(self, ds):
        """
        Add the field representing the stability criterion of shear instabilities
        for a differentially rotating fluid, \varpi * \partial_\varpi \Omega^2 (R_omega),
        for MRI analysis.
        """
        # add the gradient of the new field ("gas", "mri_O2") with the periodic boundary conditions
        ds.force_periodicity()
        ds.add_gradient_fields(("gas", "mri_O2"))

        def _mri_Rvarpi(field, data):
            rad           = data[("gas", "pns_cylindrical_radius")]
            theta         = data[("gas", "pns_cylindrical_theta")]
            omega2_grad_x = data[("gas", "mri_O2_gradient_x")]
            omega2_grad_y = data[("gas", "mri_O2_gradient_y")]

            return rad * (np.cos(theta) * omega2_grad_x + np.sin(theta) * omega2_grad_y)

        ds.add_field(("gas", "mri_Rvarpi"), function = _mri_Rvarpi, units = "1/s**2",
                     display_name = r"$R_\varpi$", sampling_type = "cell")

    def _yt_addfield_mri(self, ds, eos, center):
        """
        Add the C90 field and its associated fields for MRI analysis.

        Reference:
            Obergaulinger et al., 2009, A&A, 498, 241

        Parameters
        ----------
        ds: class instance
            yt object.
        eos: class instance
            Nuclear EoS solver.
        center: array-like of float
            Coordinate of reference center, in cm.
        """
        self._yt_addfield_mri_N2(ds, eos)
        self._yt_addfield_mri_O2(ds, center)
        self._yt_addfield_mri_Rvarpi(ds)

        def _mri_c90(field, data):
            return (data["mri_N2"] + data["mri_Rvarpi"]) / data["mri_O2"]

        def _mri_N2O2(field, data):
            return data["mri_N2"] / data["mri_O2"]

        def _mri_RvO2(field, data):
            return data["mri_Rvarpi"] / data["mri_O2"]

        def _mri_lambda(field, data):
            velo_alfven = data["alfven_speed"]
            N2          = data["mri_N2"]
            Rvarpi      = data["mri_Rvarpi"]

            return 2.0 * np.sqrt(2) * np.pi * np.abs(velo_alfven) / np.sqrt(-N2 - Rvarpi)

        ds.add_field(("gas", "mri_C90"), function = _mri_c90, units = "dimensionless",
                     display_name = "C90", sampling_type = "cell")
        ds.add_field(("gas", "mri_N2O2"), function = _mri_N2O2, units = "dimensionless",
                     display_name = r"$N^2 / \Omega^2$", sampling_type = "cell")
        ds.add_field(("gas", "mri_RvO2"), function = _mri_RvO2, units = "dimensionless",
                     display_name = r"$R_\varpi / \Omega^2$", sampling_type = "cell")
        ds.add_field(("gas", "mri_lambda"), function = _mri_lambda, units = "cm",
                     display_name = r"$\lambda_\mathrm{MRI}$", sampling_type = "cell")

    def _yt_get_fieldname(self, string):
        """
        Retrieve the field name in the format of "obj[FIELDNAME]"
        """
        matches = re.findall(self.Pattern_YTField, string)

        if matches:
            return matches[0]
        else:
            return None

    def yt_check_field(self, ds, field, center = None, eos = None):
        """
        Checks if the specified field exists in the ds object,
        and adds it as a derived field if it is not present.
        """
        if   "ye" in field and ("gas", "ye") not in ds.derived_field_list:
            self._yt_addfield_ye(ds)

        elif "pns_cylindrical_radius" in field and ("gas", "pns_cylindrical_radius") not in ds.derived_field_list:
            assert center is not None, "Center is not provided."
            self._yt_addfield_cyl_pns(ds, center)

        elif "pns_spherical_radius" in field and ("gas", "pns_spherical_radius") not in ds.derived_field_list:
            assert center is not None, "Center is not provided."
            self._yt_addfield_sph_pns(ds, center)

        elif "mri" in field and ("gas", field) not in ds.derived_field_list:
            assert eos    is not None, "EoS solver provided."
            assert center is not None, "Center is not provided."
            self._yt_addfield_mri(ds, eos, center)

        else:
            # do nothing
            pass

    def yt_cut_gainregion(self, fn):
        """
        Get the gain region.
        """
        ds = yt.load(fn)
        ad = ds.all_data()

        return ad.cut_region("obj['dEdt_Nu'] > 0.0")

    def yt_get_coord_pns(self, fn):
        """
        Get the coordinates of the highest-density cell.
        """
        ds     = yt.load(fn)
        ad     = ds.all_data()
        region = ad.cut_region('obj["gas", "density"] > 1e14')

        _, coord = ds.find_max("density", source = region)
        coord = coord.in_cgs().tolist()

        return coord

    def yt_slice(self, fn, field, width, direction, center, resolution = 1024, eos = None):
        """
        A wrapper function for the yt.slice() function.

        Parameters
        ----------
        fn: string
            Path to the HDF5 snapshot.
        field: string
            Name of target field.
        width: float
            Width of the domain, in cm.
        direction: string
            Direction normal to the slice plane.
        center: array-like of float
            Coordinate of reference center, in cm.
        resolution: integer, optional
            Number of pixel in each direction.
        eos: class instance, optional
            Nuclear EoS solver.
        """
        if direction == "x":
            label_x = "y"
            label_y = "z"
            center_axis = center[0]
        elif direction == "y":
            label_x = "x"
            label_y = "z"
            center_axis = center[1]
        elif direction == "z":
            label_x = "x"
            label_y = "y"
            center_axis = center[2]

        ds = yt.load(fn)
        self.yt_check_field(ds, field, center = center, eos = eos)

        slc      = ds.slice(axis = direction, coord = center_axis, center = center)
        frb_data = slc.to_frb(width = width, resolution = resolution)

        # shift coordinates relative to the reference center
        coord_x = frb_data["x"].in_cgs() - center[0]
        coord_y = frb_data["y"].in_cgs() - center[1]
        data    = frb_data[field].in_cgs()

        return coord_x.v, coord_y.v, data.v

    def yt_sliceplot(self, ds, field, direction, center, eos = None,
                     cb_lim = None, cb_logscale = None, show_grid = False, zoom = None,
                     roi_cond = None, fontsize = None, title = ""):
        """
        A wrapper function for the yt.SlicePlot() function.

        Parameters
        ----------
        ds: class instance
            yt object.
        field: string
            Name of target field.
        direction: string
            Direction of the slice plot.
        center: array-like of float
            Coordinate of reference center, in cm.
        eos: class instance, optional
            Nuclear EoS solver.
        cb_lim: array-like of float, optional
            Range displayed on the color bar.
        cb_logscale: boolean, optional
            Indicates whether to apply a logarithmic scale on the color bar.
        show_grid: boolean, optional
            Indicates whether to display the grid.
        zoom: float, optional
            Zoom factor.
        roi_cond: string or array-like of string, optional
            A list of conditions used to define the region of interest.
            Cells that do not match all conditions will be treated as background,
            which will be displayed in white if specified.
        fontsize: float, optional
            Font size for the displayed figure.
        title: string, optional
            Figure title.
        """
        self.yt_check_field(ds, field, center = center, eos = eos)

        # specify the width directly, rather than using the zoom factor,
        # for better control over the color bar range
        width = yt.YTArray(ds.parameters["BoxSize"][0] * ds.parameters["Unit_L"], "cm")

        if zoom:
            width /= zoom

        # set up the background
        if roi_cond:
            # add the required derived fields
            for cond in roi_cond:
                field_cond = self._yt_get_fieldname(cond)
                self.yt_check_field(ds, field_cond, center = center, eos = eos)

            region_ROI = ds.sphere(center, 0.5 * np.sqrt(2) * width)
            region_ROI = region_ROI.cut_region(roi_cond)
        else:
            region_ROI = None

        # plot
        # --> display the x and y coordinates in km
        slc = yt.SlicePlot(ds, direction, field, data_source = region_ROI,
                           center = center, width = width.in_units("km"))

        # decoration
        if title:
            slc.annotate_title(title)

        if cb_lim:
            slc.set_zlim(field, *cb_lim)

        if cb_logscale is not None:
            slc.set_log(field, log = cb_logscale)

        if show_grid:
            slc.annotate_grids()

        if fontsize:
            slc.set_font_size(fontsize)

        return slc


class gamer_io(gamer_ascii, gamer_hdf5):
    """
    Functions for handling files generated by GAMER.
    """
    Pattern_FileName = r"Data_(\d+)"

    def __init__(self, rundir = "."):
        """
        Parameters
        ----------
        rundir: string, optional
            Path to the simulation data.
        """
        gamer_ascii.__init__(self, rundir = rundir)
        gamer_hdf5.__init__ (self, rundir = rundir)

        self.rundir          = rundir
        self.param           = dict()  # dictionary  for storing the runtime parameters
        self.centquant       = None    # numpy array for storing the data in Record__CentralQuant
        self.centquant_field = None    # field name in the stored self.centquant
        self.quadmom         = None    # numpy array for storing the data in Record__QuadMom_2nd
        self.unit            = None    # dictionary  for storing the unit system
        self.unit_stamp      = None    # integer serving as a stamp for the loaded unit system

    def interp_centquant(self, field, fn = None, time = None):
        """
        Compute the interpolated value of the target field in Record__CentralQuant
        at the specified time or the physical time recorded in the HDF5 snapshot.

        Parameters
        ----------
        field: string
            Name of target field.
        fn: string, optional
            Path to the HDF5 snapshot.
        time: float, optional
            Target physical time, in second.
        """
        # get the data first
        self.get_centquant()

        # checks
        assert fn is not None or time is not None, "One of fn or time must be specified."
        assert field in self.centquant_field, "Available fields are: {}".format(self.centquant_field)

        # get the physical time
        if fn:
            time_target = self.get_time(fn)
        else:
            time_target = time

        # interpolation
        # --> use boundary values for outliers
        time = self.centquant["time"]
        data = self.centquant[field]

        interp_func = interp1d(time, data,
                               fill_value = (data[0], data[-1]),
                               bounds_error = False)
        value = interp_func(time_target)

        return value.tolist()

    def extend_filename(self, fn_or_idx):
        """
        Extend the filename to Data_{:06d} if a number is specified.
        """
        if   isinstance(fn_or_idx, int):
            # fn_or_idx is an integer
            fn_or_idx = "Data_{:06d}".format(fn_or_idx)

        elif isinstance(fn_or_idx, str) and fn_or_idx.isdigit():
            # fn_or_idx is an string of integer
            fn_or_idx = "Data_{:06d}".format(int(fn_or_idx))

        # add self.rundir if Data_* is not in current working directory
        if not os.path.isfile(fn_or_idx):
            fn_or_idx = os.path.join(self.rundir, fn_or_idx)

        return fn_or_idx

    def get_file_index(self, fn):
        """
        Retrieve the index from the HDF5 file.
        """
        matches = re.findall(self.Pattern_FileName, fn)

        if matches:
            return int(matches[-1])
        else:
            return None

    def get_allhdf5files(self, path = None):
        """
        Get all the HDF5 files, "Data_??????", in the specified directory.
        """
        if path is None:
            path = self.rundir

        fn_pattern = os.path.join(path, "Data_" + "[0-9]" * 6)
        fn_list    = glob(fn_pattern)

        assert fn_list, "No HDF5 files found in {}.".format(path)

        return fn_list

    def get_centquant(self):
        """
        Get the data in Record__CentralQuant and store in the self.centquant attribute.
        """
        if self.centquant is None:
            self.centquant = self._loader_centquant()

        self.centquant_field = self.centquant.dtype.names

    def get_quadmom(self):
        """
        Get the data in Record__QuadMom_2nd and store in the self.quadmom attribute.
        """
        if self.quadmom is None:
            self.quadmom = self._loader_quadmom()

    def get_param(self, keyword, fn = None, source = "note"):
        """
        Get the runtime parameters from various sources.

        Note that some runtime parameters for CCSN simulations are not recorded
        in the HDF5 snapshots. Additionally, some parameters are recorded under
        different names in the snapshots.

        Parameters
        ----------
        keyword: string
            Name of the runtime parameter as used in Input__Parameter. Case-insensitive.
        fn: string, optional
            Path to the HDF5 snapshot.
        source: string, optional
            Source from which the runtime parameter is obtained.
            --> "note"    : Record__Note
                "hdf5"    : HDF5 snapshot
                "testprob": Input__TestProb
        """
        # checks
        assert source in ["note", "hdf5", "testprob"], "Unsupported source: {}".format(source)

        if source == "hdf5":
            assert fn is not None, "No HDF5 snapshot specified."

        # convert the keyword to the upper case
        keyword_upper = keyword.upper()

        # check if it is already obtained
        if keyword_upper in self.param:
            return self.param[keyword_upper]

        if source == "hdf5":
            # retrieve the keyword in the HDF5 snapshot
            value = self._yt_get_param_hdf5(fn, keyword)
        else:
            # retrieve the keyword in Record__Note/Input__TestProb and
            # convert it to the corresponding datatype
            value = self._get_param_ascii(keyword, kind = source)

        # store the value for re-use
        self.param[keyword_upper] = value

        return value

    def get_time(self, fn, cgs = True):
        """
        Get the physical time of the specified file.

        Parameters
        ----------
        fn: string
            Path to the HDF5 snapshot.
        cgs: boolean, optional
            Flag to convert the physical time to the CGS unit.
        """
        time = self._h5py_get_phystime(fn)

        if cgs:
            self.get_unitsys(fn)
            time *= self.unit["T"]

        return time

    def get_unitsys(self, fn):
        """
        Get the unit system recorded in the specified HDF5 snapshot.
        """
        fnidx = self.get_file_index(fn)

        if self.unit_stamp != fnidx:
            self.unit       = self._h5py_get_unitsystem(fn)
            self.unit_stamp = fnidx

    def get_energyshift(self, fn = None):
        """
        Get the energy shift in the nuclear EoS Table.

        Retrieve the adopted NUC_TABLE and return the corresponding shift energy defined
        in constant.py. Currently, only LS220 and SFHo are supported.

        Parameters
        ----------
        fn: string
            Path to the HDF5 snapshot.
            If specified, retrieve the nuclear EoS table recorded in the HDF5 snapshot.
        """
        if fn:
            nuctable = self.get_param("NucTable", fn = fn, source = "hdf5")
        else:
            nuctable = self.get_param("Nuc_Table", source = "note")

        for key, value in NuEoS_EnergyShift.items():
            if key in nuctable:
                return value

    def get_pns_coord(self, fn, source = "ascii"):
        """
        Get the coordinates of the center of proto-neutron star.

        Parameters
        ----------
        fn: string
            Path to the HDF5 snapshot.
        source: string, optional
            Source from which the runtime parameter is obtained.
            --> "ascii": Record__CentralQuant.
                "hdf5" : HDF5 snapshot.
        """
        assert source in ["ascii", "hdf5"], "Unsupported source: {}".format(source)

        if source == "hdf5":
            coord = self.yt_get_coord_pns(fn)
        else:
            time  = self.get_time(fn)
            coord = [self.interp_centquant("ccsn_{}".format(d), time = time)
                     for d in "xyz"]

        return coord

    def get_center(self, fn, center):
        """
        Convert the input center to the box center or the PNS center if specified as a string.
        """
        if center == "c":
            ds     = yt.load(fn)
            center = ds.domain_center.in_cgs().tolist()
        elif center == "pns_ascii":
            center = self.get_pns_coord(fn, source = "ascii")
        elif center == "pns_hdf5":
            center = self.get_pns_coord(fn, source = "hdf5")

        # convert to yt.YTArray
        center = yt.YTArray(center, "cm")

        return center

    def get_sphave_profile(self, fn, fields, rmax, center = "c",
                           field_coord = "radius", **kwargs_prof):
        """
        Compute the spherically averaged profiles of specified fields via yt.

        Parameters
        ----------
        fn: string
            Path to the HDF5 snapshot.
        fields: string or array-like of string
            Name of target fields.
        rmax: float
            Maximum radius of the profile.
        center: string or array-like of float, optional
            Coordinate of reference center, in cm.
        field_coord: string, optional
             Name of binning field.
        kwargs_prof: dict, optional
            Additional parameters passed to the yt.create_profile() function.
        """
        ds     = yt.load(fn)
        fields = convert_sequence(fields)

        for field in fields:
            self.yt_check_field(ds, field, center = center)

        self.yt_check_field(ds, field_coord, center = center)

        # get the profile
        sphere  = ds.sphere(center, rmax)
        profile = yt.create_profile(sphere, field_coord, fields, **kwargs_prof)

        # remove empty bins and convert the data into the CGS unit
        mask_nonempty = (profile.weight != 0.0)
        data_cgs = list()

        radius = profile.x[mask_nonempty].in_cgs().value
        data_cgs.append(radius)

        for field in fields:
            # apply the mask and convert to CGS unit
            data = profile[field][mask_nonempty].in_cgs()
            data_cgs.append(data)

        return np.array(data_cgs).T
