#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Class for analyzing and visualizing GAMER data for CCSN simulations.
#

__all__ = "gamervis",


import os

import yt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import medfilt

from .io import gamer_io
from .helper import *
from .constant import *
from .lib.lib_accretion import *
from .lib.lib_lowtw import *
from .lib.lib_timescale import *
from .lib.lib_gw import *


class gamervis(gamer_io):
    """
    Class for analyzing and visualizing GAMER data for CCSN simulations.
    """
    Field_Name = {"density"                        : "Dens",
                  "radial_velocity"                : "Vrad",
                  "pns_sph_vradius"                : "Vrad",
                  "ye"                             : "Ye",
                  "Ye"                             : "YeDens",
                  "magnetic_field_spherical_radius": "Brad",
                  "magnetic_field_spherical_theta" : "Btht",
                  "magnetic_field_spherical_phi"   : "Bphi" }  # alias of field name for output ASCII files
    fmt_ascii_header = "{:>14s}"
    fmt_ascii_data   = "%14.6e"

    def __init__(self, rundir = ".", tbounce = 0.0, nuctable = None):
        """
        Parameters
        ----------
        rundir: string, optional
            Path to the simulation data.
        tbounce: float, optional
            Physical time of core bounce, in second.
        nuctable: string, optional
            Path to the nuclear EoS table.
            If set to "runtime", the value recorded in Record__Note will be used.
        """
        gamer_io.__init__(self, rundir = rundir)

        self.rundir  = rundir
        self.tbounce = tbounce
        self.eos     = None     # nuclear EoS solver

        # initialize the nuclear EoS solver if specified
        if nuctable == "runtime":
            nuctable = self.get_param("Nuc_Table", source = "note")

        if nuctable:
            from .nueos import nueos
            self.eos = nueos(nuctable)

    def generate_mask(self, x, x_lower = None, x_upper = None):
        # Constructs a boolean mask based on the specified selection criteria.
        mask = np.full(len(x), True, dtype = bool)

        if x_lower is not None:
            mask &= (x >= x_lower)

        if x_upper is not None:
            mask &= (x <= x_upper)

        return mask

    def calc_fieldmax(self, fn_list, fields,
                      center = "pns_ascii", selection_cond = None,
                      fnout = "FieldMax.txt", path_fnout = "."):
        """
        Find the maximum value of the specified fields within the domain,
        restricted by the given selection criteria.

        Parameters
        ----------
        fn_list: array-like of integer/string
            Path to HDF5 snapshots or their indices to be processed.
            --> Can be "Data_000010", "10", or 10.
                Use "all" to automatically retrieve all available HDF5 files.
        fields: array-like of string
            Name of target fields.
        center: string or array-like of float, optional
            Coordinate of reference center, in cm.
            --> "c"        : center of the simulation domain.
                "pns_ascii": PNS center recorded in Record__CentralQuant
                "pns_hdf5" : coordinate of highest-density cell in the HDF5 snapshot.
        selection_cond: string or array-like of string, optional
            A condition of list of conditions that define the region
            over which the maximum values are computed.
        fnout: string, optional
            Name the output ASCII file.
        path_fnout: string, optional
            Path to the output file.
        """
        # create the directory for output files
        if path_fnout != ".":
            os.makedirs(path_fnout, exist_ok = True)

        # setting
        if fn_list == "all":
            fn_list = self.get_allhdf5files(self.rundir)

        if selection_cond is not None:
            selection_cond = convert_sequence(selection_cond)

        fn_list = convert_sequence(fn_list)
        fields  = convert_sequence(fields)

        # main routine
        dataset = dict()

        for storage, fn in yt.parallel_objects(fn_list, storage = dataset):
            fn         = self.extend_filename(fn)
            phys_time  = self.get_time(fn, cgs = True)
            center_ref = self.get_center(fn, center)

            ds = yt.load(fn)
            ad = ds.all_data()

            if selection_cond:
                region = ad.cut_region(selection_cond)
            else:
                region = None

            data = [phys_time]

            # obtain and store the maximum value and coordinate for each field
            for idx_field, field in enumerate(fields):
                value, coord = ds.find_max(field, source = region)

                coord = coord.in_cgs() - center_ref
                value = value.in_cgs()

                data.extend(coord.tolist())
                data.append(value.to_value())

            # store the results
            storage.result_id = fn
            storage.result    = data

        if yt.is_root():
            # sort the data according to the physical time
            dataset = list(dataset.values())
            dataset.sort()

            # dump data
            metadata = ["Bounce Time     [s] : {:.6e}".format(self.tbounce),
                        "Reference Center    : {}".format(center),
                        "Selection Condtions : {}".format(selection_cond),
                        "",
                        "All quantities are in the CGS unit."]

            colname  = ["PhysTime"]

            for field in fields:
                if isinstance(field, str):
                    field_keys = field
                else:
                    field_keys = field[-1]

                field_keys = self.Field_Name.get(field_keys, field_keys)

                colname.extend(["{}{}".format(field_keys, suffix)
                                for suffix in ("_X", "_Y", "_Z", "")])

            header = gene_headers(metadata, colname, fmt = "{:>20s}")
            fnout = os.path.join(path_fnout, fnout)

            np.savetxt(fnout, dataset,
                       delimiter = "", fmt = "%20.7e", header = header, comments = "")

            print("Dump data to {}".format(fnout), flush = True)

    def calc_pns(self, fn_list, dens_thresh = 1e11, dens_frac = 0.1,
                 fnout = "PNS.txt", path_fnout = "."):
        """
        Compute the enclosed mass and radius of the proto-neutron star (PNS), where

            Mass_PNS   = Total mass with density > dens_thresh
            Radius_PNS = Averaged radius for cells with density between
                         dens_thresh * (1 ± dens_frac)

        Parameters
        ----------
        fn_list: array-like of integer/string
            Path to HDF5 snapshots or their indices to be processed.
            --> Can be "Data_000010", "10", or 10.
                Use "all" to automatically retrieve all available HDF5 files.
        dens_thresh: float, optional
            Density threshold for computing the PNS mass and radius.
        dens_frac: float, optional
            Fractional tolerance around the threshold density for computing the PNS radius.
        fnout: string, optional
            Name the output ASCII file.
        path_fnout: string, optional
            Path to the output file.
        """
        # checks
        assert dens_frac < 1.0, "Invalid density fraction: {}".format(dens_frac)

        # create the directory for output files
        if path_fnout != ".":
            os.makedirs(path_fnout, exist_ok = True)

        # setting
        if fn_list == "all":
            fn_list = self.get_allhdf5files(self.rundir)

        fn_list = convert_sequence(fn_list)

        # main routine
        dataset = dict()

        for storage, fn in yt.parallel_objects(fn_list, storage = dataset):
            fn        = self.extend_filename(fn)
            phys_time = self.get_time(fn, cgs = True)

            ds = yt.load(fn)
            ad = ds.all_data()

            # compute the PNS mass
            region    = ad.cut_region("obj['density'] > {:.6e}".format(dens_thresh))
            cell_mass = region["cell_mass"]

            if cell_mass.size:
                mass = cell_mass.sum().in_cgs().to_value()
            else:
                mass = np.nan

            # compute the PNS radius
            dens_lo = dens_thresh * (1.0 - dens_frac)
            dens_hi = dens_thresh * (1.0 + dens_frac)

            region      =     ad.cut_region("obj['density'] > {:.6e}".format(dens_lo))
            region      = region.cut_region("obj['density'] < {:.6e}".format(dens_hi))
            cell_radius = region["spherical_radius"]

            if cell_radius.size:
                radius = cell_radius.mean().in_cgs().to_value()
            else:
                radius = np.nan

            # store the results
            storage.result_id = fn
            storage.result    = phys_time, mass, radius

        if yt.is_root():
            # sort the data according to the physical time
            dataset = list(dataset.values())
            dataset.sort()

            # dump data
            metadata = ["Bounce Time           [s] : {:.6e}".format(self.tbounce),
                        "Density Threshold [g/cm3] : {:.6e}".format(dens_thresh),
                        "Density Fraction          : {:.6e}".format(dens_frac),
                        "",
                        "All quantities are in the CGS unit.",
                        "NaN indicates that no cells meet the selection criteria."]
            colname  = ["PhysTime", "PNS_Mass", "PNS_Radius"]
            colunit  = ["[s]", "[g]", "[cm]"]

            header = gene_headers(metadata, colname, colunit = colunit, fmt = self.fmt_ascii_header)
            fnout  = os.path.join(path_fnout, fnout)

            np.savetxt(fnout, dataset,
                       delimiter = "", fmt = self.fmt_ascii_data, header = header, comments = "")

            print("Dump data to {}".format(fnout), flush = True)

    def calc_profile(self, fn_list, fields = None,
                     rmax = 3.0e8, center = "pns_ascii", logscale = True, nbin = 128,
                     fnout_prefix = "Profile_SphAve", path_fnout = "."):
        """
        Compute the spherically averaged profiles of specified fields via yt,
        and dump the profiles to ASCII files.

        Parameters
        ----------
        fn_list: array-like of integer/string
            Path to HDF5 snapshots or their indices to be processed.
            --> Can be "Data_000010", "10", or 10.
                Use "all" to automatically retrieve all available HDF5 files.
        fields: array-like of string, optional
            Name of target fields.
        rmax: float, optional
            Maximum radius of the profile, in cm.
        center: string or array-like of float, optional
            Coordinate of reference center, in cm.
            --> "c"        : center of the simulation domain.
                "pns_ascii": PNS center recorded in Record__CentralQuant
                "pns_hdf5" : coordinate of highest-density cell in the HDF5 snapshot.
        logscale: boolean, optional
            Indicates whether to apply a logarithmic scale to the bin edges.
        nbin: integer, optional
            Number of bins.
        fnout_prefix: string, optional
            Prefix for the output filename.
        path_fnout: string, optional
            Path to the output file.
        """
        # checks
        assert logscale in [True, False], "Invalid logscale: {}".format(logscale)

        # create the directory for output files
        if path_fnout != ".":
            os.makedirs(path_fnout, exist_ok = True)

        # setting
        if fn_list == "all":
            fn_list = self.get_allhdf5files(self.rundir)

        fn_list = convert_sequence(fn_list)

        if fields is None:
            fields = ["density", "pns_sph_vradius", "ye", "Ye", "Temp",
                      "Pres", "Entr", "Pote"]
        else:
            fields = convert_sequence(fields)

        colname = ["Radius"] + [self.Field_Name.get(f, f)  for f in fields]

        field_coord = "pns_sph_radius"
        rmax        = yt.YTArray(rmax, "cm")
        kwargs_prof = {"units"       : {field_coord: "km"},
                       "logs"        : {field_coord: logscale},
                       "extrema"     : {field_coord: (None, rmax)},
                       "weight_field": ("gas", "cell_mass"),
                       "n_bins"      : nbin                 }

        # main routine
        for fn in fn_list:
            fn         = self.extend_filename(fn)
            phys_time  = self.get_time(fn)
            center_ref = self.get_center(fn, center)
            profile    = self.get_sphave_profile(fn, fields, rmax, center = center_ref,
                                                 field_coord = field_coord, **kwargs_prof)

            if yt.is_root():
                # dump data
                metadata = ["File                : {}".format(fn),
                            "Physical Time   [s] : {:.6e}".format(phys_time),
                            "Bounce Time     [s] : {:.6e}".format(self.tbounce),
                            "Maximum Radius [km] : {:.3f}".format(rmax.in_units("km").v),
                            "Reference Center    : {}".format(center),
                            "Number of Bin       : {}".format(nbin),
                            "Logscale            : {}".format(logscale),
                            "",
                            "All quantities are in the CGS unit."]

                header = gene_headers(metadata, colname, fmt = self.fmt_ascii_header)

                # retrieve the file index and set up the output filename
                fn_idx = self.get_file_index(fn)

                fnout = fnout_prefix + "_{:06d}.txt".format(fn_idx)
                fnout = os.path.join(path_fnout, fnout)

                np.savetxt(fnout, profile,
                           delimiter = "", fmt = self.fmt_ascii_data, header = header, comments = "")

                print("Dump data to {}".format(fnout), flush = True)

    def calc_accretion(self, fn_list, radius, method, center = "pns_ascii",
                       width = 1.0e6, logscale = False, nbin = 64,
                       fnout = "Accretion.txt", path_fnout = "."):
        """
        Compute the accretion rate at the specified radius, and dump the data to ASCII files.

        Parameters
        ----------
        fn_list: array-like of integer/string
            Path to HDF5 snapshots or their indices to be processed.
            --> Can be "Data_000010", "10", or 10.
                Use "all" to automatically retrieve all available HDF5 files.
        radius: float
            Target spherical radius, in cm.
        method: string
            Method for computing the accretion rate.
            --> "postprocess": time derivative of the mass enclosed within the specified radius.
                "shell"      : mass flux within the shell of [radius - width, radius + width].
                "profile"    : derived from density and radial velocity interpolated from
                               the spherically averaged profile at the specified radius.
        center: string or array-like of float, optional
            Coordinate of reference center, in cm.
            --> "c"        : center of the simulation domain.
                "pns_ascii": PNS center recorded in Record__CentralQuant
                "pns_hdf5" : coordinate of highest-density cell in the HDF5 snapshot.
        width: float, optional
            Half-width of the shell for the "shell" method, in cm.
        logscale: boolean, optional
            Indicates whether to apply a log scale to the bin edges, for the "profile" method.
        nbin: integer, optional
            Number of bins, for the "profile" method.
        fnout: string, optional
            Name the output ASCII file.
        path_fnout: string, optional
            Path to the output file.
        """
        # checks
        assert method in ["postprocess", "shell", "profile"], "Unsupported method {}.".format(method)

        # create the directory for output files
        if path_fnout != ".":
            os.makedirs(path_fnout, exist_ok = True)

        # setting
        if fn_list == "all":
            fn_list = self.get_allhdf5files(self.rundir)

        fn_list = convert_sequence(fn_list)

        if method == "postprocess":
            assert len(fn_list) > 1, "Insufficient specified files for the post-process method."

        # main routine
        dataset = dict()

        for storage, fn in yt.parallel_objects(fn_list, storage = dataset):
            fn         = self.extend_filename(fn)
            phys_time  = self.get_time(fn)
            center_ref = self.get_center(fn, center)

            if   method == "postprocess":
                acc_rate = calc_massenc(fn, radius = radius, center = center_ref)

            elif method == "shell":
                acc_rate = calc_accretion_shell(fn, radius = radius, width = width,
                                                center = center_ref)

            elif method == "profile":
                acc_rate = calc_accretion_profile(fn, radius = radius, center = center_ref,
                                                  logscale = logscale, nbin = nbin)

            # store the results
            storage.result_id = fn
            storage.result    = phys_time, acc_rate

        if yt.is_root():
            # sort the data according to the physical time
            dataset = list(dataset.values())
            dataset.sort()

            # compute the time derivative of the enclosed mass for the post-process method
            if method == "postprocess":
                dataset = list(zip(*dataset))
                dataset = calc_derivative(*dataset)
                dataset = list(zip(*dataset))

            # dump data
            metadata = ["Method                : {}".format(method),
                        "Reference Center      : {}".format(center),
                        "Radius           [cm] : {:.6e}".format(radius)]

            if method == "shell":
                metadata += ["Shell Half-Width [cm] : {:.6e}".format(width)]

            if method == "profile":
                metadata += ["Log Scale             : {}".format(logscale),
                             "Number of Bin         : {}".format(nbin)]

            metadata += ["",
                         "All quantities are in the CGS unit."]
            colname   = ["PhysTime", "AccRate"]
            colunit   = ["[s]", "[g/s]"]

            header = gene_headers(metadata, colname, colunit = colunit, fmt = self.fmt_ascii_header)
            fnout  = os.path.join(path_fnout, fnout)

            np.savetxt(fnout, np.array(dataset),
                       delimiter = "", fmt = self.fmt_ascii_data, header = header, comments = "")

            print("Dump data to {}".format(fnout), flush = True)

    def calc_timescale_shock(self, fn_list, center = "pns_ascii", logscale = False, nbin = 128,
                             fnout = "Timescale_ShockExp.txt", path_fnout = "."):
        """
        Compute the heating and advection timescale at various radii, along with
        the shock expansion onset condition (the ratio of advection to heating timescales).

        Parameters
        ----------
        fn_list: array-like of integer/string
            Path to HDF5 snapshots or their indices to be processed.
            --> Can be "Data_000010", "10", or 10.
                Use "all" to automatically retrieve all available HDF5 files.
        center: string or array-like of float, optional
            Coordinate of reference center, in cm.
            --> "c"        : center of the simulation domain.
                "pns_ascii": PNS center recorded in Record__CentralQuant
                "pns_hdf5" : coordinate of highest-density cell in the HDF5 snapshot.
        logscale: boolean, optional
            Indicates whether to apply a log scale to the bin edges, for the "profile" method.
        nbin: integer, optional
            Number of bins used to construct spherically averaged profiles.
        fnout: string, optional
            Name the output ASCII file.
        path_fnout: string, optional
            Path to the output file.
        """
        # create the directory for output files
        if path_fnout != ".":
            os.makedirs(path_fnout, exist_ok = True)

        # setting
        if fn_list == "all":
            fn_list = self.get_allhdf5files(self.rundir)

        fn_list = convert_sequence(fn_list)

        rad_field = ["rsh_min", "rsh_ave_V", "rsh_ave_Vinv", "rsh_max"]
        rad_unit  = ["(Rsh_Min)", "(Rsh_V)", "(Rsh_Vinv)", "(Rsh_Max)"]

        if self.get_param("NEUTRINO_SCHEME") == "LEAKAGE":
            rad_field += ["leak_radns_nue", "leak_radns_nua", "leak_radns_nux"]
            rad_unit  += ["(RadNS_Nue)", "(RadNS_Nua)", "(RadNS_Nux)"]

        rad_size = len(rad_field)

        # main routine
        dataset = dict()

        for storage, fn in yt.parallel_objects(fn_list, storage = dataset):
            fn         = self.extend_filename(fn)
            phys_time  = self.get_time(fn)
            center_ref = self.get_center(fn, center)

            # estimate the advection timescale at various definitions of shock radius
            # and at the radius of neutrino sphere
            radius_list = [self.interp_centquant(field, time = phys_time)
                           for field in rad_field]

            dataset_heat = calc_timescale_heating(fn)  # Eth, dEdt, tau_heat
            dataset_adv  = calc_timescale_advection(fn, radius = radius_list,
                                                    center = center_ref, logscale = logscale,
                                                    nbin = nbin)  # Mass, dM, tau_adv
            cond_shkexp  = [tau_adv / dataset_heat[2]
                            for tau_adv in dataset_adv[-rad_size:]]

            # store the results
            storage.result_id = fn
            storage.result    = phys_time, *dataset_heat, *dataset_adv, *cond_shkexp

        if yt.is_root():
            # sort the data according to the physical time
            dataset = list(dataset.values())
            dataset.sort()

            # dump data
            metadata = ["Bounce Time  [s] : {:.6e}".format(self.tbounce),
                        "Reference Center : {}".format(center),
                        "Log Scale        : {}".format(logscale),
                        "Number of Bin    : {}".format(nbin),
                        "",
                        "All quantities are in the CGS unit."]
            colname  = ["PhysTime", "Eth", "dEth_dt", "tau_heat"] \
                     + ["Mass_gain"] \
                     + ["dM_gain"] * rad_size \
                     + ["tau_adv"] * rad_size \
                     + ["tau_adv/tau_heat"] * rad_size
            colunit  = ["[s]", "[erg]", "[erg/s]","[s]"] \
                     + ["[g]"] \
                     + ["{} [g/s]".format(f)  for f in rad_unit] \
                     + ["{} [s]".format(f)    for f in rad_unit] \
                     + ["{} [1]".format(f)    for f in rad_unit]

            header = gene_headers(metadata, colname, colunit, fmt = "{:>20s}")
            fnout  = os.path.join(path_fnout, fnout)

            np.savetxt(fnout, np.array(dataset),
                       delimiter = "", fmt = "%20.7e", header = header, comments = "")

            print("Dump data to {}".format(fnout), flush = True)

    def plot_centquant(self, field, tbounce = None, weight = "V", axes = None,
                       savefig = False, fnout_prefix = "CCSN", path_fnout = ".", **kwargs_plt):
        """
        Visualize the evolution of quantities recorded in Record__CentralQuant,
        where time is displayed in milliseconds.

        Parameters
        ----------
        field: string
            Name of target field.
            --> "dens"  : peak density.
                "ye"    : electron fraction associated with the highest-density cell.
                "rsh"   : minimum, average, and maximum shock radius.
                "lum_nu": neutrino luminosity
        tbounce: float, optional
            Physical time of core bounce, in second.
        weight: string, optional
            Weighting method of the mean shock radius to be displayed.
            --> "V"   : cell volume.
                "Vinv": inverse cell volume.
        axes: Matplotlib axes object, optional
            Axes object for plotting the data.
        savefig: boolean, optional
            Flag indicating whether to save the figure.
        fnout_prefix: string, optional
            Prefix for the output filename.
        path_fnout: string, optional
            Path to the output file.
        kwargs_plt: dict, optional
            Additional parameters passed to the plt.plot() function.
            --> "xlim": range for the x coordinate, in second.
        """
        # checks
        assert field in ["dens", "ye", "rsh", "lum_nu"], "Unsupported field: {}".format(field)

        if field == "rsh":
            assert weight in ["V", "Vinv"], "Weigth {} is not supported for the mean shock radius!".format(weight)

        if field == "lum_nu":
            assert self.get_param("NEUTRINO_SCHEME") == "LEAKAGE", "No data for Neutrino luminosity."

        if tbounce == "auto":
            tbounce = self.tbounce

        # create the directory for output files
        if path_fnout != ".":
            os.makedirs(path_fnout, exist_ok = True)

        # load the data if it has not been loaded yet
        self.get_centquant()

        # get the data and set the label
        if tbounce:
            time   = self.centquant["time"] - tbounce  # in second
            xlabel = "Time after Bounce [ms]"

        else:
            time   = self.centquant["time"]  # in second
            xlabel = "Time [ms]"


        if   field == "dens":
            # peak density
            data   = self.centquant["dens"] / 1.0e14  # in 10^14 g/cm3
            ylabel = r"Central Density [$10^{14}$ g cm$^{-3}$]"

        elif field == "ye":
            # electron fraction associated with the highest-density cell
            data   = self.centquant["ye"]  # dimensionless
            ylabel = r"Electron Fraction"

        elif field == "rsh":
            # shock radius
            data   = [self.centquant["rsh_min"  ],
                      self.centquant["rsh_ave_V"]  if weight == "V"  else self.centquant["rsh_ave_Vinv"],
                      self.centquant["rsh_max"  ]]
            data   = [i / km2cm  for i in data]  # in km
            ylabel = r"Shock Radius [km]"

        elif field == "lum_nu":
            # neutrino luminosity
            data   = [self.centquant["leak_lum_nue"],
                      self.centquant["leak_lum_nua"],
                      self.centquant["leak_lum_nux"] / 4.0]
            ylabel = r"Neutrino Luminosity [erg/s]"

        # trim the data if xlim presents in the kwargs_plt
        if "xlim" in kwargs_plt:
            xlim = kwargs_plt.pop("xlim")

            # construct the mask
            mask = self.generate_mask(time, *xlim)

            # apply the mask
            time = time[mask]

            if isinstance(data, list):
                data = [d[mask]  for d in data]
            else:
                data = data[mask]

        # visualization
        if axes is None:
            fig, axes = plt.subplots()

        # remove conflict parameters in the input kwargs_plt
        if field in ["rsh", "lum_nu"]:
            if "label" in kwargs_plt:
                kwargs_plt.pop("label")

            if "ls" in kwargs_plt:
                kwargs_plt.pop("ls")

        if field == "rsh":
            line, = axes.plot(time * sec2ms, data[0], ls = "dotted", **kwargs_plt)

            # ensure the display data uses a consistent color
            line_color = line.get_color()

            axes.plot(time * sec2ms, data[1], ls = "solid",  c = line_color, **kwargs_plt)
            axes.plot(time * sec2ms, data[2], ls = "dashed", c = line_color, **kwargs_plt)

            # add legend
            lines  = axes.get_lines()
            labels = ["Min",
                      "Mean (Weight = V)" if weight == "V"  else "Mean (Weight = 1/V)",
                      "Max"]

            legend = axes.legend(lines[:3], labels, loc = "best", framealpha = 0)
            axes.add_artist(legend)

        elif field == "lum_nu":
            line, = axes.plot(time * sec2ms, data[0], ls = "solid", **kwargs_plt)

            # ensure the display data uses a consistent color
            line_color = line.get_color()

            axes.plot(time * sec2ms, data[1], ls = "dashed",  c = line_color, **kwargs_plt)
            axes.plot(time * sec2ms, data[2], ls = "dotted", c = line_color, **kwargs_plt)

            # add legend
            lines  = axes.get_lines()
            labels = [r"$\nu_\mathrm{e}$", r"$\nu_\mathrm{a}$", r"$\nu_\mathrm{x}$",]
            legend = axes.legend(lines[:3], labels, loc = "best", framealpha = 0)
            axes.add_artist(legend)

        else:
            axes.plot(time * sec2ms, data, **kwargs_plt)

        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)

        if savefig:
            fnout_field = "".join(part.capitalize()
                                  for part in field.split("_"))
            fnout = "{}_{}.png".format(fnout_prefix, fnout_field)
            fnout = os.path.join(path_fnout, fnout)

            plt.savefig(fnout, dpi = 300, bbox_inches = "tight")
            print("Save figure to {}".format(fnout), flush = True)

    def plot_gw(self, figtype, phi, theta, gw_mode = "plus", dist = 10.0,
                fs = None, time_window = None, time_window_ref = "bounce",
                kwargs_strain = None, kwargs_spect = None, kwargs_asd = None, kwargs_plt = None,
                axes = None, savefig = False, fnout_prefix = "GW", path_fnout = "."):
        """
        Visualize different characteristics of GW emissions,
        including GW strains, spectrograms, and amplitude spectral density.

        Parameters
        ----------
        figtype: string
            Type of the generated figure.
            --> "strain"     : GW strains.
                "spectrogram": spectrogram of GW emissions.
                "asd"        : amplitude spectral density.
        phi: float
            Azimuthal angle of the observations, in radians.
        theta: float
            Polar angle of the observations, in radians.
        gw_mode: string, optional
            Component of GW emissions to be displayed.
            --> "plus" : plus mode.
                "cross": cross mode.
                "both" : L2 norm of the plus and cross modes.
        dist: float, optional
            Distance between the source and the observers, in kpc.
        fs: float, optional
            Target sampling frequency, in Hz.
        time_window: array-like of float, optional
            Beginning and end of time window to keep, in second.
        time_window_ref: float or string, optional
            Reference time origin used to shift the time specified in time_window.
        kwargs_strain: dict, optional
            Additional parameters used to configure the strain visualization.
            --> "scaling_power": rescale the strain by a factor of 10^scaling_power.
        kwargs_spect: dict, optional
            Additional parameters used to configure the spectrogram computation.
            --> "fmin" : lower bound of the frequency range (y-axis) shown in the figure.
                "fmax" : upper bound of the frequency range (y-axis) shown in the figure.
                "vmin" : lower bound of the color bar shown in the figure, in log scale.
                "vmax" : upper bound of the color bar shown in the figure, in log scale.
                Also accepts any parameters supported by lib_gw.calc_spectrogram().
        kwargs_asd: dict, optional
            Additional parameters used to configure the amplitude spectral density computation.
            --> "num_ave": the window size used in scipy.signal.medfilt() for median filtering.
                "fmin"   : lower bound of the frequency range (x-axis) shown in the figure.
                "fmax"   : upper bound of the frequency range (x-axis) shown in the figure.
                Also accepts any parameters supported by lib_gw.calc_spectrum().
        kwargs_plt: dict, optional
            Additional parameters passed to the visualization function.
        axes: Matplotlib axes object, optional
            Axes object for plotting the data.
        savefig: boolean, optional
            Flag indicating whether to save the figure.
        fnout_prefix: string, optional
            Prefix for the output filename.
        path_fnout: string, optional
            Path to the output file.
        """
        # checks
        assert figtype in ["strain", "spectrogram", "asd"], "Unsupported figtype: {}".format(figtype)
        assert gw_mode in ["plus", "cross", "both"], "Unsupported gw_mode: {}".format(gw_mode)
        assert isinstance(time_window_ref, float) or \
               time_window_ref in [None, "bounce"], "Invalid time_window_ref."

        # create the directory for output files
        if path_fnout != ".":
            os.makedirs(path_fnout, exist_ok = True)

        # get the GW strains
        self.get_quadmom()

        A_plus, A_cross = calc_amplitude(self.quadmom["Ixx"], self.quadmom["Ixy"],
                                         self.quadmom["Ixz"], self.quadmom["Iyy"],
                                         self.quadmom["Iyz"], self.quadmom["Izz"],
                                         phi, theta)

        if gw_mode == "plus":
            A_cross      = None
            label_strain = r"$h_{\plus}$"
        elif gw_mode == "cross":
            A_plus       = None
            label_strain = r"$h_{\times}$"
        else:
            label_strain = r"$h$"

        gw_time   = self.quadmom["time"]
        gw_strain = calc_strain(A_plus = A_plus, A_cross = A_cross, dist = dist)

        # retrieve the sampling frequency from Input__TestProb if not specified
        if fs is None:
            dt_sample = self.get_param("CCSN_GW_DT", source = "testprob") \
                      * self.get_param("UNIT_T",     source = "note")  # in second

            fs = 1.0 / dt_sample

        # get the time window for displaying the data
        if time_window is None:
            time_window = gw_time[0], gw_time[-1]
        else:
            # set up the time_window_ref first
            if time_window_ref == "bounce":
                time_window_ref = self.tbounce
            elif time_window_ref is None:
                time_window_ref = 0.0

            # update each entry if not set
            # --> note that we subtract the shift "time_window_ref" here
            if time_window[0] is None:
                time_window = gw_time[0] - time_window_ref, time_window[1]
            if time_window[1] is None:
                time_window = time_window[0], gw_time[-1] - time_window_ref

            # now shifts the time window according to the reference point
            time_window = (time_window[0] + time_window_ref,
                           time_window[1] + time_window_ref )

        # set up the corresponding keyword parameters
        if kwargs_strain is None:
            kwargs_strain = dict()

        if kwargs_spect is None:
            kwargs_spect = dict()

        if kwargs_asd is None:
            kwargs_asd = dict()

        if kwargs_plt is None:
            kwargs_plt = dict()

        # main routine
        # --> always display the time as values relative to bounce, in millisecond
        if axes is None:
            fig, axes = plt.subplots()

        if figtype == "strain":
            # strain
            scaling_power = kwargs_strain.get("scaling_power", 21.0)

            if scaling_power != 1:
                # only display the integer part of the scaling power
                ylabel = r"$10^{" + "{}".format(int(scaling_power)) + r"}$" \
                       + label_strain

            time = (gw_time - self.tbounce) * sec2ms  # in ms
            data = gw_strain * 10**scaling_power
            xlim = [(time_bdry - self.tbounce) * sec2ms  for time_bdry in time_window]

            axes.plot(time, data, **kwargs_plt)
            axes.axhline(0.0, c = "k", ls = "dashed", zorder = -1)
            axes.set_xlabel("Time after Bounce [ms]")
            axes.set_ylabel(ylabel)
            axes.set_xlim(*xlim)

            if savefig:
                fnout = "{}_Strain.png".format(fnout_prefix)

        elif figtype == "spectrogram":
            # spectrogram
            method = kwargs_spect.pop("method", "wavelet")
            fmin   = kwargs_spect.pop("fmin", 100)
            fmax   = kwargs_spect.pop("fmax", None)
            vmin   = kwargs_spect.pop("vmin", None)
            vmax   = kwargs_spect.pop("vmax", None)

            if method == "wavelet" and "dj" not in kwargs_spect:
                kwargs_spect["dj"] = 0.01

            time, freq, spec = calc_spectrogram(gw_time, gw_strain,
                                                fs = fs, time_window = time_window,
                                                method = method, **kwargs_spect)

            # customize the output for display
            mask = self.generate_mask(freq, fmin, fmax)
            freq = freq[mask]
            spec = spec[mask, :]

            time     = (time - self.tbounce) * sec2ms
            spec_log = np.log10(spec)

            vmin = spec_log.min()  if vmin is None  else  np.log10(vmin)
            vmax = spec_log.max()  if vmax is None  else  np.log10(vmax)

            # plot
            img = axes.pcolormesh(time, freq, spec_log,
                                  vmin = vmin, vmax = vmax, rasterized = True, **kwargs_plt)
            axes.set_xlabel("Time after Bounce [ms]")
            axes.set_ylabel("Frequency [Hz]")

            # add color bar
            plt.subplots_adjust(bottom = 0.24)

            pos_y   = 0.12
            height  = 0.02
            pos     = axes.get_position()
            pos_cb  = pos.x0, pos_y, pos.width, height
            axes_cb = axes.figure.add_axes(pos_cb)

            cb = axes.figure.colorbar(img, cax = axes_cb, orientation = "horizontal",
                                      ticks = range(int(vmin), int(vmax) + 1))
            cb.set_label(r"$\log_{10}($" + label_strain + r"$)$", labelpad = 4)

            if savefig:
                fnout = "{}_Spectrogram.png".format(fnout_prefix)

        else:
            # amplitude spectral density
            num_ave = kwargs_asd.pop("num_ave", 1)
            fmin    = kwargs_asd.pop("fmin", 100)
            fmax    = kwargs_asd.pop("fmax", None)

            freq, asd = calc_spectrum(gw_time, gw_strain,
                                      fs = fs, time_window = time_window,
                                      **kwargs_asd)

            # customize the output for display
            if num_ave > 1:
                asd = medfilt(asd, num_ave)

            mask = self.generate_mask(freq, fmin, fmax)
            freq = freq[mask]
            asd  = asd [mask]

            # plot
            axes.semilogy(freq, asd, **kwargs_plt)
            axes.set_xlabel("Frequency [Hz]")
            axes.set_ylabel(r"Amplitude Spectral Density [Hz$^{-1/2}$]")

            if savefig:
                fnout = "{}_ASD.png".format(fnout_prefix)

        if savefig:
            fnout = os.path.join(path_fnout, fnout)

            plt.savefig(fnout, dpi = 300, bbox_inches = "tight")
            print("Save figure to {}".format(fnout), flush = True)

    def plot_slice(self, fn_list, fields, direction, center = "pns_ascii", tbounce = None, tstart = None,
                   lim = None, logscale = None, show_grid = False, auto_scale = False,
                   zoom = 1.0, zoom_max = None, zoom_auto = False, zoom_auto_runmin = False,
                   roi_cond = None, fontsize = None,
                   fnout_prefix = "SlicePlot", path_fnout = ".", **kwargs):
        """
        Create a slice plot of the simulation data from HDF5 snapshots,
        wrapping the yt.SlicePlot() function.

        Parameters
        ----------
        fn_list: array-like of integer/string
            Path to HDF5 snapshots or their indices to be processed.
            --> Can be "Data_000010", "10", or 10.
                Use "all" to automatically retrieve all available HDF5 files.
        fields: string or array-like of string
            Name of target fields.
        direction: string
            Direction of the slice plots.
        center: string or array-like of float, optional
            Coordinate of reference center, in cm.
            --> "c"        : center of the simulation domain.
                "pns_ascii": PNS center recorded in Record__CentralQuant
                "pns_hdf5" : coordinate of highest-density cell in the HDF5 snapshot.
        tbounce: float, optional
            Physical time of core bounce, in second.
        tstart: float, optional
            Skip HDF snapshots with a physical time earlier than the specified value, in second.
        lim: array-like or a dict mapping fields to array-like, optional
            Specify the range displayed on the color bar for each field.
        logscale: boolean or a dict mapping fields to boolean, optional
            Indicates whether to apply a logarithmic scale on the color bar for each field.
        show_grid: boolean, optional
            Indicates whether to display the grid.
        auto_scale: boolean, optional
            Indicate whether to extend the color bar range for a series of slice plots.
        zoom: float, optional
            Zoom factor.
        zoom_max: float, optional
            Maximum zoom factor.
        zoom_auto: boolean, optional
            Indicate whether to update the zoom factor based on the maximum shock radius
            for a series of slice plots.
        zoom_auto_runmin: boolean, optional
            Indicate whether to use a running minimum zoom when zoom_auto is enabled.
        roi_cond: string or array-like of string, optional
            A list of conditions used to define the region of interest.
            Cells that do not match all conditions will be treated as background,
            which will be displayed in white if specified.
        fontsize: float, optional
            Font size for the displayed figure.
        fnout_prefix: string, optional
            Prefix for the output filename.
        path_fnout: string, optional
            Path to the output file.
        """
        # create the directory for output files
        if path_fnout != ".":
            os.makedirs(path_fnout, exist_ok = True)

        # setting
        if fn_list == "all":
            fn_list = self.get_allhdf5files(self.rundir)

        fn_list = convert_sequence(fn_list)
        fields  = convert_sequence(fields)

        if roi_cond is not None:
            roi_cond = convert_sequence(roi_cond)

        if   lim is None:
            lim = dict( [(f, None)  for f in fields] )
        elif isinstance(lim, (tuple, list)):
            lim = dict( [(f, lim)   for f in fields] )

        if   logscale is None:
            logscale = dict( [(f, False)     for f in fields] )
        elif logscale in [True, False, None]:
            logscale = dict( [(f, logscale)  for f in fields] )

        assert isinstance(lim,      dict), "lim must be None, a dict, or a tuple/list."
        assert isinstance(logscale, dict), "logscale must be None, a dict, or a Boolean value."

        if tbounce == "auto":
            tbounce = self.tbounce

        if tstart == "bounce":
            tstart = self.tbounce

        zoom_used = list()

        # main routine
        for fn in fn_list:
            # preparation
            fn         = self.extend_filename(fn)
            fn_idx     = self.get_file_index(fn)
            phys_time  = self.get_time(fn, cgs = True)
            center_ref = self.get_center(fn, center)

            if tstart and phys_time < tstart:
                print("Skip {}".format(fn), flush = True)
                continue
            else:
                print("Processing {}".format(fn), flush = True)

            if tbounce:
                time  = (phys_time - tbounce) * sec2ms
                title = r"$t_\mathrm{pb}$ = " + "{:7.2f} ms".format(time)
            else:
                time  = phys_time * sec2ms
                title = "Time = {:7.2f} ms".format(time)

            # update the zoom factor automatically
            if zoom_auto:
                zoom_stride = 4  # minimum stride for changing the zoom factor

                rsh_max = self.interp_centquant("rsh_max", time = phys_time)
                boxsize_half = 0.5 * self.get_param("BOX_SIZE") \
                             * self.get_param("UNIT_L")

                if rsh_max != 0.0:
                    zoom = int(boxsize_half / rsh_max / zoom_stride) * zoom_stride

                if zoom_max:
                    zoom = min(zoom, zoom_max)

                # use the a running minimum with a window size of 3
                if zoom_auto_runmin and zoom_used:
                    zoom = min(zoom, min(zoom_used[-2:]))

                print("Update zoom factor to {}".format(zoom), flush = True)

                # store the used zoom factor
                zoom_used.append(zoom)

            ds = yt.load(fn)

            for f in fields:
                cb_lim = lim.get(f, None)
                cb_log = logscale.get(f, None)

                fnout = fnout_prefix + "_{}_{}_{:06d}.png".format(f, direction, fn_idx)
                fnout = os.path.join(path_fnout, fnout)

                slc = self.yt_sliceplot(ds, field = f, direction = direction, center = center_ref, eos = self.eos,
                                        cb_lim = cb_lim, cb_logscale = cb_log, show_grid = show_grid, zoom = zoom,
                                        title = title, roi_cond = roi_cond, fontsize = fontsize)

                # update the color bar range automatically
                if auto_scale:
                    field_disp = next(iter(slc.frb.data))
                    data_disp  = slc.frb.data[field_disp].v

                    if cb_lim is None:
                        data_min = np.nanmin(data_disp)
                        data_max = np.nanmax(data_disp)
                        cb_lim   = float(data_min), float(data_max)
                    else:
                        # non-decreasing range
                        data_max = np.nanmax(data_disp)
                        print(data_max)

                        if cb_lim[1] is None:
                            cb_lim = cb_lim[0], data_max
                        else:
                            cb_lim = cb_lim[0], max( float(data_max), cb_lim[1] )

                    # update the displayed color bar again
                    slc.set_zlim(f, *cb_lim)

                    # store the new color bar range
                    lim[f] = cb_lim

                    print("Update color bar range of {} to {}".format(f, cb_lim), flush = True)

                # miscellaneous
                if "cmap" in kwargs:
                    slc.set_cmap(f, kwargs["cmap"])

                # save figure
                slc.save(fnout)

    def plot_densfluct(self, fn_list, radius, center = "pns_ascii",
                       tbounce = None, tstart = "bounce",
                       resolution = 1024, nbin = 128, vlim = None,
                       fnout_prefix = "SlicePlot_DensFluct", path_fnout = "."):
        """
        Create a slice plot of density fluctuation for the CCSN simulations.

        Parameters
        ----------
        fn_list: array-like of integer/string
            Path to HDF5 snapshots or their indices to be processed.
            --> Can be "Data_000010", "10", or 10.
                Use "all" to automatically retrieve all available HDF5 files.
        radius: float
            Half-width of the domain, in cm.
        center: string or array-like of float, optional
            Coordinate of reference center, in cm.
            --> "c"        : center of the simulation domain.
                "pns_ascii": PNS center recorded in Record__CentralQuant
                "pns_hdf5" : coordinate of highest-density cell in the HDF5 snapshot.
        tbounce: float, optional
            Physical time of core bounce, in second. Use to control the figure title.
        tstart: float, optional
            Skip HDF snapshots with a physical time earlier than the specified value, in second.
        resolution: integer, optional
            Number of pixel in each direction.
        nbin: integer, optional
            Number of bins for constructing the azimuthally averaged density.
        vlim: array-like of float, optional
            Specify the range displayed on the color bar.
        fnout_prefix: string, optional
            Prefix for the output filename.
        path_fnout: string, optional
            Path to the output file.
        """
        # create the directory for output files
        if path_fnout != ".":
            os.makedirs(path_fnout, exist_ok = True)

        # setting
        if fn_list == "all":
            fn_list = self.get_allhdf5files(self.rundir)

        fn_list = convert_sequence(fn_list)

        if tbounce == "auto":
            tbounce = self.tbounce

        if tstart == "bounce":
            tstart = self.tbounce

        # main routine
        for fn in fn_list:
            # preparation
            fn         = self.extend_filename(fn)
            fn_idx     = self.get_file_index(fn)
            phys_time  = self.get_time(fn, cgs = True)
            center_ref = self.get_center(fn, center)

            xylim = -radius / km2cm, radius / km2cm  # in km

            if tstart and phys_time < tstart:
                print("Skip {}".format(fn), flush = True)
                continue
            else:
                print("Processing {}".format(fn), flush = True)

            if tbounce:
                time  = (phys_time - tbounce) * sec2ms
                title = r"$t_\mathrm{pb}$ = " + "{:7.2f} ms".format(time)
            else:
                time  = phys_time * sec2ms
                title = "Time = {:7.2f} ms".format(time)

            # compute the density fluctuation
            # --> note that the returned coord_x and coord_y are larger than the specified radius
            coord_x, coord_y, densfluct = calc_densfluct(fn, radius = radius, center = center_ref,
                                                         resolution = resolution, nbin = nbin)
            coord_x_km = coord_x / km2cm  # in km
            coord_y_km = coord_y / km2cm  # in km

            # set up the range and the extend parameter for the color bar
            mask = (coord_x > -1.1 * radius) & (coord_x  < 1.1 * radius)
            densfluct_min = densfluct[mask].min()
            densfluct_max = densfluct[mask].max()

            if vlim is None:
                vlim = densfluct_min, densfluct_max
            else:
                if vlim[0] is None:
                    vlim = densfluct_min, vlim[1]
                if vlim[1] is None:
                    vlim = vlim[0], densfluct_max

            if vlim[0] > densfluct_min:
                if vlim[1] < densfluct_max:
                    cb_extend = "both"
                else:
                    cb_extend = "min"
            else:
                if vlim[1] < densfluct_max:
                    cb_extend = "max"
                else:
                    cb_extend = "neither"

            # visualization
            fig, ax = plt.subplots()

            norm = mcolors.TwoSlopeNorm(vmin = vlim[0], vcenter = 0, vmax = vlim[1])
            img = ax.pcolormesh(coord_x_km, coord_y_km, densfluct,
                                shading = "gouraud", cmap = "bwr",
                                norm = norm, rasterized = True)

            ax.set_xlabel("X [km]")
            ax.set_ylabel("Y [km]")
            ax.set_title(title)
            ax.set_aspect("equal", "box")
            ax.set_xlim(*xylim)
            ax.set_ylim(*xylim)

            cb = plt.colorbar(img, extend = cb_extend)
            cb.set_label(r"$(\rho - \bar{\rho}) / \bar{\rho}$", labelpad = 8)

            fig.tight_layout()

            fnout = fnout_prefix + "_{:06d}.png".format(fn_idx)
            fnout = os.path.join(path_fnout, fnout)

            plt.savefig(fnout, dpi = 300, bbox_inches = "tight")
            plt.close(fig)

            print("Save figure to {}".format(fnout), flush = True)
