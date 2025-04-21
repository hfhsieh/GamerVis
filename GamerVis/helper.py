#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Helper functions for various purposes.
#

__all__ = ("convert_datatype",
           "convert_datatype_numpy",
           "convert_sequence",
           "gene_headers",
           "calc_derivative"  )


import numpy as np


def convert_datatype(value):
    # Convert the input string to the corresponding datatype.
    if value.isdigit():
        return int(value)

    try:
        return float(value)
    except ValueError:
        # return as string if not a number
        return value


def convert_datatype_numpy(value):
    # Convert the numpy datatype of input value to the Python built-in datatype.
    if isinstance(value, (np.integer, int)):
        return int(value)
    elif isinstance(value, (np.floating, float)):
        return float(value)
    elif isinstance(value, bytes):
        return value.decode("utf-8")
    elif isinstance(value, str):
        return value
    elif isinstance(value, (np.ndarray, list, tuple)):
        return tuple(value)
    else:
        return value


def convert_sequence(par):
    # Check if the input parameter is a sequence, and convert it to a list if it is not.
    if isinstance(par, (int, float, str)):
        par = [par]
    elif not isinstance(par, (tuple, list, range)):
        raise TypeError("Expected a number/string or a sequence of numbers/strings")

    return par


def gene_headers(metadata, colname, colunit = None, fmt = "{:>14s}"):
    """
    Generate a string containing the given metadata and columne name in the specified formt.

    The output string is in format of
        # metadata
        # ...
        # =====================
        # colname_0  colname_1  ...  colname_n
        # colunit_0  colunit_1  ...  colunit_n

    Parameters
    ----------
    metadata: array-like of string
        Metadata shown in the header, with each entry displayed on a separate line.
    colname: array-like of string
        Name of each column.
    colunit: array-like of string, optional
        Units of the fields in each column.
    fmt: string, optional
        String format.
    """
    # add the prefix "#" to metadata
    metadata = ["# " + line  if line  else  "#"
                for line in metadata]

    # column info
    num_col = len(colname)
    colidx  = ["{:d}".format(i)  for i in range(1, num_col + 1)]
    colinfo = [colidx, colname]

    if colunit:
        colinfo += [colunit]

    for idx, info in enumerate(colinfo):
        info = "".join([fmt.format(item)  for item in info])
        info = "#" + info[1:]

        colinfo[idx] = info

    # combine all information
    header = metadata + ["# " + "=" * 50] + colinfo
    header = "\n".join(header)

    return header


def calc_derivative(xs, ys):
    """
    Compute the derivate of y = y(x) using

        forward/backward Euler difference for unevenly spaced data and first/last data
        central difference                for   evenly spaced data
    """
    assert len(xs) == len(ys), "Inconsistent size of x and y."

    size_x     = len(xs)
    x_new      = list()
    derivative = list()

    for idx in range(size_x):
        if   idx == 0:
            # forward Euler difference
            x     = 0.5 * (xs[idx] + xs[idx+1])
            slope = (ys[idx+1] - ys[idx]) / (xs[idx+1] - xs[idx])

            x_new.append(x)
            derivative.append(slope)

        elif idx == size_x - 1:
            if size_x > 2:
                # backward Euler difference
                x     = 0.5 * (xs[idx] + xs[idx-1])
                slope = (ys[idx] - ys[idx-1]) / (xs[idx] - xs[idx-1])

                x_new.append(x)
                derivative.append(slope)

        else:
            h1 = xs[idx  ] - xs[idx-1]
            h2 = xs[idx+1] - xs[idx  ]

            if h1 != h2:
                # unevenly spaced, use forward Euler difference
                x     = 0.5 * (xs[idx] + xs[idx+1])
                slope = (ys[idx+1] - ys[idx]) / h2

                x_new.append(x)
                derivative.append(slope)
            else:
                # evely spaced, use central difference
                x     = xs[idx]
                slope = (ys[idx+1] - ys[idx-1]) / (2.0 * h2)

                x_new.append(x)
                derivative.append(slope)

    return x_new, derivative
