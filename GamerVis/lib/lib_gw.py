#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Library for GW astronomy, including functions for
#       - compute the GW amplitude and strain
#       - compute the spectrogram of GW emissions
#       - compute the amplitude density spectrum (ASD) of GW emissions
#
#  Reference:
#    Scheidegger et al. 2008, A&A, 490, 231
#    Moore et al. 2015, Class. Quantum Grav., 32, 015014
#    Andresen et al. 2017, MNRAS, 468, 2032
#

__all__ = ["resampling",
           "calc_amplitude",
           "calc_strain",
           "calc_spectrum",
           "calc_spectrogram",
           "calc_significance"]


import numpy as np
import pycwt as wavelet
from scipy.fftpack import fft
from scipy.interpolate import interp1d, splrep, splev
from scipy.signal import stft, periodogram

from ..constant import *


def resampling(time, data, fs = 8192.0, time_window = None,
               method = "Bspline"):
    """
    Resample the given time-series data to the specified sampling frequency.

    Parameters
    ----------
    time: array-like of float
        Time stamp of each data, in second.
    data: array-like of float
        Measurements at each time stamp.
    fs: float, optional
        Target sampling frequency, in Hz.
    time_window: array-like of float, optional
        Beginning and end of time window to keep, in second.
    method: string, optional
        Interpolation scheme for resampling the time-series data.
        --> Supports "linear" and "Bspline".
    """
    assert method in ["linear", "Bspline"], "Unsupported sampling method: {}".format(method)
    assert len(time) == len(data), "Mismatch between the length of time and data arrays."
    assert np.all(np.diff(time) > 0.0), "The time stamp must increase monotonically."

    # set up the time window
    if time_window is None:
        time_window = [None, None]

    # convert to a list for manipulation
    time_window = list(time_window)

    if time_window[0] is None:
        time_window[0] = time[0]
    else:
        time_window[0] = max(time_window[0], time[0])

    if time_window[1] is None:
        time_window[1] = time[-1]
    else:
        time_window[1] = min(time_window[1], time[-1])

    # generate the time stamp of new time-series data
    time_sample = np.arange(*time_window, 1.0 / fs)

    # sample the data at the specified time stamp
    if method == "linear":
        interp_func = interp1d(time, data)
        data_sample = interp_func(time_sample)
    else:
        # note that the B-spline method returns NaN if there are duplicants
        interp_func = splrep(time, data)
        data_sample = splev(time_sample, interp_func)

    return time_sample, data_sample


def calc_amplitude(Ixx, Ixy, Ixz, Iyy, Iyz, Izz, phi, theta):
    """
    Compute the GW amplitudes, A_plus and A_cross, from the second-order time derivative
    of the mass quadrupole moment in Cartesian coordinate.

    Parameters
    ----------
    Ixx: array-like of float
        xx component of the second-order time derivative of the mass quadrupole moment.
    Ixy: array-like of float
        xy component of the second-order time derivative of the mass quadrupole moment.
    Ixz: array-like of float
        xz component of the second-order time derivative of the mass quadrupole moment.
    Iyy: array-like of float
        yy component of the second-order time derivative of the mass quadrupole moment.
    Iyz: array-like of float
        yz component of the second-order time derivative of the mass quadrupole moment.
    Izz: array-like of float
        zz component of the second-order time derivative of the mass quadrupole moment.
    phi: float
        Azimuthal angle of the observations, in radians.
    theta: float
        Polar angle of the observations, in radians.
    """
    # convert the mass quadrupole moment from Cartesian coordinates to spherical coordinates
    sin_phi   = np.sin(phi)
    cos_phi   = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    Ipp = Ixx * sin_phi**2 + Iyy * cos_phi**2 - 2 * Ixy * sin_phi * cos_phi

    Itp = (Iyy - Ixx) * cos_theta * sin_phi * cos_phi \
        + Ixy * cos_theta * (cos_phi**2 - sin_phi**2) \
        + Ixz * sin_theta * sin_phi \
        - Iyz * sin_theta * cos_phi

    Itt = (Ixx * cos_phi**2 + Iyy * sin_phi**2 + 2 * Ixy * sin_phi * cos_phi) * cos_theta**2 \
        + Izz * sin_theta**2 \
        - 2 * (Ixz * cos_phi + Iyz * sin_phi) * sin_theta * cos_theta

    # compute the plus and cross components of the amplitude
    A_plus  = Itt - Ipp
    A_cross = 2 * Itp

    return A_plus, A_cross


def calc_strain(A_plus, A_cross, dist = 10.0):
    """
    Compute the GW strain via

        h^{TT}_{ij} = sqrt( A_plus^2 + A_cross^2 ) / dist

    where we ignore the polarization response functions.

    Parameters
    ----------
    A_plus: array-like of float
        Amplitude of the plus mode of GW emissions.
    A_cross: array-like of float
        Amplitude of the cross mode of GW emissions.
    dist: float, optional
        Distance between the source and the observers, in kpc.
    """
    factor = 1.0 / (dist * kpc2cm)

    if A_plus is None:
        return factor * A_cross
    elif A_cross is None:
        return factor * A_plus
    else:
        return factor * np.hypot(A_plus, A_cross)


def calc_spectrum(time, strain, fs = 8192.0, time_window = None,
                  method = "periodogram", detrend = "linear", **kwargs_periodogram):
    """
    Compute the amplitude spectral density (ASD) of the GW characteristic strain.

    Parameters
    ----------
    time: array-like of float
        Time stamp of each data, in second.
    strain: array-like of float
        Measurement of strain at each time stamp.
    fs: float, optional
        Target sampling frequency, in Hz.
    time_window: array-like of float, optional
        Beginning and end of time window to keep, in second.
    method: string, optional
        Scheme for computing the amplitude spectral density.
        --> "fft"        : Fourier method. See Moore et al., 2015 (arXiv:1408.0740).
            "periodogram": periodogram method.
    detrend: string, optional
        Method used for detrending the time-series data. See scipy.signal.periodogram().

    Note
    ----
    The amplitude obtained using the "fft" method could be up to 10 times larger
    than that obtained with the "periodogram" method.
    """
    assert method in ["fft", "periodogram"], "Unsupported method: {}".format(method)

    # re-sample the data
    _, strain = resampling(time, strain, fs = fs, time_window = time_window)

    # compute the amplitude spectral density (ASD)
    # --> the data are truncated at frequencies up to 0.5 * fs (the Nyquist frequency)
    if method == "fft":
        # Follow Moore et al. (2015), see Figure (A2)
        num_data      = len(strain)
        num_data_half = num_data // 2

        # The normalization factor originates from the backward normalization in scipy.fft(),
        # where the factor 2 comes from the one-side power density
        fac_norm = 2. / num_data

        # the sample frequency
        freq = np.linspace(0, 0.5 * fs, num_data_half)  # may be not accurate enough

        # the FFT of the GW strains
        data = fft(strain)
        data = fac_norm * np.abs(data[:num_data_half])

        # no factor 2 here since we use the the one-sided value of FFT
        hchar = freq * data

        # compute the ASD
        mask_nonzero = (freq != 0.0)

        ASD = hchar[mask_nonzero] / np.sqrt(freq[mask_nonzero])

    else:
        # compute the power spectrum density (PSD) using scipy.periodogram
        # --> (1) note that the scaling must be "density"
        #     (2) the returned PSD is in unit of V^2 / Hz
        freq, Pxx = periodogram(strain, fs = fs, window = "hann", scaling = "density",
                                detrend = detrend, **kwargs_periodogram)

        # compute the amplitude spectral density (ASD) via ASD = sqrt(PSD * freq)
        # --> note that `return_onesided` defaults to True in scipy.periodogram(),
        #     so the normalization factor 2 is already included
        ASD = np.sqrt(freq * Pxx)

    return freq, ASD


def calc_spectrogram(time, strain, fs = 8192.0, time_window = None,
                     method = "wavelet", **kwargs):
    """
    Compute the spectrogram of GW strains.

    Parameters
    ----------
    time: array-like of Ffloat
        Time stamp of each data, in second.
    strain: array-like of float
        Measurement of strain at each time stamp.
    fs: float, optional
        Target sampling frequency, in Hz.
    time_window: array-like of float, optional
        Beginning and end of time window to keep, in second.
    method: string, optional
        Scheme for computing the spectrogram of GW strains.
        --> "stft"   : short-time Fourier transform method.
            "wavelet": wavelet method.

    Note
    ----
    For the wavelet method,
        increase dt or 1/fs  -> increase the power
        increase dt * s0_fac -> decrease the frequency range (upper and lower bound)
        increase dj          -> increase the resolution in the frequency domain
        increase J_fac       -> decrease the lower limit of frequency

    The effect of time interval (or sampling frequency) arises because the waves amplitudes
    returned by pycwt.wavelet.cwt() are not normalized and still depend on the sampling
    frequency. To correct this, we multiple the waves amplitudes by sqrt(fs^-1)
    to eliminate this dependence.
    """
    assert method in ["stft", "wavelet"], "Unsupported method: {}".format(method)

    # re-sample the data
    time, strain = resampling(time, strain, fs = fs, time_window = time_window)

    # compute the periodogram
    if method == "stft":
        # store the begging time of data
        time_start = time[0]

        # the unit of spectrum returned by scipy.stft() is same as the input data
        freq, time, spectrogram = stft(strain, fs = fs, **kwargs)

        # correct the time stamp, as the values returned by scipy.stft() start from 0
        time += time_start

        # multiply the amplitude by 2 to compensate for the energy loss
        # caused by the Hann window function
        spectrogram = np.abs(2 * spectrogram)

    else:
        # create the mother wavelet
        mother = wavelet.Morlet(6)

        # set up the parameters passed to wavelet.cwt()
        dt = 1.0 / fs

        s0_fac = kwargs.get("s0_fac", 2.0)
        s0     = s0_fac * dt  # Smallest scale of the wavelet

        dj    = kwargs.get("dj", 1.0 / 12.0)  # number of sub-octaves per octaves
        J_fac = kwargs.get("J_fac", 7.0)      # J_fac-th powers of two with dj sub-octaves
        J     = J_fac / dj

        # the returned fft is already normalized by sqrt(N), where N is the number of data
        wave, scales, freq, coi, fft, fftfreq = wavelet.cwt(strain, dt, dj, s0, J, mother)

        # rectify the amplitude according to Liu+ (2007)
        spectrogram = np.abs(wave) / np.sqrt(scales[:, None])

        # note that there is an additional term, sqrt(N), in the normalization factor
        # in pycwt.wavelet.cwt() (line 103 of wavelet.py).
        #
        # Here, we re-scale the wave amplitude by sqrt(fa^-1) to eliminate the dependence
        # on the sampling rate.
        spectrogram *= np.sqrt(dt)

    return time, freq, spectrogram


def calc_significance(time, strain, fs = 8192.0, time_window = None,
                     sig_level = 0.95, s0_fac = 2.0, dj = 1.0 / 12.0, J_fac = 7.0):
    """
    Compute the significance of the GW spectrogram obtained from the wavelet method.

    Parameters
    ----------
    time: array-like of float
        Time stamp of each data, in second.
    strain: array-like of float
        Measurement of strain at each time stamp.
    fs: float, optional
        Target sampling frequency, in Hz.
    time_window: array-like of float, optional
        Beginning and end of time window to keep, in second.
    sig_level: float, optional
        Target significant level.
    s0_fac: float, optional
        Smallest scale of the wavelet, in units of time interval.
    dj: float, optional
        Number of sub-octaves per octaves.
    J_fac: float, optional
        J_fac-th powers of two with dj sub-octaves.
    """
    # re-sample the data
    time, strain = resampling(time, strain, fs = fs, time_window = time_window)

    # create the mother wavelet
    mother = wavelet.Morlet(6)

    dt = 1.0 / fs
    s0 = s0_fac * dt
    J  = J_fac / dj

    # compute the power (square of wave) with rectification based on Liu+ (2007)
    wave, scales, freq, coi, fft, fftfreq = wavelet.cwt(strain, dt, dj, s0, J, mother)
    power = np.abs(wave)**2 / scales[:, None]

    # examine the significant test at specified significant level
    signif, _ = wavelet.significance(strain, dt, scales, sigma_test = 0,
                                     significance_level = sig_level, wavelet = mother)

    significance = np.ones(shape = [1, strain.size]) * signif[:, None]
    significance = power / significance

    return time, freq, significance
