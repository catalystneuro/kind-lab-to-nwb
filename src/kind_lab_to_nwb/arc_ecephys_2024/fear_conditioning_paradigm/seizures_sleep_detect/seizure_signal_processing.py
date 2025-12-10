"""
    Dual band peaks seizure detection algorithm and interface
    Copyright (C) 2025 Domagoj Anticic, Kind Lab, The University of Edinburgh

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


from scipy.signal import butter, lfilter
import pandas as pd
import numpy as np

def butter_bandpass(lowcut, highcut, fs, order=6):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(sig, lowcut, highcut, fs, order=3):
    """Bandpass filter

    Parameters
    ----------
    sig : np.ndarray
        Signal to filter
    lowcut : float
        Lower cutoff frequency
    highcut : float
        Upper cutoff frequency
    fs : float
        Sampling frequency
    order : int
        Order of the filter

    Returns
    -------
    y : np.ndarray
        Filtered signal
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, sig)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high')
    return b, a

def butter_highpass_filter(sig, cutoff, fs, order=5):
    """Highpass filter

    Parameters
    ----------
    sig : np.ndarray
        Signal to filter
    cutoff : float
        Lower cutoff frequency
    fs : float
        Sampling frequency
    order : int
        Order of the filter

    Returns
    -------
    y : np.ndarray
        Filtered signal
    """
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, sig)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

def butter_lowpass_filter(sig, cutoff, fs, order=5):
    """Lowpass filter

    Parameters
    ----------
    sig : np.ndarray
        Signal to filter
    cutoff : float
        High cutoff frequency
    fs : float
        Sampling frequency
    order : int
        Order of the filter

    Returns
    -------
    y : np.ndarray
        Filtered signal
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, sig)
    return y

def average_power(sig, window_size):
    """Compute rolling window average power of a signal in a band

    Parameters
    ----------
    sig : np.array
        Signal to be averaged
    window_size : int
        Rolling window size
    """
    power = np.power(np.abs(sig), 2)
    in_windows = np.lib.stride_tricks.sliding_window_view(power, window_size)
    rolling_average = np.mean(in_windows, axis=-1)
    return rolling_average