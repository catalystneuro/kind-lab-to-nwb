"""
    LFP/EEG Analysis Pipeline for Fear Conditioning Paradigm
    Copyright (C) 2023 Dr Paul Rignanese, Kind Lab, University of Edinburgh

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

import numpy as np
from mne import create_info, Epochs
from mne.io import RawArray
from mne.time_frequency import tfr_multitaper


def first_spec(eegch, ch_names, tstartstop, samprate, events):
    ds = 1  # Downsampling factor (if desired)
    W = 2
    T = 1  # seconds
    p = 1
    t = (2 * T * W) - p

    movingwin = (T, 0.2)
    fmin, fmax = 0, 80  # Frequency range in Hz

    info = create_info(list(ch_names), samprate)
    # Create a RawArray from the EEG data
    raw = RawArray(eegch, info=info)
    epochs = Epochs(raw, np.column_stack((events.index, events.values, np.ones(len(events)))).astype(int))
    # Find indices within the specified time range
    start, stop = tstartstop

    # Define the parameters for multitaper analysis
    freqs = np.arange(fmin+1, fmax + 1)
    time_bandwidth = t / 2.0
    sfreq = samprate
    n_cycles = freqs / freqs[1]

    # power = tfr_array_multitaper(eegch.iloc[:, :20000].values.reshape(1, 15, len(eegch.iloc[:, :20000].transpose())), sfreq=sfreq, freqs=freqs, n_jobs=16, verbose=True)
    power = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, n_jobs=16, verbose=True, picks=list(ch_names)[:12])

    spec = {
        'S': power[0].data,
        't': power[0].times,
        'f': power[0].freqs,
    }

    return spec

# # Example usage:
# eegch = np.random.rand(1, 1000)  # Replace with your EEG data
# samprate = 1000
# tstartstop = [0, len(eegch) / samprate]
#
# spec = first_spec(eegch, tstartstop, samprate)
