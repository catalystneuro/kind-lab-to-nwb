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

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.params import ask_to_skip_frames
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.params import periods_len_minutes_sleep
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.params import theta_low, theta_high, delta_low, delta_high
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.seizure_signal_processing import butter_bandpass_filter, average_power
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.seizure_timings_processing import filter_short_starts_ends
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.seizure_timings_processing import merge_close_epochs, filter_short_epochs
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.seizure_timings_processing import plot_intervals, good_periods_mask, asserts_starts_ends
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.seizure_timings_processing import starts_ends_from_states, states_from_starts_ends, \
    merge_close_starts_ends

window_size_time = 5 #seconds
ratio_threshold = 2.5
# freezing
max_freeze_gap = 5 #seconds
min_freeze_duration = 20 #seconds
# ratio
max_ratio_gap = 10 #seconds    # not used !! - we merge freeze and NREM, so I think doing it for ratio too is unnecessary
min_ratio_duration = 20 #seconds
# NREM
min_NREM_duration = 120 #seconds
max_NREM_gap = 20 #seconds

def slice_epoch_starts_ends_by_intervals (starts, ends, start_idx, end_idx):
    """Given an array of interval start and end samples and an interval, return starts and ends only from interval, with edge cases

    Parameters
    ----------
    starts : np.array
        Array of start sample indices
    ends : np.array
        Array of end samples indices
    start_idx : int
        Start index of interval
    end_idx : int
        End index of interval

    Returns
    -------
    interval_starts : np.array
        Array of start indices within interval
    interval_ends : np.array
        Array of end indices within interval
    """
    # get indeces of those which end in period and start in period, so this includes cases at edge of window
    idx = np.where((ends > start_idx) & (starts < end_idx))[0]
    # extract values
    interval_starts = starts[idx]
    interval_ends = ends[idx]
    return interval_starts, interval_ends

# TODO
def auto_theshold(sig, val):
    pass

def sleep_detection_fear_paradigm_integration(input_signal, spect_data, motion_signal, freezing_periods, bad_periods_in,
                                    sample_rate, animal_info, session_folder):
    """Interfaces sleep detection with fear conditioning paradigm code"""

    from params import AnalysisParams
    params = AnalysisParams()

    spectrogram = spect_data["Sxx"]

    animal_folder = animal_info['Folder'].to_numpy(copy=True)[0]

    input_signal = input_signal.to_numpy()

    motion_detection_threshold = params.threshold_motion_detection

    NREM_periods = sleep_detection(input_signal, spectrogram, motion_signal, freezing_periods, bad_periods_in, sample_rate,
                    animal_folder, session_folder, motion_detection_threshold)
    return NREM_periods

def sleep_detection(input_signal, spectrogram, motion_signal, freezing_periods, bad_periods, sample_rate, animal_folder, session_folder,
                    motion_detection_threshold):
    """NREM sleep detection given LFP signal, motion signal and bad periods in data. Uses condition of stillness and theta/delta ratio

    All detection is done on total signal provided at once

    Parameters
    ----------
    input_signal : np.array
        LFP signal from where sleep is to be detected
    spectrogram : np.array
        Spectrogram data, used only for plotting
    motion_signal
        Motion signal from where stillness is to be detected
    freezing_periods : pd.dataframe
        Previously detected freezing periods from motion_signal data
    bad_periods : pd.dataframe
        Previously detected bad periods from motion_signal data. Pass None if you do not wish to use this
    sample_rate : int
        DAQ sampling rate
    animal_folder : dict
        Dictionary of animal_info key-value pairs
    session_folder : str
        Path to session folder

    Returns
    -------
    NREM_periods : pd.dataframe
        Detected NREM periods in dataframe with columns "onset" and "offset"
    """

    print("Running sleep detection...")

    # rolling window dimensions
    window_size = int(window_size_time*sample_rate)
    # must be odd
    if window_size % 2 == 0:
        window_size += 1
    edge = window_size // 2

    # make sure data not modified
    signal = input_signal.copy()

    # nans will not work with filtering
    signal = np.nan_to_num(signal)

    # drift not removed - ratio of bands is used so it shouldn't matter

    # extract bands
    theta_band = butter_bandpass_filter(signal, theta_low, theta_high, sample_rate)
    delta_band = butter_bandpass_filter(signal, delta_low, delta_high, sample_rate)

    # calculate band powers and ratio
    theta_pwr = average_power(theta_band, window_size)
    delta_pwr = average_power(delta_band, window_size)
    ratio_pwr = np.divide(theta_pwr, delta_pwr)

    # when ratio is below threshold
    #auto_theshold(ratio_pwr, 1)
    ratio_states = np.where(ratio_pwr < ratio_threshold, True, False)
    ratio_starts, ratio_ends = starts_ends_from_states(ratio_states, edge)  # we add an offset due to rolling window width

    # filter/merge ratio
    ratio_starts, ratio_ends = filter_short_starts_ends(ratio_starts, ratio_ends, min_ratio_duration * sample_rate)
    #ratio_starts, ratio_ends = merge_close_starts_ends(ratio_starts, ratio_ends, max_ratio_gap * sample_rate)
    # filtering short is primarily done to remove lots of short intervals which merging could merge into something which shouldn't be there
    # merging is primarily to deal with short breaks between long epochs

    # merge short gaps and filter short intervals
    freezing_periods = filter_short_epochs(freezing_periods, min_freeze_duration * sample_rate)
    freezing_periods = merge_close_epochs(freezing_periods, max_freeze_gap * sample_rate)

    # convert to numpy arrays as they are faster
    freezing_periods = freezing_periods.to_numpy(copy=True)     # copy is used to make sure we don't modify
    freeze_starts = freezing_periods[:, 0]
    freeze_ends = freezing_periods[:, 1]
    freeze_mask = states_from_starts_ends(freeze_starts, freeze_ends, len(signal))

    good_mask, bad_starts, bad_ends = good_periods_mask(bad_periods, len(signal), return_starts_ends = True)

    # NREM detect
    ratio_mask = states_from_starts_ends(ratio_starts, ratio_ends, len(signal))

    # (NOT bad) AND (ratio AND freeze)
    NREM = np.bitwise_and(good_mask.astype(bool), np.bitwise_and(ratio_mask, freeze_mask))
    NREM_starts, NREM_ends = starts_ends_from_states(NREM, 0)

    # filter/merge NREM
    NREM_starts, NREM_ends = merge_close_starts_ends(NREM_starts, NREM_ends, max_NREM_gap * sample_rate)
    NREM_starts, NREM_ends = filter_short_starts_ends(NREM_starts, NREM_ends, min_NREM_duration * sample_rate)

    # it is possible that we merged over a bad region now, e.g. a short bad region interrupts long sleep
    # if this is undesireable the following lines should be commented out
    NREM2 = states_from_starts_ends(NREM_starts, NREM_ends, len(signal))
    NREM2 = np.bitwise_and(NREM2, good_mask.astype(bool))
    NREM_starts, NREM_ends = starts_ends_from_states(NREM2, 0)

    # following is only for visualisation - all detection was already done!
    # split into periods
    n_samples = len(input_signal)
    print('Session is {} minutes long'.format(n_samples / sample_rate / 60))
    nb_periods = int(np.ceil(n_samples / sample_rate / 60 / periods_len_minutes_sleep))
    print('breaking down in {} periods of 20 minutes max.'.format(nb_periods))
    samples_per_interval = periods_len_minutes_sleep * 60 * sample_rate
    # motion has different sampling rate, a scale of the data one
    samples_per_interval_motion = periods_len_minutes_sleep * 60 * sample_rate * (len(motion_signal) / len(signal))

    # create time axes for plotting
    times_all = np.linspace(0, (len(signal)-1)/sample_rate, num=len(signal))
    times_all_edge_effect = times_all[edge:-edge]
    times_all_motion = np.linspace(0, (len(signal) - 1) / sample_rate, num=len(motion_signal))

    # detect spectrogram decimation
    decimate = len(signal)/spectrogram.shape[1]

    assert len(signal) == len(theta_band)
    assert len(theta_pwr) == len(times_all_edge_effect)

    # process each interval
    print("Press Q to proceed to next window")
    if ask_to_skip_frames:
        start = int(input("Skip N frames: "))
    else:
        start = 0
    i = start
    while i < nb_periods:
        # start and end samples
        start_idx = int(i * samples_per_interval)
        end_idx = int((i + 1) * samples_per_interval)
        start_idx_motion = int(i * samples_per_interval_motion)
        end_idx_motion = int((i + 1) * samples_per_interval_motion)

        spect_start = int(start_idx / decimate)
        spect_end = int(end_idx / decimate)
        spect_section = spectrogram[:, spect_start:spect_end]

        # extract continuous signals
        interval_theta = theta_band[start_idx:end_idx]
        interval_delta = delta_band[start_idx:end_idx]
        interval_theta_pwr = theta_pwr[start_idx:end_idx]
        interval_delta_pwr = delta_pwr[start_idx:end_idx]
        interval_ratio = ratio_pwr[start_idx:end_idx]
        interval_motion_signal = motion_signal[start_idx_motion:end_idx_motion]
        # extract times
        x_times = times_all[start_idx:end_idx]
        x_times_edge_effect = times_all_edge_effect[start_idx:end_idx]
        x_times_motion = times_all_motion[start_idx_motion:end_idx_motion]

        # extract epochs
        interval_freeze_starts, interval_freeze_ends = slice_epoch_starts_ends_by_intervals(freeze_starts, freeze_ends, start_idx, end_idx)
        interval_bad_starts, interval_bad_ends = slice_epoch_starts_ends_by_intervals(bad_starts, bad_ends, start_idx, end_idx)
        interval_ratio_starts, interval_ratio_ends = slice_epoch_starts_ends_by_intervals(ratio_starts, ratio_ends, start_idx, end_idx)
        interval_NREM_starts, interval_NREM_ends = slice_epoch_starts_ends_by_intervals(NREM_starts, NREM_ends, start_idx, end_idx)

        # plot
        title = "Animal = " + str(animal_folder) + ", session = " + str(session_folder)
        fig, axs = plt.subplots(6, sharex=True, figsize=(18, 10), num=title)
        ax1_2 = axs[1].twinx()
        ax2_2 = axs[2].twinx()

        # signal
        axs[0].imshow(spect_section, cmap='inferno', aspect='auto', origin='lower',
                      extent=[x_times[0], x_times[-1], 0, 15], vmin=-25, vmax=25)
        axs[0].set_ylabel('Frequency/Hz')
        axs[0].set_title("Spectrogram")

        axs[1].plot(x_times, interval_theta, alpha=1, linewidth=0.5, color='b')
        axs[1].set_ylabel('Amplitude/a.u.')
        axs[1].set_title("Theta band")
        ax1_2.set_ylabel("Band power", color="r")
        ax1_2.tick_params(axis='y', labelcolor='r')

        axs[2].plot(x_times, interval_delta, alpha=1, linewidth=0.5, color='b')
        axs[2].set_ylabel('Amplitude/a.u.')
        axs[2].set_title("Delta band")
        ax2_2.set_ylabel("Band power", color="r")
        ax2_2.tick_params(axis='y', labelcolor='r')

        # powers
        ax1_2.plot(x_times_edge_effect, interval_theta_pwr, c='r')
        ax2_2.plot(x_times_edge_effect, interval_delta_pwr, c='r')

        axs[3].plot(x_times_motion , interval_motion_signal, c='b')
        if motion_detection_threshold is not None:
            axs[3].axhline(motion_detection_threshold, linestyle='--', c='g')
        axs[3].set_ylabel('Amplitude/a.u.')
        axs[3].set_title("Motion")

        plot_intervals(axs[4], interval_freeze_starts, interval_freeze_ends, sample_rate,'bo-', 1)
        plot_intervals(axs[4], interval_ratio_starts, interval_ratio_ends, sample_rate, 'bo-', 0)
        plot_intervals(axs[4], interval_bad_starts, interval_bad_ends, sample_rate, 'bo-', -1)
        plot_intervals(axs[4], interval_NREM_starts, interval_NREM_ends, sample_rate, 'ro-', -2)

        axs[4].set_xlim(x_times[0], x_times[-1])
        axs[4].set_ylim(-3, 2)
        axs[4].set_yticks([-1, 0, 1, -2])
        axs[4].set_yticklabels(["bad", "ratio", "non-moving", "NREM"])

        # ratio
        axs[5].plot(x_times_edge_effect, interval_ratio, c='r')
        axs[5].axhline(ratio_threshold, linestyle='--', c='g')
        axs[5].set_ylim(0, 10)
        axs[5].set_xlabel('Time/s')
        axs[5].set_ylabel('Amplitude/a.u.')
        plt.show()

        i += 1

    asserts_starts_ends(NREM_starts, NREM_ends, len(signal))

    NREM_periods = pd.DataFrame({'onset': NREM_starts, 'offset': NREM_ends})

    print(NREM_periods)

    # todo in future: add detection of REM and awake

    return NREM_periods