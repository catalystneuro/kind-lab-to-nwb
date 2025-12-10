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

import os
import pickle
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mne.time_frequency import tfr_array_multitaper
from mne_connectivity import seed_target_multivariate_indices
from mne_connectivity import spectral_connectivity_epochs
from scipy.signal import hilbert, lfilter, filtfilt, iirnotch
from scipy.stats import zscore
from tqdm import tqdm

from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.params import AnalysisParams
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.raw_data_extraction import extract_raw_acc_lpf_events
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.timings_processing import extract_led_data

params = AnalysisParams()

class ThresholdSelector:
    def __init__(self, ax, data):
        ax.set_facecolor('black')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self.ax = ax
        self.data = data
        self.floating_threshold = None
        self.ax.axhline(0, color='gray', linewidth=0.8)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_click)

    def on_hover(self, event):
        if event.inaxes == self.ax:
            self.floating_threshold = event.ydata

            if self.floating_threshold is not None:
                self.ax.lines[-1].remove()  # Remove previous threshold line
            self.ax.axhline(self.floating_threshold, color='red', linestyle='--', label='Threshold')
            self.ax.figure.canvas.draw()

    def on_click(self, event):
        if event.inaxes == self.ax:
            self.floating_threshold = event.ydata
            self.selected_threshold = event.ydata

            self.ax.axhline(self.floating_threshold, color='red', linestyle='--', label='Threshold')
            self.ax.figure.canvas.draw()


def generate_animal_analysis_path(session_results_folder, analysis_type, data_type, extension='pkl'):
    return {f'{data_type}_{analysis_type}': f'{session_results_folder}{data_type}_{analysis_type}.{extension}'}

def lowpass_filtering_for_motion(cutoff_freq, frequencies, accelerometer_data):
    # Find the index corresponding to the cutoff frequency
    x_axis = accelerometer_data[:, 0]
    y_axis = accelerometer_data[:, 1]
    z_axis = accelerometer_data[:, 2]

    # Compute the corresponding frequencies
    fft_x = np.fft.fft(x_axis)
    fft_y = np.fft.fft(y_axis)
    fft_z = np.fft.fft(z_axis)

    # Apply cutoff frequency to filter the signal

    cutoff_index = np.abs(frequencies - cutoff_freq).argmin()

    # Set the magnitude spectrum values above the cutoff index to zero
    fft_x_filtered = fft_x.copy()
    fft_x_filtered[cutoff_index:] = 0
    fft_y_filtered = fft_y.copy()
    fft_y_filtered[cutoff_index:] = 0
    fft_z_filtered = fft_z.copy()
    fft_z_filtered[cutoff_index:] = 0
    # Inverse FFT to obtain the filtered signals
    x_axis_filtered = np.fft.ifft(fft_x_filtered)
    y_axis_filtered = np.fft.ifft(fft_y_filtered)
    z_axis_filtered = np.fft.ifft(fft_z_filtered)

    x_axis_filtered = np.real(x_axis_filtered)
    y_axis_filtered = np.real(y_axis_filtered)
    z_axis_filtered = np.real(z_axis_filtered)

    return x_axis_filtered, y_axis_filtered, z_axis_filtered


def delay_video_frames_timestamps(frames_numbers, frames_timestamps, delay):
    frames_timestamps = [frames_timestamp + delay for frames_timestamp in frames_timestamps]  #
    frames_timestamps = dict(zip(frames_numbers, frames_timestamps))

    return frames_timestamps


def extract_motion_for_each_video_frame_timestamp(frames_numbers, frames_timestamps, processed_motion, threshold):
    frames_motion = {}
    frames_immobile = {}

    for frame_number, timestamp in tqdm(frames_timestamps.items(), total=len(frames_numbers)):
        if not processed_motion.loc[timestamp - 100:timestamp + 100].index.empty:
            closest_index = min(processed_motion.loc[timestamp - 100:timestamp + 100].index,
                                key=lambda x: abs(x - timestamp))
            closest_index_motion = processed_motion.loc[closest_index]
            frames_motion[frame_number] = closest_index_motion * 200000 - 800
            if closest_index_motion < threshold:
                frames_immobile[frame_number] = True
            else:
                frames_immobile[frame_number] = False
        else:
            frames_motion[frame_number] = 0 * 200000 - 800
            frames_immobile[frame_number] = False

    return frames_motion, frames_immobile


def extract_frequencies_from_acc_data(accelerometer_data, sample_rate):

    frequencies = np.fft.fftfreq(len(accelerometer_data), 1 / sample_rate)
    frequencies = np.abs(frequencies)

    return frequencies


def extract_signal_within_timings(signal_df, timings_df, min_len_epoch, decimate, zscore_each):
    extracted_signal = []  # Create an empty DataFrame to store extracted signal

    # Iterate through each row in the DataFrame
    for index, row in timings_df.iterrows():
        onset = row['onset']
        offset = row['offset']

        # Filter the signal DataFrame to select only the columns within the timing range
        selected_columns = signal_df.columns[(signal_df.columns >= onset) & (signal_df.columns <= offset)]

        extracted_signal.append(signal_df[selected_columns])
    # Initialize a new list to store the segmented epochs
    segmented_epochs_list = []

    # Iterate over epochs (DataFrames) in the original list
    for original_epoch in extracted_signal:
        n_channels, n_times = original_epoch.shape

        # Calculate the number of segments
        num_segments = n_times // min_len_epoch

        # Split the original epoch into segments
        for i in range(int(num_segments)):
            start_idx = i * min_len_epoch
            end_idx = (i + 1) * min_len_epoch
            segmented_epoch = original_epoch.iloc[:, int(start_idx):int(end_idx)]

            if segmented_epoch.shape[1] == min_len_epoch:
                if not pd.isna(segmented_epoch).any().any():
                # Append the segmented epoch to the new list
                    if zscore_each:
                        if decimate:
                            segmented_epochs_list.append(zscore(segmented_epoch.iloc[:, ::decimate], axis=1))
                        else:
                            segmented_epochs_list.append(zscore(segmented_epoch, axis=1))

                    else:
                        if decimate:

                            segmented_epochs_list.append(segmented_epoch.iloc[:, ::decimate])
                        else:
                            segmented_epochs_list.append(segmented_epoch)

    data_array = np.array(segmented_epochs_list)

    return data_array


def extract_and_save_avg_spectrograms(full_lfp_data, sample_rate, timings, onset_or_offset, time_window_before,
                                      timewindow_after,
                                      result_path, decimate, label, redo=False):
    if not os.path.exists(result_path) or redo:
        data_arrays = []
        for times in timings.iterrows():
            if not np.isnan(times[1][onset_or_offset]):
                start_idx = pd.Series(full_lfp_data.columns).sub(
                    times[1][onset_or_offset] - time_window_before).abs().idxmin()
                end_idx = pd.Series(full_lfp_data.columns).sub(
                    times[1][onset_or_offset] + timewindow_after).abs().idxmin()

                num_timepoints = int(time_window_before + timewindow_after)  # int(time_window_ms/1000*(time_step*2))  # Calculate the number of columns
                cropped_lfp_data = full_lfp_data.iloc[:, start_idx:end_idx]  # .dropna()

                if len(cropped_lfp_data.columns) == num_timepoints and len(cropped_lfp_data.index):
                    cropped_lfp_data = pd.DataFrame(zscore(cropped_lfp_data, axis=1, nan_policy='omit'))

                    data_arrays.append(cropped_lfp_data)

        freqs = np.arange(params.spect_min_freq, params.spect_max_freq,
                          params.spect_freq_resolution)  # Example frequency range

        time_bandwidth_product = (params.spect_n_tapers + 1) / 2  # Compute time-bandwidth product

        if len(data_arrays):
            spectrograms = tfr_array_multitaper(np.array(data_arrays), sfreq=sample_rate, freqs=freqs, decim=decimate,
                                                n_cycles=params.spect_n_cycles, verbose=True, output='avg_power',
                                                time_bandwidth=time_bandwidth_product, n_jobs=-1)
        else:
            spectrograms = None

        if spectrograms is not None:
            with h5py.File(result_path, 'w') as f:
                f.create_dataset(label, data=spectrograms, compression='gzip')
    else:
        with h5py.File(result_path, 'r') as f:
            spectrograms = f[label][:]

    return spectrograms


def extract_clean_freezing_transitions(df, sample_rate):
    # Constants
    min_duration_samples = 5 * sample_rate  # 5 seconds
    min_isolation_samples = 5 * sample_rate  # 5 seconds

    # Calculate durations and add a column for it
    df['duration'] = df['offset'] - df['onset']

    # Filter based on duration
    df = df[df['duration'] >= min_duration_samples]

    # Create a shifted DataFrame to compute the isolation condition
    df['next_onset'] = df['onset'].shift(-1)
    df['prev_offset'] = df['offset'].shift(1)

    # Fill NaNs for boundary rows
    df['next_onset'] = df['next_onset'].fillna(df['next_onset'].max() + min_isolation_samples)
    df['prev_offset'] = df['prev_offset'].fillna(df['prev_offset'].min() - min_isolation_samples)

    # Filter based on isolation
    df = df[(df['onset'] - df['prev_offset'] >= min_isolation_samples) &
            (df['next_onset'] - df['offset'] >= min_isolation_samples)]

    # Drop helper columns
    df = df.drop(columns=['duration', 'next_onset', 'prev_offset'])
    return df


def extract_and_save_avg_coherograms(full_lfp_data, areas_dict_animal, sample_rate, timings, onset_or_offset,
                                     time_window_before, time_window_after,
                                     result_path, decimate, redo=False):

    if decimate:
        sample_rate = sample_rate/decimate

    if redo or not os.path.exists(result_path):
        data_arrays = []
        for times in timings.iterrows():
            if not np.isnan(times[1][onset_or_offset]):
                start_idx = pd.Series(full_lfp_data.columns).sub(
                    times[1][onset_or_offset] - time_window_before).abs().idxmin()
                end_idx = pd.Series(full_lfp_data.columns).sub(
                    times[1][onset_or_offset] + time_window_after).abs().idxmin()

                num_timepoints = int((time_window_before + time_window_after) /
                        decimate)  # int(time_window_ms/1000*(time_step*2))  # Calculate the number of columns
                cropped_lfp_data = full_lfp_data.iloc[:, start_idx:end_idx].dropna()

                if int(len(cropped_lfp_data.columns)/decimate) == num_timepoints and len(cropped_lfp_data.index):
                    if decimate:
                        cropped_lfp_data = pd.DataFrame(zscore(cropped_lfp_data.iloc[:, ::decimate], axis=1, nan_policy='omit'))
                    else:
                        cropped_lfp_data = pd.DataFrame(zscore(cropped_lfp_data, axis=1, nan_policy='omit'))

                    data_arrays.append(cropped_lfp_data)

        freqs = np.arange(params.coheros_min_freq, params.coheros_max_freq, params.coheros_freq_resolution)

        time_bandwidth_product = (params.coheros_n_tapers + 1) / 2
        univariate_measures = ['coh', 'imcoh', 'dpli']

        if len(data_arrays):

            avg_within_timings_univariate_coherograms_measures = spectral_connectivity_epochs(np.array(data_arrays),
                                                                                              method=univariate_measures,
                                                                                              names=list(
                                                                                                  areas_dict_animal.values()),
                                                                                              sfreq=sample_rate,
                                                                                              cwt_freqs=freqs,
                                                                                              cwt_n_cycles=params.coheros_n_cycles,
                                                                                              mode='cwt_morlet',
                                                                                              n_jobs=1,
                                                                                              # need those jobs and block size settings to avoid memory error
                                                                                              mt_bandwidth=time_bandwidth_product)

        else:
            avg_within_timings_univariate_coherograms_measures = False

        if avg_within_timings_univariate_coherograms_measures:
            with h5py.File(result_path, 'w') as f:
                for i, univariate_measure in enumerate(univariate_measures):
                    f.create_dataset(
                        univariate_measure,
                        data=avg_within_timings_univariate_coherograms_measures[i].get_data(),
                        compression='gzip'
                    )

    return None


def get_first_blue_frame(video_blue_frames):
    fig, ax = plt.subplots(figsize=(25, 10))
    ax.plot(video_blue_frames, alpha=0.5)
    selector = ThresholdSelector(ax, video_blue_frames)

    ax.set_xlabel('Time')
    ax.set_ylabel('Video Blue Intensity')
    plt.show()

    if selector.selected_threshold is not None:
        crossing_index = np.where(np.array(video_blue_frames) >= selector.selected_threshold)[0][0]

    return crossing_index


def acceleration_magnitude_extraction_for_motion(accelerometer_data, sample_rate, cutoff_freq_motion):
    if not np.isnan(accelerometer_data).all():
        frequencies = extract_frequencies_from_acc_data(accelerometer_data, sample_rate)

        x_axis_filtered_motion, y_axis_filtered_motion, z_axis_filtered_motion = lowpass_filtering_for_motion(
            cutoff_freq_motion, frequencies,
            accelerometer_data)


        processed_acc_to_motion = pd.DataFrame(
            np.vstack([x_axis_filtered_motion, y_axis_filtered_motion, z_axis_filtered_motion]).transpose(),
            columns=['x_filtered', 'y_filtered', 'z_filtered'])

        # Normalize the filtered data
        processed_acc_to_motion['x_normalized'] = (processed_acc_to_motion['x_filtered'] - processed_acc_to_motion[
            'x_filtered'].mean()) / \
                                                  processed_acc_to_motion[
                                            'x_filtered'].std()
        processed_acc_to_motion['y_normalized'] = (processed_acc_to_motion['y_filtered'] - processed_acc_to_motion[
            'y_filtered'].mean()) / \
                                                  processed_acc_to_motion[
                                            'y_filtered'].std()
        processed_acc_to_motion['z_normalized'] = (processed_acc_to_motion['z_filtered'] - processed_acc_to_motion[
            'z_filtered'].mean()) / \
                                                  processed_acc_to_motion['z_filtered'].std()

        # Calculate the magnitude of the acceleration vector
        processed_acc_to_motion['motion'] = np.sqrt(processed_acc_to_motion['x_normalized'].diff() ** 2 +
                                                    processed_acc_to_motion['y_normalized'].diff() ** 2 +
                                                    processed_acc_to_motion['z_normalized'].diff() ** 2)

        processed_motion = processed_acc_to_motion['motion'].iloc[
                           ::4]  # sampling rate not as fast as the lfp, removes duplicates

    else:
        processed_motion = pd.Series(np.nan)

    return processed_motion


def compute_whole_session_spectrograms(lfp_data_filtered, params):
    if params.zscore_whole_session_spects:
        lfp_data_filtered = pd.DataFrame(zscore(lfp_data_filtered, axis=1))

    freqs = np.arange(params.whole_session_spect_min_freq, params.whole_session_spect_max_freq,
                      params.whole_session_spect_freq_resolution)

    data_array = lfp_data_filtered.values.reshape(1, lfp_data_filtered.shape[0], lfp_data_filtered.shape[1])

    new_spectrograms_dict = tfr_array_multitaper(data_array, sfreq=params.sample_rate, freqs=freqs,
                                                 n_cycles=params.whole_session_spect_n_cycles,
                                                 verbose=True, output='power',
                                                 time_bandwidth=params.whole_session_spect_time_bandwidth,
                                                 n_jobs=params.whole_session_spect_n_jobs,
                                                 decim=params.whole_session_spect_decimation)
    new_spectrograms_dict = new_spectrograms_dict[0]

    spectrograms_dict = {}
    for i, row in lfp_data_filtered.reset_index(drop=True).iterrows():
        spectrograms_dict[i] = {'spect_time_index': lfp_data_filtered.columns[::params.whole_session_spect_decimation],
                                'Sxx': 10 * np.log10(new_spectrograms_dict[i]), 'f': freqs}

    return spectrograms_dict


def extract_signal_around_led_events(lfp_filtered, cs_timings, led_events, time_before=5, time_after=35,
                                     sampling_rate=2000):
    extracted_signals = {}

    # Iterate through each row (assuming each row corresponds to a signal)
    for row_idx, row in lfp_filtered.iterrows():
        signal_data = row.values  # Convert the row to a NumPy array

        signal_snapshots = []  # Store signal snapshots for each onset time
        # Loop through each onset time in cs_timings

        for i, timings in cs_timings.iterrows():
            # Calculate the start and end indices for the time window around the onset
            cs_led_events = led_events.loc[(led_events >= timings['onset']) & (led_events <= timings['offset'])]

            for n, e in enumerate(cs_led_events):
                start_index = int((e - time_before * sampling_rate))
                end_index = int((e + time_after * sampling_rate))

                # Ensure that the indices are within the data range
                start_index = max(start_index, 0)
                end_index = min(end_index, len(signal_data))

                # Extract the signal data for this time window
                signal_snapshot = signal_data[start_index:end_index]

                # Append the signal snapshot to the list
                signal_snapshot = np.insert(signal_snapshot, 0, i)
                signal_snapshots.append(signal_snapshot)

        df = pd.DataFrame(signal_snapshots)
        df_columns = df.columns[:-1].to_list()
        df_columns.insert(0, 'cs_number')
        df.columns = df_columns
        extracted_signals[row_idx] = df

    # Convert the list of DataFrames to a single DataFrame
    return extracted_signals


def extract_motion_around_timings(motion, epochs_timings, onset_offset, time_before=5, time_after=5,
                                  sampling_rate=2000):

    epochs_timings = epochs_timings[onset_offset]
    signal_snapshots = []

    # Loop through each onset time in cs_timings
    for timing in epochs_timings.values:
        # Calculate the start and end indices for the time window around the onset

        start_index = int((timing - time_before * sampling_rate))
        end_index = int((timing + time_after * sampling_rate))

        # Ensure that the indices are within the data range
        start_index = max(start_index, motion.index.min())
        end_index = min(end_index, motion.index.max())

        # Extract the signal data for this time window
        signal_snapshot = motion.loc[start_index:end_index]

        # Append the signal snapshot to the list
        signal_snapshots.append(signal_snapshot.values)

    df = pd.DataFrame(signal_snapshots)

    # Convert the list of DataFrames to a single DataFrame
    return df


def extract_signal_around_cs_onsets(lfp_filtered, cs_timings, time_before=5, time_after=35, sampling_rate=2000):
    extracted_signals = {}

    # Iterate through each row (assuming each row corresponds to a signal)
    for row_idx, row in lfp_filtered.iterrows():
        signal_data = row.values  # Convert the row to a NumPy array

        signal_snapshots = []  # Store signal snapshots for each onset time

        # Loop through each onset time in cs_timings
        for i, onset_time in enumerate(cs_timings['onset']):
            # Calculate the start and end indices for the time window around the onset
            start_index = int((onset_time - time_before * sampling_rate))
            end_index = int((onset_time + time_after * sampling_rate))

            # Ensure that the indices are within the data range
            start_index = max(start_index, 0)
            end_index = min(end_index, len(signal_data))

            # Extract the signal data for this time window
            signal_snapshot = signal_data[start_index:end_index]

            # Append the signal snapshot to the list
            signal_snapshot = np.insert(signal_snapshot, 0, i)
            signal_snapshots.append(signal_snapshot)

        df = pd.DataFrame(signal_snapshots)
        df_columns = df.columns[:-1].to_list()
        df_columns.insert(0, 'cs_number')
        df.columns = df_columns
        extracted_signals[row_idx] = df

    # Convert the list of DataFrames to a single DataFrame
    return extracted_signals


def extract_signal_around_timing_series(motion, timing_onset, timing_offset, time_before, time_after,
                                        sampling_rate=2000):
    # Calculate the start and end indices for the time window around the onset
    start_index = int((timing_onset - time_before * sampling_rate))
    end_index = int((timing_offset + time_after * sampling_rate))

    # Extract the signal data for this time window
    signal_snapshot = motion.loc[start_index:end_index]

    # Convert the list of DataFrames to a single DataFrame
    return signal_snapshot


def extract_signal_around_timing_df(lfp_filtered, timing_onset, timing_offset, time_before, time_after,
                                    sampling_rate=2000):
    # Calculate the start and end indices for the time window around the onset
    start_index = int((timing_onset - time_before))
    end_index = int((timing_offset + time_after))

    # Extract the signal data for this time window
    signal_snapshot = lfp_filtered.loc[:, start_index:end_index]

    # Convert the list of DataFrames to a single DataFrame
    return signal_snapshot


def average_spectrograms(aligned_spectrograms):
    average_spectrograms = {}
    for area, spects in aligned_spectrograms.items():

        if spects:
            all_have_some_nans = all(df.isna().any().any() for df in spects)
            if all_have_some_nans:
                # If all dataframes are NaN, return a NaN-filled dataframe with the same shape
                avg_spectrogram_onset = np.full_like(aligned_spectrograms[area][0], np.nan)

            else:
                # Calculate the mean while ignoring NaN values
                # Convert the list of DataFrames to a 3D NumPy array
                array_of_dfs = np.array([df.values for df in spects])

                # Calculate the mean along the third axis (axis=0 in 3D array)
                avg_spectrogram_onset = np.nanmean(array_of_dfs, axis=0)
            average_spectrograms[area] = avg_spectrogram_onset
        else:
            average_spectrograms[area] = np.nan
    return average_spectrograms


def average_coherograms(aligned_coherograms):
    average_coherograms = {}
    for comb, cohero in aligned_coherograms.items():
        all_have_some_nans = all(df.isna().any().any() for df in cohero)

        if all_have_some_nans:
            # If all dataframes are NaN, return a NaN-filled dataframe with the same shape
            avg_coherogram_onset = False

        else:
            # Calculate the mean while ignoring NaN values
            avg_coherogram_onset = np.nanmean(aligned_coherograms[comb], axis=0)
        average_coherograms[comb] = avg_coherogram_onset

    return average_coherograms


def compute_parallel_cross_corr(within_timings_lfp):

    def cross_cor(args):
        i, j, data = args
        x = data[:, i, :]
        y = data[:, j, :]
        cross_corr = np.mean([np.correlate(x[k], y[k], mode='full') for k in range(data.shape[0])], axis=0)
        cross_corr /= np.sqrt(np.mean(np.sum(x ** 2, axis=1)) * np.mean(np.sum(y ** 2, axis=1)))
        return i, j, cross_corr.max()

    n_epochs, n_channels, n_samples = within_timings_lfp.shape
    cross_corr_results = np.zeros((n_channels, n_channels))
    #
    # Create argument list for parallel processing
    args_list = [(i, j, within_timings_lfp) for i in range(n_channels) for j in range(i + 1, n_channels)]
    # Initialize Pool for parallel processing
    with Pool() as pool:
        #    Parallel computation of cross-correlation
        results = list(tqdm(pool.imap(cross_cor, args_list), total=len(args_list)))
    #
    # Update cross_corr_results with parallel results
    for result in results:
        i, j, corr_value = result
        cross_corr_results[i, j] = corr_value
        cross_corr_results[j, i] = corr_value

    return cross_corr_results


def compute_phase_differences(data, sample_rate, lowcut, highcut):
    """
       Compute phase differences for all channel combinations, including self-pairs.

       Parameters:
       -----------
       data : ndarray
           Input data of shape (n_epochs, n_channels, n_samples).

       Returns:
       --------
       phase_diffs : ndarray
           Phase differences for all channel combinations, including self-pairs.
           Shape: (n_epochs, n_channels * n_channels, n_samples).
       """

    filtered_data = bandpass_filter_epochs(data, lowcut, highcut, sample_rate)

    n_epochs, n_channels, n_samples = filtered_data.shape
    phase_diffs = np.zeros((n_epochs, n_channels * n_channels, n_samples))

    for epoch_idx in range(n_epochs):
        epoch_data = filtered_data[epoch_idx]

        # Compute the analytic signal using Hilbert transform
        analytic_signals = hilbert(epoch_data, axis=1)
        phases = np.angle(analytic_signals)  # Extract instantaneous phase

        # Calculate phase differences for all combinations of channels
        comb_idx = 0
        for i in range(n_channels):
            for j in range(n_channels):
                phase_diffs[epoch_idx, comb_idx] = np.unwrap(phases[i] - phases[j])
                comb_idx += 1

    return phase_diffs


def extract_and_save_within_timings_coherence(lfp, areas_dict_animal, timings, sample_rate, results_path, redo=False):
    time_bandwidth_product = (params.cohe_n_tapers + 1) / 2  # Compute time-bandwidth product

    seeds, targets = seed_target_multivariate_indices([[i] for i in range(len(areas_dict_animal))], [[i] for i in range(len(areas_dict_animal))])
    # Remove entries with matching ints
    filtered_list1 = []
    filtered_list2 = []

    for item1, item2 in zip(seeds, targets):
        if item1[0] != item2[0]:
            filtered_list1.append(item1)
            filtered_list2.append(item2)

    # Remove entries with the same [int] to [int] pairs
    unique_pairs = set()

    final_list1 = []
    final_list2 = []

    for item1, item2 in zip(filtered_list1, filtered_list2):
        pair = (item1[0], item2[0])
        if pair not in unique_pairs:
            unique_pairs.add(pair)
            final_list1.append(item1)
            final_list2.append(item2)

    indices_multivar = (final_list1, final_list2)
    univariate_measures = ['coh', 'imcoh', 'dpli', 'wpli']

    multivariate_measures = ['gc']
    if redo or not os.path.exists(results_path):
        if len(timings):
            within_timings_lfp = extract_signal_within_timings(lfp, timings,
                                                               min_len_epoch=params.epochs_length * params.sample_rate,
                                                               decimate=False, zscore_each=False)
            if len(within_timings_lfp):
                avg_within_timings_univariate_coherence_measures = spectral_connectivity_epochs(
                    within_timings_lfp,
                    method=['coh', 'imcoh', 'dpli'],
                    fmin=params.cohe_min_freq,
                    fmax=params.cohe_max_freq,
                    sfreq=sample_rate,
                    names=list(areas_dict_animal.values()),
                    mode='multitaper', n_jobs=-1,
                    mt_bandwidth=time_bandwidth_product
                )

                avg_within_timings_multivariate_coherence_measures = spectral_connectivity_epochs(
                    within_timings_lfp, indices=indices_multivar,
                    method=multivariate_measures,
                    fmin=params.cohe_min_freq,
                    fmax=params.cohe_max_freq,
                    sfreq=sample_rate, names=list(areas_dict_animal.values()),
                    mode='multitaper', n_jobs=-1,
                    mt_bandwidth=time_bandwidth_product
                )

                avg_within_timings_univariate_coherence_wpli = spectral_connectivity_epochs(
                    within_timings_lfp,
                    method=['wpli'],
                    fmin=params.cohe_min_freq,
                    fmax=params.cohe_max_freq,
                    sfreq=sample_rate,
                    indices=([i[0] for i in seeds], [i[0] for i in targets]),
                    mode='multitaper', n_jobs=-1,
                    mt_bandwidth=time_bandwidth_product
                )

                with h5py.File(results_path, 'w') as f:
                    f.create_dataset('coh', data=avg_within_timings_univariate_coherence_measures[0].get_data(),
                                     compression='gzip')
                    f.create_dataset('imcoh', data=avg_within_timings_univariate_coherence_measures[1].get_data(),
                                     compression='gzip')
                    f.create_dataset('dpli', data=avg_within_timings_univariate_coherence_measures[2].get_data(),
                                     compression='gzip')
                    f.create_dataset('wpli', data=avg_within_timings_univariate_coherence_wpli.get_data(),
                                     compression='gzip')
                    f.create_dataset('gc', data=avg_within_timings_multivariate_coherence_measures.get_data(),
                                     compression='gzip')

    return None


def extract_merge_two_hab_sessions_cond(animal_base_folder, timing_type_cond,
                                        valid_electrodes_cleaned, geno, folder,
                                        psds):
    session_1_dir_cond = animal_base_folder + 'Hab_1/'
    session_1_results_folder_cond = session_1_dir_cond + '/results/'
    animal_psds_cond_session_1 = pd.read_hdf(session_1_results_folder_cond + '{}_psds.h5'.format(timing_type_cond),
                                             '{}_psds'.format(timing_type_cond))

    session_2_dir_cond = animal_base_folder + 'Hab_2/'
    session_2_results_folder_cond = session_2_dir_cond + '/results/'
    animal_psds_cond_session_2 = pd.read_hdf(session_2_results_folder_cond + '{}_psds.h5'.format(timing_type_cond),
                                             '{}_psds'.format(timing_type_cond))

    if not animal_psds_cond_session_1.empty or animal_psds_cond_session_2.empty:
        animal_psds = pd.concat([animal_psds_cond_session_1, animal_psds_cond_session_2]).groupby(level=0).mean()

        animal_psds['area'] = valid_electrodes_cleaned
        animal_psds['geno'] = geno
        animal_psds['animal'] = folder
        animal_psds['cond'] = 'Hab_' + timing_type_cond

        psds = pd.concat([psds, animal_psds], ignore_index=True)

    return psds


def extract_and_save_within_timings_avg_psds(lfp, timings, sample_rate, results_path, results_key, redo=False):
    freqs = np.arange(params.psds_min_freq, params.psds_max_freq,
                      params.psds_freq_resolution)  # Example frequency range
    decimate = False

    if not os.path.exists(results_path) or redo:
        if len(timings):

            mean = np.nanmean(lfp.values, axis=1, keepdims=True)
            std = np.nanstd(lfp.values, axis=1, keepdims=True)
            zscored = (lfp.values - mean) / std
            zscored_lfp = pd.DataFrame(zscored, index=lfp.index,
                                       columns=lfp.columns)  # need to do it like this because scipy does not have the option to ignore nans for zscoring

            within_timings_lfp = extract_signal_within_timings(zscored_lfp, timings, min_len_epoch=10000,
                                                               decimate=decimate, zscore_each=False)
            if len(within_timings_lfp):

                timings_avg_psds = tfr_array_multitaper(within_timings_lfp, sfreq=sample_rate, freqs=freqs,
                                                        n_cycles=params.psds_n_cycles,
                                                        time_bandwidth=params.psds_time_bandwidth, n_jobs=-1,
                                                        output='avg_power', verbose=True)
                within_timings_avg_psds = pd.DataFrame(timings_avg_psds.mean(axis=2))
            else:
                within_timings_avg_psds = pd.DataFrame()

        else:
            within_timings_avg_psds = pd.DataFrame()

        if os.path.exists(results_path):
            os.remove(results_path)
            print(f"File {results_path} deleted, writing a new one.")
        else:
            print(f"File {results_path} not found, writing a new one.")

        within_timings_avg_psds.to_hdf(results_path, results_key)

    return None


def raw_data_preprocessing(session_results_paths, folder, session_folder, src, params):
    if not os.path.exists(session_results_paths['lfp_preprocessed_data']) or params.redo_raw_data_extraction:
        lfp_data, accelerometer_data, ttl_events, sample_rate = extract_raw_acc_lpf_events(folder, params.data_type,
                                                                                           session_folder, src)

        lfp_data_filtered = notch_filter_lfp(lfp_data, params.notch_frequency, params.quality_factor, sample_rate)

        processed_motion = acceleration_magnitude_extraction_for_motion(
            accelerometer_data, sample_rate, params.motion_processing_cutoff_freq)

        lfp_data_filtered.to_hdf(session_results_paths['lfp_preprocessed_data'], key='lfp_processed')
        processed_motion.to_hdf(session_results_paths['motion_preprocessed_data'], key='acc_processed')

        ttl_events.to_hdf(session_results_paths['ttl_events_preprocessed_data'], key='ttl_events_processed')

    else:
        lfp_data_filtered = pd.read_hdf(session_results_paths['lfp_preprocessed_data'], 'lfp_processed')
        processed_motion = pd.read_hdf(session_results_paths['motion_preprocessed_data'], 'acc_processed')

        ttl_events = pd.read_hdf(session_results_paths['ttl_events_preprocessed_data'], 'ttl_events_processed')
    led_data = extract_led_data(ttl_events, data_type=params.data_type)

    return lfp_data_filtered, processed_motion, led_data


def notch_filter_lfp(lfp_data, notch_frequency, quality_factor, sample_rate):
    nyquist = sample_rate / 2
    assert notch_frequency < nyquist, "notch_frequency must be less than the Nyquist frequency (sample_rate / 2)"
    b, a = iirnotch(notch_frequency / nyquist, quality_factor)
    lfp_data_filtered = lfp_data.apply(lambda row: lfilter(b, a, row))
    return lfp_data_filtered



def extract_led_pulses_locked_traces(session_results_paths, cleaned_lfp_data_filtered, timings, led_data):
    if not os.path.exists(session_results_paths['led_events_locked_traces']):
        # Save single_led_locked_200ms DataFrame with a specific key
        led_pulses_locked_traces = extract_signal_around_led_events(cleaned_lfp_data_filtered, timings['cs_timings_raw'],
                                                                    led_data['led_pulses_events'],
                                                                    time_before=0.2, time_after=0.2)
        with open(session_results_paths['led_events_locked_traces'], 'wb') as file:
            pickle.dump(led_pulses_locked_traces, file)

    else:
        with open(session_results_paths['led_events_locked_traces'], 'rb') as file:
            led_pulses_locked_traces = pickle.load(file)

    return led_pulses_locked_traces


def extract_signal_not_around_led_events_to_mock_data_for_bootstrapping(lfp_filtered, noncs_timings, time_before=5,
                                                                        time_after=35,
                                                                        sampling_rate=2000):
    extracted_signals = {}

    # Iterate through each row (assuming each row corresponds to a signal)
    for row_idx, row in lfp_filtered.iterrows():
        signal_data = row.values  # Convert the row to a NumPy array

        signal_snapshots = []  # Store signal snapshots for each onset time
        # Loop through each onset time in cs_timings

        for i, timings in tqdm(noncs_timings.iterrows()):
            # Calculate the start and end indices for the time window around the onset
            mock_noncs_led_events = np.arange(int(timings['onset']), int(timings['offset']), step=1)

            for n, e in enumerate(mock_noncs_led_events):
                start_index = int((e - time_before * sampling_rate))
                end_index = int((e + time_after * sampling_rate))

                # Ensure that the indices are within the data range
                start_index = max(start_index, 0)
                end_index = min(end_index, len(signal_data))

                # Extract the signal data for this time window
                signal_snapshot = signal_data[start_index:end_index]

                # Append the signal snapshot to the list
                signal_snapshot = np.insert(signal_snapshot, 0, i)
                signal_snapshots.append(signal_snapshot)

        df = pd.DataFrame(signal_snapshots)
        df_columns = df.columns[:-1].to_list()
        df_columns.insert(0, 'noncs_number')
        df.columns = df_columns
        extracted_signals[row_idx] = df

    # Convert the list of DataFrames to a single DataFrame
    return extracted_signals


def extract_led_pulses_locked_traces_and_random_timings_trances_for_bootstrapping(session_results_paths,
                                                                                  cleaned_lfp_data_filtered, timings,
                                                                                  led_data):
    if not os.path.exists(session_results_paths['led_events_locked_traces']):
        # Save single_led_locked_200ms DataFrame with a specific key
        led_pulses_locked_traces = extract_signal_around_led_events(cleaned_lfp_data_filtered, timings['cs_raw'],
                                                                    led_data['led_pulses_events'],
                                                                    time_before=0.2, time_after=0.2)
        with open(session_results_paths['led_events_locked_traces'], 'wb') as file:
            pickle.dump(led_pulses_locked_traces, file)

    else:
        with open(session_results_paths['led_events_locked_traces'], 'rb') as file:
            led_pulses_locked_traces = pickle.load(file)

    if not os.path.exists(session_results_paths['mock_led_events_locked_traces']):

        mock_pulses_locked_traces = extract_signal_not_around_led_events_to_mock_data_for_bootstrapping(
            cleaned_lfp_data_filtered, timings['noncs'],
            time_before=0.2, time_after=0.2)
        with open(session_results_paths['mock_led_events_locked_traces'], 'wb') as file:
            pickle.dump(mock_pulses_locked_traces, file)

    else:
        with open(session_results_paths['mock_led_events_locked_traces'], 'rb') as file:
            mock_pulses_locked_traces = pickle.load(file)

    return led_pulses_locked_traces, mock_pulses_locked_traces


def extract_cs_onset_locked_traces(session_results_paths, cleaned_lfp_data_filtered, timings):
    if not os.path.exists(session_results_paths['cs_onsets_locked_traces']):
        # Save cs_locked_2s DataFrame with a specific key
        cs_locked_2s = extract_signal_around_cs_onsets(cleaned_lfp_data_filtered, timings['cs_timings_raw'], time_before=10,
                                                       time_after=30)
        with open(session_results_paths['cs_onsets_locked_traces'], 'wb') as file:
            pickle.dump(cs_locked_2s, file)

    else:
        with open(session_results_paths['cs_onsets_locked_traces'], 'rb') as file:
            cs_locked_2s = pickle.load(file)
    return cs_locked_2s


def extract_whole_session_spectrograms(session_results_paths, lfp_data_filtered, params):
    if not os.path.exists(session_results_paths['session_spectrograms']) or params.redo_whole_session_spects:
        spectrograms_dict = compute_whole_session_spectrograms(lfp_data_filtered, params)

        with open(session_results_paths['session_spectrograms'], 'wb') as f:
            pickle.dump(spectrograms_dict, f)
    else:
        with open(session_results_paths['session_spectrograms'], 'rb') as f:
            spectrograms_dict = pd.read_pickle(f)   # compatible with older pickle formats if using pd for this

    return spectrograms_dict

def compute_hab_results_paths_dict(session_results_folder, animal_types):
    analysis_types = ['psds', 'coherences', 'imaginary_coherences']
    data_types = ['good_epochs', 'freezing', 'nonfreezing', 'seizure', 'nonseizure', 'awake', 'sleep']
    session_results_paths = {}
    for analysis_type in analysis_types:
        for data_type in data_types:
            if analysis_type == 'psds':
                paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type, 'h5')
            else:
                paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    analysis_types = ['spectrograms', 'coherograms']
    data_types = ['freezing_onsets', 'freezing_offsets', 'seizure_onsets', 'seizure_offsets']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    analysis_types = ['motion_traces']
    data_types = ['freezing_onsets', 'freezing_offsets', 'seizure_onsets', 'seizure_offsets']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    analysis_types = ['seizures']
    data_types = ['n', 'length']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    analysis_types = ['preprocessed_data']
    data_types = ['lfp', 'motion', 'ttl_events']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type, extension='h5')
            session_results_paths.update(paths)
    analysis_types = ['spectrograms']
    data_types = ['session']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    analysis_types = ['psds', '4Hz_power', '8Hz_power']
    data_types = ['good_epochs']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type, extension='h5')
            session_results_paths.update(paths)
    analysis_types = ['timings']
    data_types = ['nonseizure', 'bad_epochs', 'freezing', 'nonfreezing', 'seizure', 'sleep']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type, extension='h5')
            session_results_paths.update(paths)
    analysis_types = ['locked_traces']
    data_types = ['freezing_onsets', 'freezing_offsets', 'seizure_onsets', 'seizure_offsets']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    analysis_types = ['timings_metadata']
    data_types = ['all']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    return session_results_paths


def compute_baseline_results_paths_dict(session_results_folder, animal_types):
    analysis_types = ['psds', 'coherences', 'imaginary_coherences']
    data_types = ['good_epochs', 'freezing', 'nonfreezing', 'seizure', 'nonseizure', 'stim', 'nonstim']
    session_results_paths = {}
    for analysis_type in analysis_types:
        for data_type in data_types:
            if analysis_type == 'psds':
                paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type, 'h5')
            else:
                paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)

            session_results_paths.update(paths)
    analysis_types = ['spectrograms', 'coherograms']
    data_types = ['freezing_onsets', 'freezing_offsets', 'seizure_onsets', 'seizure_offsets', 'stim_onsets',
                  'stim_offsets']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    analysis_types = ['motion_traces']
    data_types = ['freezing_onsets', 'freezing_offsets', 'seizure_onsets', 'seizure_offsets']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    analysis_types = ['seizures']
    data_types = ['n', 'length']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    analysis_types = ['preprocessed_data']
    data_types = ['lfp', 'motion', 'ttl_events']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type, extension='h5')
            session_results_paths.update(paths)
    analysis_types = ['spectrograms']
    data_types = ['session']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    analysis_types = ['psds', '4Hz_power', '8Hz_power']
    data_types = ['good_epochs']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type, extension='h5')
            session_results_paths.update(paths)
    analysis_types = ['timings']
    data_types = ['nonseizure', 'bad_epochs', 'freezing', 'nonfreezing', 'seizure', 'sleep', 'stim', 'nonstim']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type, extension='h5')
            session_results_paths.update(paths)
    analysis_types = ['locked_traces']
    data_types = ['freezing_onsets', 'freezing_offsets', 'seizure_onsets', 'seizure_offsets']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    analysis_types = ['timings_metadata']
    data_types = ['all']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    return session_results_paths


def compute_recall_results_paths_dict(session_results_folder, animal_types):
    analysis_types = ['psds', 'coherences', 'imaginary_coherences']
    data_types = ['good_epochs', 'cs', 'noncs', 'cs_1', 'cs_2', 'cs_3', 'cs_4', 'cs_5', 'cs_6', 'cs_7', 'cs_8', 'cs_9', 'cs_10',  'noncs_1', 'noncs_2', 'noncs_3', 'noncs_4', 'noncs_5', 'noncs_6', 'noncs_7', 'noncs_8', 'noncs_9', 'noncs_10', 'noncs_11', 'firsts_cs', 'middles_cs', 'lasts_cs', 'freezing', 'nonfreezing', 'seizure',
                  'nonseizure', 'pre_cs', 'freezing_within_pre_cs', 'nonfreezing_within_pre_cs', 'freezing_within_cs',
                  'nonfreezing_within_cs', 'freezing_within_noncs', 'nonfreezing_within_noncs',
                  'freezing_within_cs_1', 'freezing_within_cs_2', 'freezing_within_cs_3', 'freezing_within_cs_4', 'freezing_within_cs_5', 'freezing_within_cs_6', 'freezing_within_cs_7', 'freezing_within_cs_8', 'freezing_within_cs_9', 'freezing_within_cs_10',
                  'freezing_within_noncs_1', 'freezing_within_noncs_2', 'freezing_within_noncs_3', 'freezing_within_noncs_4', 'freezing_within_noncs_5', 'freezing_within_noncs_6', 'freezing_within_noncs_7', 'freezing_within_noncs_8', 'freezing_within_noncs_9', 'freezing_within_noncs_10', 'freezing_within_noncs_11',
                  'nonfreezing_within_cs_1', 'nonfreezing_within_cs_2', 'nonfreezing_within_cs_3', 'nonfreezing_within_cs_4',
                  'nonfreezing_within_cs_5', 'nonfreezing_within_cs_6', 'nonfreezing_within_cs_7', 'nonfreezing_within_cs_8',
                  'nonfreezing_within_cs_9', 'nonfreezing_within_cs_10',
                  'nonfreezing_within_noncs_1', 'nonfreezing_within_noncs_2', 'nonfreezing_within_noncs_3',
                  'nonfreezing_within_noncs_4', 'nonfreezing_within_noncs_5', 'nonfreezing_within_noncs_6',
                  'nonfreezing_within_noncs_7', 'nonfreezing_within_noncs_8', 'nonfreezing_within_noncs_9',
                  'nonfreezing_within_noncs_10', 'nonfreezing_within_noncs_11',
                  'freezing_within_firsts_cs', 'nonfreezing_within_firsts_cs',
                  'freezing_within_middles_cs', 'nonfreezing_within_middles_cs',
                  'freezing_within_lasts_cs', 'nonfreezing_within_lasts_cs']
    session_results_paths = {}
    for analysis_type in analysis_types:
        for data_type in data_types:
            if analysis_type == 'psds':
                paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type, 'h5')
            else:
                paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    analysis_types = ['spectrograms', 'coherograms']
    data_types = ['freezing_within_firsts_cs_onsets', 'freezing_within_firsts_cs_offsets']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    analysis_types = ['motion_traces']
    data_types = ['freezing_onsets', 'freezing_offsets']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    analysis_types = ['lfp_traces']
    data_types = ['cs_onsets', 'led_pulses']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    analysis_types = ['seizures']
    data_types = ['n', 'length']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    analysis_types = ['preprocessed_data']
    data_types = ['lfp', 'motion', 'ttl_events']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type, extension='h5')
            session_results_paths.update(paths)
    analysis_types = ['spectrograms']
    data_types = ['session']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    analysis_types = ['psds', '4Hz_power', '8Hz_power']
    data_types = ['good_epochs']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type, extension='h5')
            session_results_paths.update(paths)
    analysis_types = ['timings']
    data_types = ['nonseizure', 'bad_epochs', 'cs', 'freezing', 'seizure', 'sleep']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type, extension='h5')
            session_results_paths.update(paths)
    analysis_types = ['locked_traces']
    data_types = ['cs_onsets', 'led_events', 'mock_led_events']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    analysis_types = ['timings_metadata']
    data_types = ['all']
    for analysis_type in analysis_types:
        for data_type in data_types:
            paths = generate_animal_analysis_path(session_results_folder, analysis_type, data_type)
            session_results_paths.update(paths)
    return session_results_paths

def update_indices(dictionary, bad_channels):
    # Remove the bad channels
    for channel in sorted(bad_channels, reverse=True):
        del dictionary[channel]

    # Update the indices of the remaining channels
    updated_dict = {}
    for i, key in enumerate(sorted(dictionary.keys())):
        updated_dict[i] = dictionary[key]

    return updated_dict


def average_dataframes(dataframes):
    dataframes = [df for df in dataframes if df is not None]

    if not dataframes: # Check if there are any dataframes left
        return None
    average_df = sum(dataframes) / len(dataframes)

    return average_df


def bad_lfp_channels_removal(lfp_filtered, bad_channels):
    good_lfp_filtered = lfp_filtered.drop(bad_channels).reset_index(drop=True)  # drop channels labeled as bad
    return good_lfp_filtered


def extract_delta_from_coherograms_all_animals(avg_coheros, lower_freq, upper_freq):
    frequencies = np.linspace(lower_freq, upper_freq, avg_coheros[list(avg_coheros.keys())[0]].shape[0])

    freq_mask = (frequencies >= 2) & (frequencies <= 6)

    results = []

    for key, coherogram in avg_coheros.items():
        # Average over the 2-6 Hz range
        avg_over_freq = coherogram[freq_mask, :].mean(axis=0)
        # Add to results as a DataFrame with sample indices
        results.append(pd.DataFrame({"sample": np.arange(avg_over_freq.shape[0]),
                                     "delta_value": avg_over_freq,
                                     "animal": key}))

    # Combine all results into one DataFrame
    delta_for_all_animals = pd.concat(results, ignore_index=True)
    return delta_for_all_animals


def extract_delta_from_spectrograms_all_animals(avg_coheros, lower_freq, upper_freq):
    frequencies = np.linspace(lower_freq, upper_freq, avg_coheros[list(avg_coheros.keys())[0]].shape[0])

    freq_mask = (frequencies >= 2) & (frequencies <= 6)

    results = []

    for key, coherogram in avg_coheros.items():
        # Average over the 2-6 Hz range
        avg_over_freq = coherogram[freq_mask, :].mean(axis=0)
        # Add to results as a DataFrame with sample indices
        results.append(pd.DataFrame({"sample": np.arange(avg_over_freq.shape[0]),
                                     "delta_value": avg_over_freq,
                                     "animal": key}))

    # Combine all results into one DataFrame
    delta_for_all_animals = pd.concat(results, ignore_index=True)
    return delta_for_all_animals


def is_safe_onset(onset, bad_intervals, freezing_intervals):
    """Check if the 5s before onset is free of any bad or freezing epochs."""
    safe_start = onset - 10000  # 5 seconds before onset

    # Check for bad epochs
    for _, row in bad_intervals.iterrows():
        bad_onset, bad_offset = row['onset'], row['offset']
        if not (safe_start >= bad_offset or onset <= bad_onset):  # Overlap check
            return False

    # Check for previous freezing epochs
    for _, row in freezing_intervals.iterrows():
        freeze_onset, freeze_offset = row['onset'], row['offset']
        if not (safe_start >= freeze_offset or onset <= freeze_onset):  # Overlap check
            return False

    return True


def filter_freezing_epochs(timings):
    freezing_df = timings['freezing']
    bad_epochs = pd.concat([timings['bad_epochs'], timings['seizure']], ignore_index=True)

    # Find the earliest CS onset #TODO pecial case for no cs sessionss
    # cs_onset_threshold = timings['cs_raw_1']['onset'].min()

    # First, filter based on safe onset
    safe_epochs = freezing_df[freezing_df['onset'].apply(lambda onset: is_safe_onset(onset, bad_epochs, freezing_df))]

    # Then, keep only those that occur after the earliest CS onset
    # safe_epochs = safe_epochs[safe_epochs['onset'] >= cs_onset_threshold]
    safe_epochs = safe_epochs[safe_epochs['offset'] - safe_epochs['onset'] >= 10000]

    return safe_epochs.reset_index(drop=True)


def pan_sessions_timings_power_connectivity_processing(good_lfp_filtered, cleaned_good_lfp_filtered, areas_animal_clean,
                                                       timings, params, session_preprocessed_data_paths,
                                                       session_results_folder):
    for time_condition in params.pan_sessions_epochs_frequency_analysis_timings:
        extract_and_save_within_timings_avg_psds(
            cleaned_good_lfp_filtered,
            timings[time_condition],
            params.sample_rate,
            session_preprocessed_data_paths['epoch_analysis']['psd'][f'{time_condition}'],
            f'psd_{time_condition}',
            redo=params.redo_pwr
        )
        extract_and_save_within_timings_coherence(
            cleaned_good_lfp_filtered,
            areas_animal_clean,
            timings[time_condition],
            params.sample_rate,
            session_preprocessed_data_paths['epoch_analysis']['coherence'][f'{time_condition}'],
            redo=params.redo_cohe
        )

    for time_condition, event_type, t_before, t_after in params.pan_sessions_time_frequency_analysis_timings:
        timing_label = (time_condition, event_type, t_before, t_after)
        formatted_timing_label = "_".join(map(str, timing_label))

        extract_and_save_avg_spectrograms(
            good_lfp_filtered,
            params.sample_rate,
            timings[time_condition],
            event_type,
            t_before,
            t_after,
            session_preprocessed_data_paths['time_frequency_analysis']['spectrogram'][formatted_timing_label],
            params.spects_decimation,
            'spectrogram_' + formatted_timing_label,
            redo=params.redo_spects
        )

        extract_and_save_avg_coherograms(
            good_lfp_filtered,
            areas_animal_clean,
            params.sample_rate,
            timings[time_condition],
            event_type,
            t_before,
            t_after,
            session_preprocessed_data_paths['time_frequency_analysis']['coherogram'][formatted_timing_label],
            params.coheros_decimation,
            redo=params.redo_coheros
        )


def baseline_session_timings_power_connectivity_processing(cleaned_good_lfp_filtered, areas_animal_clean, timings, sample_rate, session_preprocessed_data_paths, session_results_folder, time_conditions, redo_pwr, redo_cohe):
    for condition in time_conditions:
        extract_and_save_within_timings_avg_psds(
            cleaned_good_lfp_filtered,
            timings[condition],
            sample_rate,
            session_preprocessed_data_paths[f'{condition}_psds'],
            f'{condition}_psds',
            redo=redo_pwr
        )
    for condition in time_conditions:
        extract_and_save_within_timings_coherence(
            cleaned_good_lfp_filtered,
            areas_animal_clean,
            timings[condition],
            sample_rate,
            session_preprocessed_data_paths[f'{condition}_coherences'],
            redo=redo_cohe
        )


def recall_session_timings_power_connectivity_processing(good_lfp_filtered, cleaned_good_lfp_filtered,
                                                         areas_animal_clean, timings, sample_rate,
                                                         session_preprocessed_data_paths, session_results_folder,
                                                         time_conditions, spectrogram_coherogram_settings, redo_pwr,
                                                         redo_cohe, redo_spects, redo_coheros):
    for time_condition in time_conditions:
        extract_and_save_within_timings_avg_psds(
            cleaned_good_lfp_filtered,
            timings[time_condition],
            sample_rate,
            session_preprocessed_data_paths[f'{time_condition}_psds'],
            f'{time_condition}_psds',
            redo=redo_pwr
        )

    for time_condition in time_conditions:
        extract_and_save_within_timings_coherence(
            cleaned_good_lfp_filtered,
            areas_animal_clean,
            timings[time_condition],
            sample_rate,
            session_preprocessed_data_paths[f'{time_condition}_coherences'],
            redo=redo_cohe
        )

    for time_condition, event_type, t_before, t_after in spectrogram_coherogram_settings:
        extract_and_save_avg_spectrograms(
            good_lfp_filtered,
            sample_rate,
            timings[time_condition],
            event_type,
            f'{time_condition}_{event_type}',
            t_before,
            t_after,
            session_results_folder,
            10,
            redo=redo_spects
        )

    for time_condition, event_type, t_before, t_after in spectrogram_coherogram_settings:
        extract_and_save_avg_coherograms(
            good_lfp_filtered,
            areas_animal_clean,
            sample_rate,
            timings[time_condition],
            event_type,
            t_before,
            t_after,
            session_results_folder,
            10,
            redo=redo_coheros
        )


def cond_session_timings_power_connectivity_processing(good_lfp_filtered, cleaned_good_lfp_filtered, areas_animal_clean,
                                                       timings, sample_rate, session_preprocessed_data_paths,
                                                       session_results_folder, time_conditions,
                                                       spectrogram_coherogram_settings, redo_pwr, redo_cohe,
                                                       redo_spects, redo_coheros):
    for time_condition in time_conditions:
        extract_and_save_within_timings_avg_psds(
            cleaned_good_lfp_filtered,
            timings[time_condition],
            sample_rate,
            session_preprocessed_data_paths[f'{time_condition}_psds'],
            f'{time_condition}_psds',
            redo=redo_pwr
        )

    for time_condition in time_conditions:
        extract_and_save_within_timings_coherence(
            cleaned_good_lfp_filtered,
            areas_animal_clean,
            timings[time_condition],
            sample_rate,
            session_preprocessed_data_paths[f'{time_condition}_coherences'],
            redo=redo_cohe
        )

    for time_condition, event_type, t_before, t_after in spectrogram_coherogram_settings:
        extract_and_save_avg_spectrograms(
            good_lfp_filtered,
            sample_rate,
            timings[time_condition],
            event_type,
            f'{time_condition}_{event_type}',
            t_before,
            t_after,
            session_results_folder,
            10,
            redo=redo_spects
        )

    for time_condition, event_type, t_before, t_after in spectrogram_coherogram_settings:
        extract_and_save_avg_coherograms(
            good_lfp_filtered,
            areas_animal_clean,
            sample_rate,
            timings[time_condition],
            event_type,
            f'{time_condition}_{event_type}',
            t_before,
            t_after,
            session_results_folder,
            10,
            redo=redo_coheros
        )

from scipy.signal import butter
# not to be confused with butter_bandpass_filter used by seizures - possible duplication
def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def bandpass_filter_epochs(data, lowcut, highcut, fs, order=3):
    """
    Apply a bandpass filter to the data.

    Parameters:
    - data: ndarray, shape (n_epochs, n_channels, n_samples)
    - lowcut: float, low frequency cutoff
    - highcut: float, high frequency cutoff
    - fs: float, sampling frequency
    - order: int, order of the filter (default is 3)

    Returns:
    - y: ndarray, filtered data with the same shape (n_epochs, n_channels, n_samples)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    # Apply filter along the last axis (samples)
    y = np.zeros_like(data)
    for epoch_idx in range(data.shape[0]):
        for channel_idx in range(data.shape[1]):
            y[epoch_idx, channel_idx] = filtfilt(b, a, data[epoch_idx, channel_idx])

    return y


def normalize_phase_differences(phase_differences):
    """
    Normalize phase differences to the range [0, 360 degrees].

    Parameters:
    - phase_differences: ndarray, shape (n_differences,)

    Returns:
    - normalized_phases: ndarray, shape (n_differences,)
    """
    return (phase_differences % (2 * np.pi))
