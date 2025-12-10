"""
    Dual band peaks seizure detection algorithm and interface
    Copyright (C) 2025 Domagoj Anticic and Paul Rignanese, Kind Lab, The University of Edinburgh

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

import sys
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
from scipy.signal import find_peaks

from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.seizures_param import f_band_low, f_band_high, f_hf_cut, merge_gap, short_cutoff, window_power, threshold_power_factor, use_power_threshold, pad_power_duration
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.seizures_param import periods_len_minutes_seizure, peak_dist, verification_padding_time
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.seizures_param import verify_seizures, live_constant_adjust, restart_after_adjust, constant_saving_mode, remove_drift
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.seizures_param import save_seizure_spreadsheet
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.seizures_param import (exclude_bad_from_input_sig, start_idx_of_screening, end_idx_of_screening, merge_and_filter,
                                                  remove_bad_from_detected, no_user_input)

from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.seizure_signal_processing import butter_bandpass_filter, butter_highpass_filter, average_power
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.seizure_timings_processing import states_from_starts_ends, starts_ends_from_states, merge_close_starts_ends
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.seizure_timings_processing import merge_close_epochs, filter_short_starts_ends
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.seizure_timings_processing import plot_intervals, good_periods_mask, asserts_starts_ends

from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.utilities import (split_signal_into_frames, screening_start_end_idx_asserts,
                                             load_backup, starts_ends_invert_and_to_pandas, write_seizures_nonseizures_to_file,
                                             save_backup)
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.utilities import Constants

matplotlib.use('TkAgg')

class seizureVerifier:
    """Window for seizure verification"""
    def __init__(self, ax, middle, edge1, edge2, prev, next_):
        """Main window initialisation

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Ax object to plot on
        middle: float
            Time in middle of seizure
        edge1: float
            Left boundary of seizure
        edge2: float
            Right boundary of seizure
        prev: float
            End of previous seizure
        next_: float
            Start of next seizure
        """
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self.ax = ax
        self.floating_boundary = None
        self.floating_boundary_left = None
        self.floating_boundary_right = None
        self.changes = False
        self.discard = False
        #self.ax.axvline(edge1, color='green', linewidth=2.0)
        #self.ax.axvline(edge2, color='green', linewidth=2.0)
        ax.axvspan(edge1, edge2, color='red', alpha=0.35)
        if prev != None:
            self.ax.axvline(prev, color='orange', linestyle='--', linewidth=2.0, label='previous seizure end')
        if next_ != None:
            self.ax.axvline(next_, color='blue', linestyle='--', linewidth=2.0, label='next seizure start')
        self.ax.axvline(middle, color='gray', linewidth=0.8)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.ax.figure.canvas.mpl_connect('key_press_event', self.on_press)

    def on_hover(self, event):
        """Show dashed line when hovering

        Parameters
        ----------
        event : matplotlib.event.Event
        """
        if event.inaxes == self.ax:
            self.floating_boundary = float(event.xdata)

            # continually removes previous line (should never be none)
            if self.floating_boundary is not None:
                self.ax.lines[-1].remove()  # Remove previous line

            self.ax.axvline(self.floating_boundary, color='grey', linestyle='--')
            self.ax.figure.canvas.draw()

    def on_click(self, event):
        """Handle both left and right click (start and end boundary)

        Parameters
        ----------
        event : matplotlib.event.Event
        """
        if event.inaxes == self.ax:
            self.floating_boundary = event.xdata
            # modify start boundary
            if event.button is MouseButton.LEFT:
                # delete existing hover line
                if self.floating_boundary is not None:
                    self.ax.lines[-1].remove()
                # delete old line
                for i in range(len(self.ax.lines)):
                    if plt.getp(self.ax.lines[i], 'label') == 'start boundary':
                        self.ax.lines[i].remove()
                        break
                # draw new line
                self.floating_boundary_left = self.floating_boundary
                self.ax.axvline(self.floating_boundary_left, color='red', linestyle='--', label='start boundary')

            # modify end boundary
            if event.button is MouseButton.RIGHT:
                # delete existing hover line
                if self.floating_boundary is not None:
                    self.ax.lines[-1].remove()
                # delete old line
                for i in range(len(self.ax.lines)):
                    if plt.getp(self.ax.lines[i], 'label') == 'end boundary':
                        self.ax.lines[i].remove()
                        break
                # draw new line
                self.floating_boundary_right = self.floating_boundary
                self.ax.axvline(self.floating_boundary_right, color='magenta', linestyle='--', label='end boundary')

            self.ax.axvline(self.floating_boundary, color='grey', linestyle='--', label='y boundary')
            self.ax.figure.canvas.draw()

    def on_press(self, event):
        """Handle keystrokes

        Parameters
        ----------
        event : matplotlib.event.Event
        """
        sys.stdout.flush()
        # Close without saving
        if event.key == 'x':
            self.discard = True
            print("  > Discard")
            plt.close()
        # Close with saving
        elif event.key == 'y':
            # if no modifications were made
            if self.floating_boundary_left is None and self.floating_boundary_right is None:
                print("  > Accept WITHOUT modifications (none made by user)")
            else:
                self.changes = True
                print("  > Accept WITH modifications")
            plt.close()
        # Reset boundaries
        elif event.key == 'r':
            self.floating_boundary = None
            self.floating_boundary_left = None
            self.floating_boundary_right = None
            self.changes = False
            self.discard = False
            print("  > Accept WITHOUT modifications CHANGES RESET")
            plt.close()


class SeizureViewAll:
    """Handling of activation of constant changing window in main seizure window"""
    def __init__(self, axs):
        """Initialisation

        Parameters
        ----------
        axs: matplotlib.axes.Axes
        """
        self.changes = False
        self.reject_all = False
        self.accept_all = False
        for ax in axs:
            ax.figure.canvas.mpl_connect('key_press_event', self.on_press)

    def on_press(self, event):
        """Handle keystrokes

        Parameters
        ----------
        event : matplotlib.event.Event
        """
        sys.stdout.flush()
        # if constant window opened
        if event.key == 'c':
            if live_constant_adjust:
                self.changes = True
                print("Changing constants")
                plt.close()
        if event.key == 'k':
            print("REJECT ALL SEIZURES")
            self.reject_all = True
            plt.close()
        if event.key == 'l':
            print("ACCEPT ALL SEIZURES")
            self.accept_all = True
            plt.close()
        # if window closed
        if event.key == 'y' or event.key == 'x':
            print("Moving on...")
            plt.close()

def variance_of_differences(x_times, peaks, window_size):
    """Compute variances of time differences using rolling window

    Parameters
    ----------
    x_times: numpy.ndarray
        Time domain
    peaks: numpy.ndarray
        Peak times
    window_size: int
        Rolling window size

    Returns
    -------
    var_diff: numpy.ndarray
        Variance of rolling window average of time differences between peaks
    """
    # if there are more peaks than the window size
    if len(peaks) > window_size:
        diff = np.diff(x_times[peaks])  # Difference between peak times
        Aw = np.lib.stride_tricks.sliding_window_view(diff, window_size)        # Rolling window
        var_diff = np.var(Aw, axis=-1)  # Variance of rolling window
    else:
        #in case data is shorter than window (sometimes at file end)
        var_diff = np.array([])
    return var_diff

def verficiation_message():
    """Message to print upon using seizure verification"""
    print("Verification of identified seizures:")
    print("Showing identified seizures with padding, green lines indicate identified seizure boundaries")
    print("If present, a dashed orange line indicates previous seizure end, a dashed blue line indicates start of next seizure in window\n")
    print("press X - discard identified seizure")
    print("press Y - accept seizure as is or with modifications" )
    print("press R - disregard modifications and accept original")
    print("left click - set new start time, red line")
    print("right click - set new end time, magenta line\n")
    print("Each new boundary's last drawn version is taken as final")

def low_var_region(var_diff, x_times, peaks, sample_rate, threshold, spread, window_size, edge):
    """Detect region of low variance of time difference between peaks in peak domain, and convert it to time domain

    Parameters
    ----------
    var_diff: numpy.ndarray
        Variance of rolling window average of time differences between peaks
    x_times: numpy.ndarray
        Time domain
    peaks: numpy.ndarray
        Peak times
    sample_rate: float
        Sampling rate of signal
    threshold: float
        Threshold BELOW which the variance is considered low enough for detection
    spread: float
        Time to pad around detected low variance region
    window_size: int
        Rolling window size
    edge: int
        1/2 of window_size - amount of peaks to account for at edges due to rolling window size

    Returns
    -------
    seizure_states: numpy.ndarray
        Boolean array indicating seizures (1 = seizure, 0 = no seizure)
    """
    # signal with peaks and times and time indices:

    #       x            x             x          x
    #       |            |             |          |
    #       |   ||||     ||    ||      |          |
    #|||||||||||||||||||||||||||||||||||||||||||||||||||||

    #       t            t             t          t
    #       1            2             3          4

    # True where below threshold, in variance domain
    seizure_states = np.where(var_diff < threshold, True, False)
    if len(seizure_states) > 0:
        # no + 1 on right as we want to get the same amount of peaks included left and right of the seizure
        starts, ends = starts_ends_from_states(seizure_states, 0)

        # account for window size as we are using a rolling average
        starts = starts - edge
        ends = ends + edge

        # prevent index error in var domain - clip if start or end outside of list
        # we cannot access a peak no. which does not exist in var domain (for max)      # todo use pad_states
        # truncation in next part "realigns" the arrays
        starts = np.clip(starts, 0, len(peaks)-1-window_size)
        ends = np.clip(ends, 0, len(peaks)-1-window_size)

        # go into time domain from peaks found by variances
        start_indeces = peaks[edge:-edge][starts] - int(spread*sample_rate)
        end_indeces = peaks[edge:-edge][ends] + int(spread*sample_rate)

        # prevent index error in time domain
        start_indeces = np.clip(start_indeces, 0, None)             # todo use pad_states
        end_indeces = np.clip(end_indeces, None, len(x_times) - 1)

        # get seizure states
        seizure_states = states_from_starts_ends(start_indeces, end_indeces, len(x_times))
        #seizure_states = np.zeros(len(x_times))
        #np.add.at(seizure_states, start_indeces, 1)
        #np.add.at(seizure_states, end_indeces, -1)     # +1 ?

        # calculate the cumulative sum to fill in the values between start and end indices
        #seizure_states = np.cumsum(seizure_states)
    else:
        seizure_states = np.zeros(len(x_times))

    return seizure_states


def ref_val_peaks(signal, quantile):
    """Get threshold for peak detection based on quantile of all detected peaks without threshold

    Parameters
    ----------
    signal: numpy.ndarray
        Signal for detection, usually filtered for a band of interest
    quantile: float
        Quantile of peak height distribution from which the peak threshold is taken

    Returns
    -------
    threshold: float
        Threshold to be used for more advanced peak detection
    """
    val = np.quantile(find_peaks(signal, height=0)[1]["peak_heights"], quantile)
    return val

def ref_val_values(signal, factor):
    """Get threshold for peak detection based on quantile of all signal points

    Parameters
    ----------
    signal: numpy.ndarray
        Signal for detection, usually filtered for a band of interest
    quantile: float
        Quantile of peak height distribution from which the peak threshold is taken

    Returns
    -------
    threshold: float
        Threshold to be used for more advanced peak detection
    """
    low = np.nanquantile(signal, 0.01)
    high = np.nanquantile(signal, 0.99)
    print(low)
    print(high)
    val = np.mean(np.array([low,high]))*factor
    return val

def histogram(sig_band, sig_hf, band_thres, hf_thres):
    """Plot histogram of detected peaks, used in testing
    UNUSED
    """
    bins = int(len(sig_band)/10)
    hist_band, bin_edges_band = np.histogram(find_peaks(sig_band)[1]["peak_heights"], bins=bins)
    hist_hf, bin_edges_hf = np.histogram(find_peaks(sig_hf)[1]["peak_heights"], bins=bins)
    plt.bar(bin_edges_band[:-1], hist_band, width=1, alpha=0.25)
    plt.bar(bin_edges_hf[:-1], hist_hf, width=1, alpha=0.25)
    plt.axvline(hf_thres, c="r", label="Highpass threshold")
    plt.axvline(band_thres, c="b", label="Bandpass threshold")
    plt.xlabel("Peak height/a.u.")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

def time_to_idx(time_axis, time):
    """Given a specific time in a time series, return index of element closest to value

    Parameters
    ----------
    time_axis: np.array
        Array of times
    time: np.array
        Specific time within limits of time axis

    Returns
    -------
    idx: int
        Index of element closest to given value
    """
    idx = abs((time_axis - float(time))).argmin()
    return idx

def ensure_no_overlaps(starts, ends, length, raise_error):
    """Ensure no overlaps present in data and return nonseizure periods.

    Create binary mask using starts and ends, thereby ensuring no overlaps when converted back to starts and ends.

    Parameters
    ----------
    starts: np.array
        Start times of seizures
    ends: np.array
        End times of seizures
    length: int
        Length of array in sample number
    raise_error: boolean
        Whether to raise an error if an overlap exists

    Returns
    -------
    starts2: np.array
        Interval start samples
    ends2: np.array
        Interval end samples
    """
    # creating a mask will flatten all starts and ends to one binary state array
    mask = states_from_starts_ends(starts, ends, length)

    starts2, ends2 = starts_ends_from_states(mask, 0)

    delta = len(starts) - len(starts2)
    if delta == 0:
        print("No overlapping periods found")
    else:
        print(f"! NOTICE ! : There were {delta} overlapping periods, now merged!")
        if raise_error:
            raise ValueError("Overlapping periods were found, but no verification was done - something went wrong with code!")

    # It should never happen that starts and ends change if manual verification wasn't done - raise error
    if raise_error:
        if not np.all(starts == starts2) or not np.all(ends == ends2):
            raise ValueError("Starts and ends before ensuring no overlap do not match and no manual verification was done - something went wrong with code!")

    assert delta >= 0 # there should never be more intervals after merging unless something has gone wrong

    return starts2, ends2

def merge_and_filter_states(states, merge_gap, short_cutoff, sample_rate):
    """Given states, merge short gap and then filter short seizures

    Parameters
    ----------
    states: np.array
        Binary states of intervals
    merge_gap: float
        Largest gap to merge
    short_cutoff: float
        Longest seizure to keep
    sample_rate: float

    Returns
    -------
    starts: np.array
        Start of merged and filtered intervals
    ends: np.array
        Ends of merged and filtered intervals
    """
    starts, ends = starts_ends_from_states(states,0)
    asserts_starts_ends(starts, ends, len(states))  # overkill but better safe than sorry
    starts, ends = merge_close_starts_ends(starts, ends, merge_gap*sample_rate)
    asserts_starts_ends(starts, ends, len(states))
    starts, ends = filter_short_starts_ends(starts, ends, short_cutoff*sample_rate)
    asserts_starts_ends(starts, ends, len(states))
    return starts, ends

def seizure_verification_manual(sig, x_times, starts, ends, verification_padding, animal_folder, animal_id, start_idx, sample_rate):
    """Open manual verification window where seizures can be accepted, rejected or modified.

    All processing done on frame basis, so indices are in reference to frame start. start_idx passed for print output only,
    internal calculations remain in reference to frame start.

    Parameters
    ----------
    sig: np.array
        Signal where seizures are being detected from
    x_times: np.array
        Time axis of signal
    starts: np.array
        Detected seizure starts
    ends: np.array
        Detected seizure ends
    verification_padding: float
        Interval to show around detected seizure
    animal_folder: str
        Folder where animal data is loaded from (for label)
    animal_id: str
        Animal ID (for label)
    start_idx: int
        Start index of frame relative to whole file
    sample_rate: float

    Returns
    -------
    starts: np.array
        Verified event starts
    ends: np.array
        Verified event ends
    """
    starts = starts.copy()
    ends = ends.copy()
    to_delete = []
    for j in range(len(starts)):
        fig, ax = plt.subplots(figsize=(20, 15))
        # calculate section times and signal
        s0 = (starts[j] - verification_padding)
        s2 = (ends[j] + verification_padding)
        # check that it is not out of bounds
        if s0 < 0:
            s0 = 0
        if s2 > len(sig) - 1:
            s2 = len(sig) - 1
        t = x_times[s0: s2]
        sig_mini = sig[s0: s2]

        # if previous seizure end inside window exits
        if j > 0 and len(t) > 0:
            prev_0 = x_times[ends[j - 1]]
            # if within bounds of current plot
            if (prev_0 > t[0]) and (prev_0 < t[-1]):
                prev = prev_0
            else:
                prev = None
        else:
            prev = None

        # if next seizure start inside window exists
        if len(starts) > j + 1 and len(t) > 0:
            next_0 = x_times[starts[j + 1]]
            # if within bounds of current plot
            if (next_0 > t[0]) and (next_0 < t[-1]):
                next_ = next_0
            else:
                next_ = None
        else:
            next_ = None

        # find timings
        middle = int((x_times[starts[j]] + x_times[ends[j]]) / 2)
        start = x_times[starts[j]]
        stop = x_times[ends[j]]

        # plot segment around detected seizure
        print(f"Seizure #{j} ({starts[j]+start_idx} to {ends[j]+start_idx} [samples], "
              f"{(starts[j]+start_idx)/sample_rate} to {(ends[j]+start_idx)/sample_rate} [s]):")
        plt.plot(t, sig_mini)
        selector = seizureVerifier(ax, middle, start, stop, prev, next_)
        # if legend is needed (if not needed a warning will show up)
        if prev is not None or next_ is not None:
            plt.legend()  # does this work?
        # write metadata
        if animal_folder is not None and animal_id is not None:
            msg = ("ID = " + animal_id + ", Folder = " + animal_folder
                   + ", window N = " + str(j))
            plt.title(msg)
        plt.show()

        # if seizure discarded
        if selector.discard == True:
            to_delete.append(j)
        # if accepted (not discarded)
        else:
            # if times modified
            if selector.changes == True:
                # get left boundary idx
                if selector.floating_boundary_left is not None:
                    # find index corresponding to value from plot
                    left = time_to_idx(x_times, selector.floating_boundary_left)
                    print(f"  > New start boundary = {left+start_idx} [samples], {(left+start_idx)/sample_rate} [s]")
                else:
                    # use original start
                    left = starts[j]

                # get right boundary idx
                if selector.floating_boundary_right is not None:
                    right = time_to_idx(x_times, selector.floating_boundary_right)
                    print(f"  > New end boundary = {right+start_idx} [samples], {(right + start_idx)/sample_rate} [s]")
                else:
                    # use original end
                    right = ends[j]

                # if order swapped, take the maximum range from the original and modified ones
                if right < left:
                    left_new = min([left, right, starts[j], ends[j]])
                    right_new = max([left, right, starts[j], ends[j]])
                    left = left_new
                    right = right_new
                    print("  > !! NOTICE !!: start is bigger than end. This is undefined. Start set to minimum of original and modified values. End set to maximum of original and modified values.")
                    print("  > !! NOTICE !!: this may not be the behaviour you expected.")
                    print(f"  > Corrected start boundary = {left+start_idx} [samples], {(left + start_idx) / sample_rate} [s]")
                    print(f"  > Corrected end boundary = {right+start_idx} [samples], {(right + start_idx)/sample_rate} [s]")
                    input(f"  > Click enter to acknowledge change")
                    # write new or overwrite with same value
                starts[j] = left
                ends[j] = right
                assert right > left

    # delete discarded sections
    starts = np.delete(starts, to_delete)
    ends = np.delete(ends, to_delete)
    return starts, ends

def plotting(animal_folder, session_folder, edge, sig, interval_band, interval_hf, interval_power, interval_times,
             spect_interval, peaks_band, peaks_hf, var_diff_hf,  var_diff_band, threshold_band, threshold_hf, threshold_power,
             live_adjustement, starts, ends, start_idx, sample_rate):
    """Plots single frame of seizure detection, including raw signal, highpass, bandpass, detected peaks, computed variances"""
    # PLOTTING
    title = "Animal = " + str(animal_folder) + ", session = " + str(session_folder)
    fig, axs = plt.subplots(4, sharex=True, figsize=(20, 10), num=title)
    ax1_2 = axs[1].twinx()
    ax2_2 = axs[2].twinx()
    ax3_2 = axs[3].twinx()
    ax3_3 = axs[3].twinx()

    # lims for plotting
    lim_band = np.quantile(abs(interval_band), 0.999) * 1.1
    lim_hf = np.quantile(abs(interval_hf), 0.999) * 1.1
    lim_sig = np.nanquantile(abs(sig), 0.9999) * 1.1

    # spectrogram
    if spect_interval is not None:
        axs[0].imshow(spect_interval, cmap='inferno', aspect='auto', origin='lower',
                      extent=[interval_times[0], interval_times[-1], 0, spect_interval.shape[0]], vmin=-30, vmax=30)
    axs[0].set_ylabel('Frequency/Hz')
    axs[0].set_title("Spectrogram")

    # Bandpass
    axs[1].plot(interval_times, interval_band, alpha=1, linewidth=0.5, color='b')
    axs[1].plot(interval_times[peaks_band], interval_band[peaks_band], "x", color='r')  # peaks
    axs[1].set_ylim(-lim_band, lim_band)
    axs[1].set_title("Bandpass")
    axs[1].set_ylabel('Amplitude/a.u.')
    # Bandpass variance
    ax1_2.plot(interval_times[peaks_band][edge:-edge], var_diff_band, c='g')
    ax1_2.set_ylim(-0.1, 2)
    ax1_2.axhline(threshold_band, linestyle='--', c='g')
    ax1_2.set_ylabel("Peak period variance", color="g")
    ax1_2.tick_params(axis='y', labelcolor='g')

    # Highpass
    axs[2].plot(interval_times, interval_hf, alpha=1, linewidth=0.5, color='b')
    axs[2].set_ylim(-lim_hf, lim_hf)
    axs[2].plot(interval_times[peaks_hf], interval_hf[peaks_hf], "x", color='r')  # peaks
    axs[2].set_title("Highpass")
    axs[2].set_ylabel('Amplitude/a.u.')
    # highpass variance
    ax2_2.plot(interval_times[peaks_hf][edge:-edge], var_diff_hf, c='g')
    ax2_2.set_ylim(-0.1, 1)
    ax2_2.axhline(threshold_hf, linestyle='--', c='g')
    ax2_2.set_ylabel("Peak period variance", color="g")
    ax2_2.tick_params(axis='y', labelcolor='g')

    # Raw signal
    axs[3].set_ylim(-lim_sig, lim_sig)
    axs[3].plot(interval_times, sig, alpha=1, linewidth=0.5, color='b')
    axs[3].set_title("Raw signal")
    axs[3].set_xlabel('Time/s')
    axs[3].set_ylabel('Amplitude/a.u.')

    # Plot power
    if interval_power is not None:
        ax3_2.plot(interval_times, interval_power, alpha=1, linewidth=0.5, color='magenta')
        ax3_2.set_ylabel('Rolling Power/a.u.', color='magenta')
        ax3_2.axhline(threshold_power, linestyle='--', c='magenta')
        ax3_2.tick_params(axis='y', labelcolor='magenta')


    # Plot intervals
    ax3_3.set_axis_off()
    plot_intervals(ax3_3, starts + start_idx, ends + start_idx, sample_rate, 'ro-', 0, alpha=0.8, linewidth=3)
    #plot_intervals(ax3_3, pstart + start_idx, pend + start_idx, sample_rate, 'mo-', 1, alpha=0.8, linewidth=3)

    # todo: Plot all intervals

    # show until window interaction
    print("Press X or Y to continue")
    if live_adjustement == True:
        print("Press C to change constants (press while in plotting window)")
    print("Press K to reject all seizures shown")
    print("Press L to accept all seizures (no manual verification even if enabled)")

    return axs

def pad_states(states, length, pad_samples):
    starts, ends = starts_ends_from_states(states, 0)
    starts = starts - pad_samples
    ends = ends + pad_samples
    starts = np.clip(starts, 0, length - 1)
    ends = np.clip(ends, 0, length - 1)
    states2 = states_from_starts_ends(starts, ends, length, 0)
    return states2

def dual_band_peaks_analysis(input_signal, spectrogram, bad_periods, sample_rate, base_dir, animal_folder, session_folder, animal_id,
                             verification=verify_seizures, live_adjustement=live_constant_adjust,
                             restart_file=restart_after_adjust, constant_saving_mode=constant_saving_mode,
                             excel_dir=save_seizure_spreadsheet,
                             exclude_bad_from_input_sig=exclude_bad_from_input_sig, start_idx_of_screening = start_idx_of_screening,
                             end_idx_of_screening = end_idx_of_screening, merge_and_filter=merge_and_filter,
                             remove_bad_from_detected=remove_bad_from_detected, backup_file=None, no_user_input=no_user_input,
                             use_power_threshold=use_power_threshold):
    """Detect seizures using dual band peak method, described in detail in README

    Parameters
    ----------
    input_signal : np.ndarray
        Signal from which to detect seizures
    spectrogram : np.ndarray
        Spectrogram to show with seizures, but no data processing is done on this
    bad_periods : pd.dataframe
        Dataframe with columns "onset" and "offset" of bad periods in data. Pass None if you do not wish to use this
    sample_rate : float
        Sampling rate of signal
    base_dir : string
        Directory where top-level folder is located, containing animal folders
    animal_folder : string
        Folder where animal-level data is to be stored for the given data
    session_folder : string
        Folder where session data is to be stored, within animal folder
    animal_id : string
        Animal ID, for labeling purposes
    verification : boolean
        Optional manual seizure verification
    live_adjustement : boolean
        Optional constant adjustment window
    restart_file : boolean
        Optional restart of seizure detection after live constant adjustment
    constant_saving_mode : string
        Optional constant saving
    excel_dir : string
        Optional saving of detected seizures and nonseizures to spreadsheet
    exclude_bad_from_input_sig: boolean
        Optional exclusion of bad intervals from input signal before any processing is done
    start_idx_of_screening: int
        Index at which to start screening at, note that quantiles still computed from entire data passed
    end_idx_of_screening: int
        Index at which to end screening at, note that quantiles still computed from entire data passed
    merge_and_filter: boolean
        Optional merging of close and filtering of short detected seizures
    remove_bad_from_detected: boolean
        Optional removal of bad intervals from detected seizures, which could happen due to edge effects and merging
    backup_file: string
        Optional backup file location where detected seizures saved after every frame
    no_user_input: boolean
        Option to completely remove user input and go through all the data, overriding any seizure verification settings

    Returns
    -------
    seizure_periods_all : pd.DataFrame
        All detected seizure intervals, with "onset" and "offset" columns
    nonseizure_periods_all : pd.DataFrame
        Nonseizures, extrapolated from detected seizures, with "onset" and "offset" columns
    """
    matplotlib.use('TkAgg')
    print(""" Dual band peaks seizure detection. Copyright (C) 2025 Domagoj Anticic and Paul Rignanese """)

    # Initialise detected seizures
    all_seizure_starts = np.array([])
    all_seizure_ends = np.array([])

    verification_padding = int(verification_padding_time * sample_rate) # no of samples - must be int!
    power_window_samples = int(window_power * sample_rate)
    window_length = periods_len_minutes_seizure*60*sample_rate

    # prompt to load backup if exists
    backup_file, start_idx_of_screening, backup_start_of_screening, all_seizure_starts, all_seizure_ends = (
        load_backup(backup_file, start_idx_of_screening, end_idx_of_screening, all_seizure_starts, all_seizure_ends,
                    len(input_signal), window_length))

    # load constants
    constants = Constants(base_dir, constant_saving_mode, animal_folder=animal_folder, session_folder=session_folder)

    # make sure data is not modified
    signal = input_signal.copy()

    screening_start_end_idx_asserts(start_idx_of_screening, end_idx_of_screening, len(input_signal), window_length)
    frame_number, samples_per_frame = split_signal_into_frames(start_idx_of_screening, end_idx_of_screening, len(signal),
                                                               sample_rate, periods_len_minutes_seizure)

    # create mask of all not bad (good) periods
    good_mask = good_periods_mask(bad_periods, len(signal))
    if exclude_bad_from_input_sig:
        signal[~good_mask] = np.nan

    # in case there is a dc offset, just replacing nans with zeroes will lead to high frequency peaks
    # so we want to remove low frequency signals before replacing nan with zero
    if remove_drift:
        print("Removing drift and DC offset...")
        # remove average - done to minimise sharp transition when padding to zero
        avg = np.nanmean(signal)
        print("Average = " + str(avg))
        signal = signal - avg
        # pad nan to zero
        signal = np.nan_to_num(signal)
        # highpass
        signal = butter_highpass_filter(signal, 1, sample_rate)

    else:
        signal = np.nan_to_num(signal)

    # extract bands from data
    sig_band = butter_bandpass_filter (signal, f_band_low, f_band_high, sample_rate)
    sig_hf = butter_highpass_filter(signal, f_hf_cut, sample_rate)

    #find heights for peak finding
    height_band = ref_val_peaks(sig_band, constants.quantile_band)
    height_hf = ref_val_peaks(sig_hf, constants.quantile_hf)
    print("Bandpass height = " + str(height_band))
    print("Highpass height = " + str(height_hf))

    # power
    if use_power_threshold:
        power = average_power(signal, power_window_samples)
        power = np.pad(power, power_window_samples//2, constant_values=(np.nan, np.nan))    # pad edges as window causes size to decrease
        threshold_power = ref_val_values(power, threshold_power_factor)
        #assert len(power) == len(signal)           TODO power is longer than signal by 1, so no problem in slicing, but should be addressed here for even and odd case
    else:
        threshold_power = None

    #histogram(sig_band, sig_hf, height_band, height_hf)

    # create time axis
    times_all = np.linspace(0, (len(sig_band)-1)/sample_rate, num=len(sig_band))

    # detect spectrogram decimation
    if spectrogram is not None:
        decimate = len(sig_band) / spectrogram.shape[1]

    # process each interval
    start_frame = 0
    i = start_frame
    # If boundaries of screening not specified, go through whole file
    if start_idx_of_screening is None:
        start_idx_of_screening = 0
    if end_idx_of_screening is None:
        end_idx_of_screening = len(signal)
    while i < frame_number:
        # start and end samples
        start_idx = int(start_idx_of_screening + i * samples_per_frame)
        end_idx = int(start_idx_of_screening + (i + 1) * samples_per_frame)
        # handle case where there is a set end of screening, to cut all analysis and screening end
        if end_idx > end_idx_of_screening:
            end_idx = end_idx_of_screening
        assert start_idx >= start_idx_of_screening and end_idx <= end_idx_of_screening

        # extract the corresponding spectrogram data
        if spectrogram is not None:
            spect_interval = spectrogram[:, int(start_idx / decimate):int(end_idx / decimate)]
        else:
            spect_interval = None

        # extract segement of time and signal
        interval_band = sig_band[start_idx:end_idx]
        interval_hf = sig_hf[start_idx:end_idx]
        interval_times = times_all[start_idx:end_idx]

        interval_signal = signal[start_idx:end_idx]
        interval_good_mask = good_mask[start_idx:end_idx]
        if use_power_threshold:
            interval_power = power[start_idx:end_idx]
        else:
            interval_power = None

        # find peaks in hf and bandpass
        peaks_band, bp_properties = find_peaks(interval_band, height=height_band, distance=peak_dist*sample_rate)
        peaks_hf, hfp_properties = find_peaks(interval_hf, height=height_hf, distance=peak_dist*sample_rate)

        # find rolling variance of differences between peak times
        var_diff_band = variance_of_differences(interval_times, peaks_band, constants.window_size)
        var_diff_hf = variance_of_differences(interval_times, peaks_hf, constants.window_size)

        # extract low variance region and power
        seizure_states_band = low_var_region(var_diff_band, interval_times, peaks_band, sample_rate, constants.threshold_band,
                                             constants.spread, constants.window_size, constants.edge)
        seizure_states_hf = low_var_region(var_diff_hf, interval_times, peaks_hf, sample_rate, constants.threshold_hf,
                                           constants.spread, constants.window_size, constants.edge)

        # find seizures and return binary mask of their states
        seizure_states= np.logical_and(seizure_states_band, seizure_states_hf)
        # exclude bad periods (detection may happen due to zero padding artefacts or if there is a gap at end of file (no variances)
        if remove_bad_from_detected:    # done by default
            seizure_states = np.logical_and(seizure_states, interval_good_mask)
        if use_power_threshold:
            power_states = np.where(interval_power > threshold_power, True, False)
            power_states = pad_states(power_states, len(interval_times), pad_power_duration*sample_rate)
            seizure_states = np.logical_and(seizure_states, power_states)

        if merge_and_filter:
            starts, ends = merge_and_filter_states(seizure_states, merge_gap, short_cutoff, sample_rate)
        else:
            starts, ends = starts_ends_from_states(seizure_states, 0)

        # if showing interface
        if not no_user_input:
            # plotting
            axs = plotting(animal_folder, session_folder, constants.edge, interval_signal, interval_band, interval_hf, interval_power, interval_times,
                            spect_interval, peaks_band, peaks_hf, var_diff_hf,  var_diff_band, constants.threshold_band, constants.threshold_hf,
                            threshold_power, live_adjustement, starts, ends, start_idx, sample_rate)
            seizure_all_window = SeizureViewAll(axs)
            plt.show()

            # if parameters are to be changed
            if live_adjustement and seizure_all_window.changes:
                print("Press Y to accept changes, press X to reject changes")
                changes_state = constants.adjust_window()
                # if parameters adjusted and changes accepted
                if changes_state == True:
                    # find heights for peak finding
                    height_band = ref_val_peaks(sig_band, constants.quantile_band)
                    height_hf = ref_val_peaks(sig_hf, constants.quantile_hf)
                    print("Bandpass height = " + str(height_band))
                    print("Highpass height = " + str(height_hf))

                    # restarts loop or only current item; i unchanged or reset
                    if restart_file == True:
                        i = start_frame
                        continue

        # if no constants are changed, finish program and then continue to next loop
        # this part has idx relative to current window, not total data
        if not no_user_input:
            if seizure_all_window.reject_all:
                starts = np.array([])
                ends = np.array([])
            elif verification and not seizure_all_window.accept_all:
                verficiation_message()
                starts, ends = seizure_verification_manual(interval_signal, interval_times, starts, ends, verification_padding, animal_folder, animal_id, start_idx, sample_rate)
            # else case - all seizures accepted as is without manual verification

        # Make sure no intervals overlap, merge if so. This could happen if user verification redefines it as such
        starts, ends = ensure_no_overlaps(starts, ends, len(interval_times), not verification)

        # verification
        asserts_starts_ends(starts, ends, len(interval_times))

        # if any seizures were present
        if len(starts) > 0 and len(ends) > 0:
            # note: starts and ends referred to beginning of window, not overall length, so this has to be accounted for
            all_seizure_starts = np.concatenate((all_seizure_starts, start_idx+starts), axis=0)
            all_seizure_ends = np.concatenate((all_seizure_ends, start_idx+ends), axis=0)
            print("New seizure starts:")
            print(start_idx+starts)
            print("New seizure ends:")
            print(start_idx+ends)

        # no seizures present - no change

        i += 1

        if backup_file is not None:
            save_backup(backup_file, all_seizure_starts, all_seizure_ends, backup_start_of_screening, start_idx_of_screening, end_idx)

    # find nonseizures, convert seizures and nonseizures to pandas and save
    # if loading from backup, earlier seizure start is there, if from this file, earlier seizure start as specified here
    if backup_start_of_screening is not None:
        absolute_start_idx = backup_start_of_screening
    else:
        absolute_start_idx = start_idx_of_screening

    asserts_starts_ends(all_seizure_starts, all_seizure_ends, end_idx_of_screening, start_idx=absolute_start_idx)

    seizure_intervals_all, nonseizure_intervals_all = starts_ends_invert_and_to_pandas(all_seizure_starts,
                                all_seizure_ends, absolute_start_idx, end_idx_of_screening)

    if excel_dir is not None:
        write_seizures_nonseizures_to_file(excel_dir, "_without_window_merging",
                                           seizure_intervals_all, nonseizure_intervals_all)

    # merge close seizures, find nonseizures, convert seizures and nonseizures to pandas and save
    all_seizure_starts_processed, all_seizure_ends_processed = merge_close_starts_ends(all_seizure_starts, all_seizure_ends, 10)   # samples

    seizure_intervals_all_processed, nonseizure_intervals_all_processed = starts_ends_invert_and_to_pandas(
                                            all_seizure_starts_processed, all_seizure_ends_processed,
                                            absolute_start_idx, end_idx_of_screening)
    print(len(nonseizure_intervals_all_processed['onset']))

    assert (len(nonseizure_intervals_all_processed['onset']) == len(seizure_intervals_all_processed['onset']) or
            len(nonseizure_intervals_all_processed['onset']) + 1 == len(seizure_intervals_all_processed['onset']) or
            len(nonseizure_intervals_all_processed['onset']) == len(seizure_intervals_all_processed['onset']) + 1)      # todo write this with difference abs <=1

    if excel_dir is not None:
        write_seizures_nonseizures_to_file(excel_dir, "_merged_window_boundaries",
                                           seizure_intervals_all_processed, nonseizure_intervals_all_processed)
    print("All detected seizures: ")
    print(seizure_intervals_all_processed)

    print("All detected nonseizures: ")
    print(nonseizure_intervals_all_processed)

    return seizure_intervals_all_processed.copy(), nonseizure_intervals_all_processed.copy()