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

import numpy as np
import pandas as pd

def states_from_starts_ends(starts, ends, length, start_idx=0):
    """Return binary mask given starts and ends of intervals which are considered 1, 0 elsewhere

    Parameters
    ----------
    starts : np.array
        Array of start indices
    ends : np.array
        Array of end indices
    length : int
        Total duration of time series in which starts and ends are

    Returns
    -------
    mask : np.array
        Binary mask representation of intervals provided
    """
    mask = np.zeros(length, dtype=bool)
    for start, end in zip(starts-start_idx, ends-start_idx):
        mask[int(start):int(end)+1] = True
    return mask

# convert states of 1s and 0s to start and end times
def starts_ends_from_states(states, offset):
    """Detect starts and ends of states of True in a binary array

    Parameters
    ----------
    states : np.array
        Array of binary states
    offset : int
        How much to offset identified starts and ends (useful if array is from rolling window calculation)

    Returns
    -------
    starts : np.array
        Array of start indices
    ends : np.array
        Array of end indices
    """

    # set boundaries to 0 to make sure intervals end at end of data, and mark their initial state
    #states[0] = 0
    #states[-1] = 0

    # changes from 1 to 0 and 0 to 1
    transitions = np.diff(states.astype(int))
    # get period starts and ends
    starts = np.where(transitions == 1)[0] + 1 + offset # where it transitions to 1 (the first 1 idx after 0s)
    ends = np.where(transitions == -1)[0] + offset # where it is last 1 (the last 1 idx after 1s)
                                            # this was -1 initially - why???
    # Handle edge cases. np.where did not detect these as there was no transition at the edge.
    if states[0] == 1:  # if start of interval at start of states
        starts = np.insert(starts, 0, offset)
    if states[-1] == 1: # if end of interval at end of states
        ends = np.append(ends, len(states)-1+offset)

    return starts, ends

def plot_intervals(ax, starts, ends, sample_rate, marker, y_val, alpha=0.5, linewidth=1):
    """Plot intervals of series as horizontal lines with markers at end and line in between

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to plot intervals
    starts : np.array
        Array of start indices
    ends : np.array
        Array of end indices
    sample_rate : int
        Sample rate
    marker : str
        Marker (end shape and colour)
    y_val : float
        y offset of horizontal line
    alpha : float
        Line opacity
    linewidth : int
        Line width
    """
    xx = np.vstack((starts, ends)) / sample_rate
    yy = np.ones(np.shape(xx)) * y_val
    ax.plot(xx, yy, marker, alpha=alpha, linewidth=linewidth)

def good_periods_mask(bad_periods, length, return_starts_ends=False):
    """Given bad periods dataframe, get good periods binary mask

    Parameters
    ----------
    bad_periods : pd.dataframe
        DataFrame with bad period onset and offset, in columns "onset" and "offset". If None, handled as when empty
    length : int
        Total duration of time series in which bad periods are

    Returns
    -------
    good_mask
        Binary mask of good periods
    """

    if bad_periods is not None:
        if not bad_periods.empty:
            bad_periods2 = bad_periods.to_numpy(copy=True)       # copy is used to make sure we don't modify original
            # get bad period binary mask
            bad_starts = bad_periods2[:, 0]
            bad_ends = bad_periods2[:, 1]
            bad_mask = states_from_starts_ends(bad_starts, bad_ends, length)

            # exclude bad periods
            good_mask = np.logical_not(bad_mask)
        else:
            good_mask = np.ones(length)
            bad_starts = np.array([])
            bad_ends = np.array([])
    else:
        good_mask = np.ones(length)
        bad_starts = np.array([])
        bad_ends = np.array([])

    if return_starts_ends:
        return good_mask, bad_starts, bad_ends
    else:
        return good_mask

def filter_short_epochs(df, min_duration):
    """
    Filter out epochs with duration shorter than min_duration.

    Function made by paulrignanese, but moved here so repo could be standalone

    Parameters:
        df (DataFrame): DataFrame with "onset" and "offset" columns.
        min_duration (int): Minimum duration threshold for an epoch to be kept.

    Returns:
        filtered_df (DataFrame): DataFrame with short epochs removed.
    """
    filtered_df = df[df['offset'] - df['onset'] >= min_duration]
    return filtered_df

def filter_short_starts_ends(starts, ends, min_duration):
    """Filter short intervals, given start and end indices

    Parameters
    ----------
    starts : np.array
        Array of start indices
    ends : np.array
        Array of end indices
    min_duration : int
        Minimum duration of short interval (in samples)

    Returns
    -------
    starts : np.array
        Array of start indices of timings with short intervals removed
    ends : np.array
        Array of end indices of timings with short intervals removed
    """
    # filter short ratios
    diff = ends - starts
    long_idx = np.where(diff > min_duration)
    starts = starts[long_idx]
    ends = ends[long_idx]
    return starts, ends

def merge_close_starts_ends(starts, ends, max_gap):
    """Merge close intervals given starts and ends

    Parameters
    ----------
    starts : np.array
        Array of start indices
    ends : np.array
        Array of end indices
    max_gap : int
        Maximum allowed gap between end of previous interval and start of next interval

    Returns
    -------
    starts : np.array
        Array of start indices of timings with close intervals merged
    ends : np.array
        Array of end indices of timings with close intervals merged
    """
    gaps = starts[1:]- ends[:-1]    # find diffs
    short_diff_idx = np.where(gaps < max_gap)[0]   # where diffs below threshold
    starts = np.delete(starts, short_diff_idx+1)  # merge periods
    ends = np.delete(ends, short_diff_idx)
    return starts, ends

def asserts_starts_ends(starts, ends, end_idx, start_idx=0):
    assert len(starts) == len(ends)
    if len(starts) > 0 and len(ends) > 0:
        # verify end is bigger than start
        for idx0 in range(len(starts)):
            assert ends[idx0] >= starts[idx0]
        # verify events are ordered correctly and no overlapping
        for idx0 in range(len(starts) - 1):
            assert ends[idx0] <= starts[idx0 + 1]
        # verify boundaries
        assert starts[0] >= start_idx
        assert starts[-1] < end_idx
        assert ends[-1] < end_idx

def merge_close_epochs(df, max_gap):
    """
    Merge epochs that are close enough, where the gap between consecutive epochs
    is less than or equal to max_gap.

    Function made by paulrignanese, but moved here so repo could be standalone

    Parameters:
        df (DataFrame): DataFrame with "onset" and "offset" columns.
        max_gap (int): Maximum gap allowed between epochs for merging.

    Returns:
        merged_df (DataFrame): DataFrame with close epochs merged.
    """
    # Sort the DataFrame by onset time
    if not df.empty:
        df = df.sort_values(by='onset')

        merged_epochs = []  # List to store merged epochs
        current_epoch = df.iloc[0]  # Initialize with the first epoch

        for _, row in df.iterrows():
            if row['onset'] - current_epoch['offset'] <= max_gap:
                current_epoch['offset'] = max(current_epoch['offset'], row['offset'])
            else:
                merged_epochs.append(current_epoch)
                current_epoch = row

        merged_epochs.append(current_epoch) #append last epoch

        merged_df = pd.DataFrame(merged_epochs)
    else:
        merged_df = df
    return merged_df