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

import itertools
from typing import List, Tuple


def generate_time_frequency_analysis_params(
        categories: List[Tuple[List[str], Tuple[int, int]]],
        boundaries: List[str] = ("onset", "offset")
) -> List[Tuple[str, str, int, int]]:
    """
    Generate a list of timing configurations for time-frequency analysis.

    Each configuration is a tuple:
    (timing_label, boundary_type, pre_window_samples, post_window_samples)

    Parameters
    ----------
    categories : list of tuples
        Each tuple should be of the form (timing_list, time_window), where:
        - timing_list: list of str
            A list of string labels describing timing events.
        - time_window: tuple of two ints
            A pair (pre_window, post_window) in samples to define the analysis window.

    boundaries : list of str, optional
        The boundary types to use for each timing (e.g., "onset", "offset").

    Returns
    -------
    list of tuples
        Each tuple corresponds to a unique (timing_label, boundary, pre_window, post_window) combination.
    """
    output = []
    for timing_list, time_window in categories:
        for timing in timing_list:
            for boundary in boundaries:
                output.append((timing, boundary, *time_window))
    return output


def generate_epoch_frequency_analysis_params_cond(cond_basic_timings, cond_within_timings, cond_single_noncs,
                                                  cond_single_cs):
    cond_within_noncs_timings = [f"{prefix}_{part}" for prefix, part in
                                 itertools.product(cond_within_timings, cond_single_noncs)]
    cond_within_single_cs_timings = cond_single_cs + [f"{prefix}_{part}" for prefix, part in
                                                      itertools.product(cond_within_timings, cond_single_cs)]

    cond_epochs_frequency_analysis_timings = (
            cond_basic_timings + cond_within_single_cs_timings + cond_single_noncs + cond_within_noncs_timings + cond_single_cs)
    return cond_epochs_frequency_analysis_timings


def generate_epoch_frequency_analysis_params_recall(recall_basic_timings, recall_within_timings,
                                                    recall_single_noncs, recall_single_cs):
    recall_within_basic_timings = [f"{prefix}_{subset}" for prefix, subset in
                                   itertools.product(recall_within_timings, recall_basic_timings)]

    recall_within_noncs_timings = [f"{prefix}_{part}" for prefix, part in
                                   itertools.product(recall_within_timings, recall_single_noncs)]

    recall_cs_timings = recall_single_cs + [f"{prefix}_{part}" for prefix, part in
                                            itertools.product(recall_within_timings, recall_single_cs)]

    recall_epochs_frequency_analysis_timings = (
            recall_basic_timings + recall_within_basic_timings + recall_single_noncs + recall_within_noncs_timings + recall_cs_timings)

    return recall_epochs_frequency_analysis_timings
