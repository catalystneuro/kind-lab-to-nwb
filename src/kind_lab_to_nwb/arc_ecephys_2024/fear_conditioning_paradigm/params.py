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
from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd

from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.params_utils import generate_time_frequency_analysis_params, generate_epoch_frequency_analysis_params_cond, \
    generate_epoch_frequency_analysis_params_recall


@dataclass
class AnalysisParams:
    # Preprocessing parameters
    seizures_detection_mode: str = 'dual_band_peaks'  # 'dual_band_peaks' or 'cepstral'
    sleep_detection_mode: str = "theta_delta_ratio"
    data_type: str = 'openephys'  # 'openephys', 'taini', 'neurologger'
    sessions_folders: List[str] = field(
        default_factory=lambda: ['Baseline_tone_flash_hab', 'Hab_1', 'Hab_2', 'Cond', 'Recall',
                                 'Seizure_screening'])  # 'Hab_2', 'Recall','Seizure_screening''Hab_1', 'Hab_2'

    skip_excel_loading: bool = False  # ← this is to be changed for testing only

    base_dir: str = '/mnt/308A3DD28A3D9576/SYNGAP_ephys/'
    path_info_electrodes: str = field(init=False)
    sample_rate: float = 2000.0
    motion_processing_cutoff_freq: float = 2.0
    animal_types: List[str] = field(default_factory=lambda: ['wt', 'het'])
    threshold_motion_detection: float = 0.001
    redo_bad_epochs: bool = False
    redo_seizures: bool = False
    seizure_detection_channel_helper: bool = False

    redo_sleep: bool = False
    preproc_periods_len_minutes: float = 20
    notch_frequency: float = 50.0
    quality_factor: float = 7.0
    redo_raw_data_extraction: bool = False

    # whole session spectrograms params (used for bad epochs detection)
    redo_whole_session_spects: bool = False
    zscore_whole_session_spects: bool = True
    whole_session_spect_min_freq: int = 1
    whole_session_spect_max_freq: int = 85
    whole_session_spect_n_cycles: int = 2
    whole_session_spect_time_bandwidth: float = 2.0
    whole_session_spect_freq_resolution: float = 0.25
    whole_session_spect_decimation: int = 200
    whole_session_spect_n_jobs: int = 1

    preprocessed_data_types: List[str] = field(
        default_factory=lambda: ['lfp', 'motion', 'ttl_events', 'session_spectrograms'])
    preprocessed_timings: List[str] = field(
        default_factory=lambda: ['nonseizure', 'bad_epochs', 'cs', 'freezing', 'seizure', 'sleep'])

    epoch_frequency_analysis_types: List[str] = field(
        default_factory=lambda: ['psd', 'coherence', 'phase_difference', 'cross_corelation'])
    time_frequency_analysis_types: List[str] = field(
        default_factory=lambda: ['spectrogram', 'coherogram'])

    redo_pwr: bool = False
    redo_cohe: bool = False
    redo_spects: bool = False
    redo_coheros: bool = False
    extract_video_animation_recall: bool = False

    epochs_length: int = 5  # length in seconds for epoching prior to extract power and coherence

    # psds parameters
    psds_min_freq: int = 1
    psds_max_freq: int = 85
    psds_freq_resolution: int = 0.25
    psds_n_cycles: int = 2
    psds_time_bandwidth: float = 3.0

    # coherence parameters
    cohe_min_freq: int = 1
    cohe_max_freq: int = 85
    cohe_n_tapers: int = 5

    # phase difference angle parameters
    phase_diff_min_freq: int = 3
    phase_diff_max_freq: int = 6

    # spectrograms params
    spect_min_freq: int = 1
    spect_max_freq: int = 85
    spect_n_tapers: int = 5
    spect_n_cycles: int = 2
    spect_freq_resolution: float = 0.25
    spects_decimation: int = 10
    spects_time_bandwidth: float = 3.0

    # coherograms params
    coheros_min_freq: int = 1
    coheros_max_freq: int = 85
    coheros_n_tapers: int = 5
    coheros_n_cycles: int = 2
    coheros_freq_resolution: float = 0.25
    coheros_decimation: int = 10

    # Conditioning timings
    cond_cs_n: int = 6
    cond_cs_lengh: int = 9
    cond_shock_length: int = 1
    cond_pre_cs_length: int = 180
    #recall timings
    recall_pre_cs_length: int = 180
    recall_cs_length: int = 30
    recall_cs_n: int = 10


    # Power and connectivity: all sessions
    pan_sessions_epochs_frequency_analysis_timings: List[str] = field(
        default_factory=lambda: ['good', 'freezing', 'nonfreezing']
    )
    pan_sessions_time_frequency_analysis_timings_labels: List[str] = field(default_factory=lambda: ['freezing'])
    pan_sessions_time_frequency_analysis_timings_window: Tuple[int, int] = (20000, 20000)

    # Baseline session
    baseline_epochs_frequency_analysis_timings: List[str] = field(default_factory=lambda: ['stim', 'nonstim'])

    # Recall session
    recall_basic_timings: List[str] = field(default_factory=lambda: [
        "pre_cs", "cs", "noncs"])
    recall_within_timings: List[str] = field(default_factory=lambda: ["freezing_within", "nonfreezing_within"])
    recall_single_noncs: List[str] = field(default_factory=lambda: [f"noncs_{i}" for i in range(1, 11)])
    recall_single_cs: List[str] = field(default_factory=lambda: [f"cs_{i}" for i in range(1, 11)])

    recall_raw_cs_timings: List[str] = field(default_factory=lambda: [f"cs_raw_{i}" for i in range(1, 11)])
    recall_freezing_time_window: Tuple[int, int] = (20000, 20000)
    recall_cs_time_window: Tuple[int, int] = (40000, 40000)

    # Conditioning session
    cond_basic_timings: List[str] = field(default_factory=lambda: ["cs", "noncs"])
    cond_within_timings: List[str] = field(default_factory=lambda: ["freezing_within", "nonfreezing_within"])
    cond_single_noncs: List[str] = field(default_factory=lambda: [f"noncs_{i}" for i in range(1, 8)])
    cond_single_cs: List[str] = field(default_factory=lambda: [f"cs_{i}" for i in range(1, 7)])
    cond_noncs_time_window: Tuple[int, int] = (0, 40000)
    cond_cs_time_window: Tuple[int, int] = (40000, 18000)

    def __post_init__(self):
        self.path_info_electrodes = os.path.join(self.base_dir, 'channels_details_v2.xlsx')
        if not self.skip_excel_loading:
            self.all_animals_info = pd.read_excel(
                self.path_info_electrodes)  # Load animals paths and channels information (areas, bad channels...)

            self.validate_all_animals_info()

        self.pan_sessions_time_frequency_analysis_timings = generate_time_frequency_analysis_params(
            [(self.pan_sessions_time_frequency_analysis_timings_labels,
              self.pan_sessions_time_frequency_analysis_timings_window)])

        self.cond_epochs_frequency_analysis_timings = generate_epoch_frequency_analysis_params_cond(
            self.cond_basic_timings,
            self.cond_within_timings,
            self.cond_single_noncs,
            self.cond_single_cs)

        self.cond_time_frequency_analysis_timings = generate_time_frequency_analysis_params([
            (self.cond_single_noncs, self.cond_noncs_time_window),
            (self.cond_single_cs, self.cond_cs_time_window),
        ])

        self.recall_epochs_frequency_analysis_timings = generate_epoch_frequency_analysis_params_recall(
            self.recall_basic_timings,
            self.recall_within_timings,
            self.recall_single_noncs,
            self.recall_single_cs)

        self.recall_time_frequency_analysis_timings = generate_time_frequency_analysis_params([
            (self.recall_raw_cs_timings, self.recall_cs_time_window)
        ])

        self.filenames = {  # init filenames to fill it later
            'epoch_analysis': {},
            'time_frequency_analysis': {}
        }

        # Initialize filenames as a field
        self.filenames = self.generate_filenames()

    def generate_filenames(self):
        # Initialize master dictionary
        filenames = {}

        # Iterate through session types
        for session in self.sessions_folders:
            filenames[session] = {
                'epoch_analysis': {},
                'time_frequency_analysis': {},
                'preprocessed_data': {},
                'timings': {}
            }

            # Pan session analysis (included in all sessions)
            for analysis_type in self.epoch_frequency_analysis_types:
                for timing_label in self.pan_sessions_epochs_frequency_analysis_timings:
                    if isinstance(timing_label, tuple):
                        formatted_timing_label = "_".join(map(str, timing_label))
                    else:
                        formatted_timing_label = timing_label
                        filename = f"{analysis_type}_{formatted_timing_label}.h5"
                    if analysis_type not in filenames[session]['epoch_analysis']:
                        filenames[session]['epoch_analysis'][analysis_type] = {}
                    filenames[session]['epoch_analysis'][analysis_type][formatted_timing_label] = filename

            for analysis_type in self.time_frequency_analysis_types:
                for timing_label in self.pan_sessions_time_frequency_analysis_timings:

                    if isinstance(timing_label, tuple):
                        formatted_timing_label = "_".join(map(str, timing_label))
                    else:
                        formatted_timing_label = timing_label
                    filename = f"{analysis_type}_{formatted_timing_label}.h5"

                    if analysis_type not in filenames[session]['time_frequency_analysis']:
                        filenames[session]['time_frequency_analysis'][analysis_type] = {}
                    filenames[session]['time_frequency_analysis'][analysis_type][formatted_timing_label] = filename

            # Specific to 'Cond' session
            if session == 'Cond':
                for analysis_type in self.epoch_frequency_analysis_types:
                    for timing_label in self.cond_epochs_frequency_analysis_timings:
                        if isinstance(timing_label, tuple):
                            formatted_timing_label = "_".join(map(str, timing_label))
                        else:
                            formatted_timing_label = timing_label

                        filename = f"{analysis_type}_{formatted_timing_label}.h5"
                        filenames[session]['epoch_analysis'][analysis_type][formatted_timing_label] = filename

                for analysis_type in self.time_frequency_analysis_types:
                    for timing_label in self.cond_time_frequency_analysis_timings:
                        if isinstance(timing_label, tuple):
                            formatted_timing_label = "_".join(map(str, timing_label))
                        else:
                            formatted_timing_label = timing_label

                        filename = f"{analysis_type}_{formatted_timing_label}.h5"
                        filenames[session]['time_frequency_analysis'][analysis_type][formatted_timing_label] = filename

            # Specific to 'Recall' session
            if session == 'Recall':
                for analysis_type in self.epoch_frequency_analysis_types:
                    for timing_label in self.recall_epochs_frequency_analysis_timings:

                        if isinstance(timing_label, tuple):
                            formatted_timing_label = "_".join(map(str, timing_label))
                        else:
                            formatted_timing_label = timing_label
                        filename = f"{analysis_type}_{formatted_timing_label}.h5"
                        filenames[session]['epoch_analysis'][analysis_type][formatted_timing_label] = filename

                for analysis_type in self.time_frequency_analysis_types:
                    for timing_label in self.recall_time_frequency_analysis_timings:

                        if isinstance(timing_label, tuple):
                            formatted_timing_label = "_".join(map(str, timing_label))
                        else:
                            formatted_timing_label = timing_label
                        filename = f"{analysis_type}_{formatted_timing_label}.h5"
                        filenames[session]['time_frequency_analysis'][analysis_type][formatted_timing_label] = filename

            # Preprocessed data
            for data_type in self.preprocessed_data_types:
                filename = f"preprocessed_data_{data_type}"
                filenames[session]['preprocessed_data'][data_type] = filename

            # Timings
            for data_type in self.preprocessed_timings:
                filename = f"timings_{data_type}.h5"
                filenames[session]['timings'][data_type] = filename

        return filenames

    def validate_all_animals_info(self):
        df = self.all_animals_info

        # Validate first 4 columns are exactly as required
        expected_prefix_cols = ['ID', 'Genotype', 'Folder', 'source_number']
        actual_prefix_cols = list(df.columns[:4])
        if actual_prefix_cols != expected_prefix_cols:
            raise ValueError(f"First four columns must be {expected_prefix_cols}, got: {actual_prefix_cols}")

        # Validate remaining columns are ints from 0 to 15 if data_type is 'openephys'
        if self.data_type == 'openephys':
            expected_channel_cols = list(range(16))
            actual_channel_cols = list(df.columns[4:])
            if sorted(actual_channel_cols) != expected_channel_cols:
                raise ValueError(
                    f"For openephys, expected channel columns from 0 to 15 as integers. "
                    f"Got: {actual_channel_cols}"
                )


    def validate_all_other_params(self):

        def check_freq_bounds_with_decimation(min_f, max_f, decim, label):
            effective_sr = self.sample_rate if decim == 0 else self.sample_rate / decim
            nyquist = effective_sr / 2
            if max_f >= nyquist:
                raise ValueError(
                    f"{label}: max_freq {max_f} Hz exceeds Nyquist ({nyquist:.2f} Hz) after decimation={decim}")

        if self.seizures_detection_mode not in ['dual_band_peaks', 'cepstral']:
            raise ValueError(f"Invalid seizures_detection_mode: {self.seizures_detection_mode}")

        if self.data_type not in ['openephys', 'taini', 'neurologger']:
            raise ValueError(f"Invalid data_type: {self.data_type}")

        allowed_sessions = ['Seizure_screening', 'Baseline_tone_flash_hab', 'Hab_1', 'Hab_2', 'Recall', 'Cond']
        for session in self.sessions_folders:
            if session not in allowed_sessions:
                raise ValueError(f"Invalid session folder name: {session}")

        if not os.path.isdir(self.base_dir):
            raise FileNotFoundError(f"base_dir does not exist or is not a directory: {self.base_dir}")

        if not os.path.isfile(self.path_info_electrodes):
            raise FileNotFoundError(f"Electrodes info file not found: {self.path_info_electrodes}")

        if not (isinstance(self.sample_rate, float) and self.sample_rate > 0):
            raise ValueError(f"sample_rate must be a positive integer, got: {self.sample_rate}")

        if not (isinstance(self.motion_processing_cutoff_freq,
                           (int, float)) and self.motion_processing_cutoff_freq > 0):
            raise ValueError(f"motion_processing_cutoff_freq must be > 0, got: {self.motion_processing_cutoff_freq}")

        if not (isinstance(self.threshold_motion_detection, (int, float)) and self.threshold_motion_detection > 0):
            raise ValueError(f"threshold_motion_detection must be > 0, got: {self.threshold_motion_detection}")

        bool_params = ['redo_whole_session_spects', 'zscore_whole_session_spects', 'redo_seizures', 'redo_sleep',
                       'redo_pwr', 'redo_cohe', 'redo_spects', 'redo_coheros', 'redo_raw_data_extraction',
                       'redo_whole_session_spects', 'redo_bad_epochs']
        for bp in bool_params:
            val = getattr(self, bp)
            if not isinstance(val, bool):
                raise ValueError(f"{bp} must be bool, got: {val}")

        if not (isinstance(self.preproc_periods_len_minutes, (int, float)) and self.preproc_periods_len_minutes > 0):
            raise ValueError(f"preproc_periods_len_minutes must be > 0, got: {self.preproc_periods_len_minutes}")

        if not (isinstance(self.notch_frequency, (int, float)) and self.notch_frequency > 0):
            raise ValueError(f"notch_frequency must be > 0, got: {self.notch_frequency}")

        if not (isinstance(self.quality_factor, (int, float)) and self.quality_factor > 0):
            raise ValueError(f"quality_factor must be > 0, got: {self.quality_factor}")
        # Epochs length
        if not isinstance(self.epochs_length, int) or self.epochs_length <= 0:
            raise ValueError(f"epochs_length must be a positive integer in seconds, got: {self.epochs_length}")

        nyquist = self.sample_rate / 2
        # whole session Spectrograms

        if not (0 < self.whole_session_spect_min_freq < self.whole_session_spect_max_freq < nyquist):
            raise ValueError(
                f"whole_session_spect_min_freq and whole_session_spect_max_freq must be >0 and < Nyquist ({nyquist})"
            )

        if self.whole_session_spect_n_cycles <= 0:
            raise ValueError("whole_session_spect_n_cycles must be > 0")

        if self.whole_session_spect_time_bandwidth <= 0:
            raise ValueError("whole_session_spect_time_bandwidth must be > 0")

        if self.whole_session_spect_freq_resolution <= 0:
            raise ValueError("whole_session_spect_freq_resolution must be > 0")

        if self.whole_session_spect_decimation < 0:
            raise ValueError("whole_session_spect_decimation must be >= 0")

        if self.whole_session_spect_n_jobs < 1:
            raise ValueError("whole_session_spect_n_jobs must be >= 1")

        # PSDs validation
        if not (0 < self.psds_min_freq < self.psds_max_freq < nyquist):
            raise ValueError(
                f"psds_min_freq and psds_max_freq must be >0 and < Nyquist ({nyquist}), got: {self.psds_min_freq}, {self.psds_max_freq}")
        if self.psds_freq_resolution <= 0:
            raise ValueError("psds_freq_resolution must be > 0")
        if self.psds_n_cycles <= 0:
            raise ValueError("psds_n_cycles must be > 0")
        if self.psds_time_bandwidth <= 0:
            raise ValueError("psds_time_bandwidth must be > 0")

        # Coherence validation
        if not (0 < self.cohe_min_freq < self.cohe_max_freq < nyquist):
            raise ValueError("cohe_min_freq and cohe_max_freq must be >0 and < Nyquist")
        if self.cohe_n_tapers < 1:
            raise ValueError("cohe_n_tapers must be >= 1")

        # Spectrograms
        if not (0 < self.spect_min_freq < self.spect_max_freq < nyquist):
            raise ValueError("spect_min_freq and spect_max_freq must be >0 and < Nyquist")
        if self.spect_n_tapers < 1:
            raise ValueError("spect_n_tapers must be >= 1")
        if self.spect_n_cycles <= 0:
            raise ValueError("spect_n_cycles must be > 0")
        if self.spect_freq_resolution <= 0:
            raise ValueError("spect_freq_resolution must be > 0")
        if self.spects_decimation < 0:
            raise ValueError("spects_decimation must be >= 0")

        # Coherograms
        if not (0 < self.coheros_min_freq < self.coheros_max_freq < nyquist):
            raise ValueError("coheros_min_freq and coheros_max_freq must be >0 and < Nyquist")
        if self.coheros_n_tapers < 1:
            raise ValueError("coheros_n_tapers must be >= 1")
        if self.coheros_n_cycles <= 0:
            raise ValueError("coheros_n_cycles must be > 0")
        if self.coheros_freq_resolution <= 0:
            raise ValueError("coheros_freq_resolution must be > 0")
        if self.coheros_decimation < 0:
            raise ValueError("coheros_decimation must be >= 0")

        # Phase difference frequency bounds
        if not (0 < self.phase_diff_min_freq < self.phase_diff_max_freq < nyquist):
            raise ValueError(
                f"phase_diff_min_freq and phase_diff_max_freq must be >0 and < Nyquist ({nyquist}), got: "
                f"{self.phase_diff_min_freq}, {self.phase_diff_max_freq}"
            )
        # Spectrogram validation
        check_freq_bounds_with_decimation(self.spect_min_freq, self.spect_max_freq, self.spects_decimation,
                                          "spectrogram")

        # Coherogram validation
        check_freq_bounds_with_decimation(self.coheros_min_freq, self.coheros_max_freq, self.coheros_decimation,
                                          "coherogram")

        if not (isinstance(self.cond_cs_lengh, (int, float)) and self.cond_cs_lengh > 0):
            raise ValueError(f"cond_cs_lengh must be > 0, got {self.cond_cs_lengh}")

        if not (isinstance(self.cond_cs_n, (int, float)) and self.cond_cs_n > 0):
            raise ValueError(f"cond_cs_n must be > 0, got {self.cond_cs_n}")

        if not (isinstance(self.cond_pre_cs_length, (int, float)) and self.cond_pre_cs_length > 0):
            raise ValueError(f"cond_pre_cs_length must be > 0, got {self.cond_pre_cs_length}")

        if not (isinstance(self.cond_shock_length, (int, float)) and self.cond_shock_length > 0):
            raise ValueError(f"cond_shock_length must be > 0, got {self.cond_shock_length}")

        if not (isinstance(self.recall_cs_length, (int, float)) and self.recall_cs_length > 0):
            raise ValueError(f"recall_cs_length must be > 0, got {self.recall_cs_length}")

        if not (isinstance(self.recall_cs_n, (int, float)) and self.recall_cs_n > 0):
            raise ValueError(f"recall_cs_n must be > 0, got {self.recall_cs_n}")

        if not (isinstance(self.recall_pre_cs_length, (int, float)) and self.recall_pre_cs_length > 0):
            raise ValueError(f"recall_pre_cs_length must be > 0, got {self.recall_pre_cs_length}")

        print("All parameters validated successfully.")

############################################ SLEEP DETECTION ###########################################################

# sleep detection
theta_low = 5.0
theta_high = 11.0
delta_low = 1.0
delta_high = 4.0

periods_len_minutes_sleep = 20 # minutes

###################################### DUAL BAND PEAKS SEIZURE DETECTION ###############################################

# dual_band_peaks interface
seizure_channel_helper = False  # show segment of all channels to help decide channel to use
ask_to_skip_frames = False       # ask to skip N frames, useful for testing