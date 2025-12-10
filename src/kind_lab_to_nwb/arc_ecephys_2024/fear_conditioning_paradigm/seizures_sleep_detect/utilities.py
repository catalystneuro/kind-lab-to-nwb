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

import os
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider
from matplotlib import pyplot as plt
import sys

from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.seizure_timings_processing import asserts_starts_ends, starts_ends_from_states, states_from_starts_ends
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.seizures_param import (window_size_default, threshold_band_default, threshold_hf_default,
                                                  spread_default, quantile_band_default, quantile_hf_default,
                                                  constant_saving_mode)

def split_signal_into_frames(start_idx_of_screening, end_idx_of_screening, length, sample_rate, periods_len_minutes_seizure):
    """Split file into a number of frames of a certain length. Handles custom starts and ends

    Parameters
    ----------
    start_idx_of_screening: int
        Where in file to start screening
    end_idx_of_screening: int
        Where in file to end screening
    length: int
        Total file length
    sample_rate: float

    Returns
    -------
    frame_number: int
        Number of frames to detect in
    samples_per_frame: int
        Number of samples per frame
    """
    if start_idx_of_screening is not None:
        print(f"Start idx of screening set to {start_idx_of_screening}")
    if end_idx_of_screening is not None:
        print(f"End idx of screening set to {end_idx_of_screening}")

    # split into periods/windows
    if end_idx_of_screening is None and start_idx_of_screening is None:
        n_samples = length
    if start_idx_of_screening is not None and end_idx_of_screening is None:
        n_samples = length - start_idx_of_screening
    if start_idx_of_screening is None and end_idx_of_screening is not None:
        n_samples = end_idx_of_screening
    if start_idx_of_screening is not None and end_idx_of_screening is not None:
        assert end_idx_of_screening > start_idx_of_screening
        n_samples = end_idx_of_screening - start_idx_of_screening

    print(f'Session length = {n_samples / sample_rate / 60} min')
    frame_number = int(np.ceil(n_samples / sample_rate / 60 / periods_len_minutes_seizure))
    print(f'Split into {frame_number} frames of {periods_len_minutes_seizure} min long')
    samples_per_frame = periods_len_minutes_seizure * 60 * sample_rate

    return frame_number, samples_per_frame

def screening_start_end_idx_asserts(start_idx_of_screening, end_idx_of_screening, length, window_length):
    """Ensure start and end idx of screening are valid

    Parameters
    ----------
    start_idx_of_screening: int
        Where in file to start screening
    end_idx_of_screening: int
        Where in file to end screening
    length: int
        Total file length
    """
    if start_idx_of_screening is not None:
        actual_start = start_idx_of_screening
        assert start_idx_of_screening >= 0
    else:
        actual_start = 0
    if end_idx_of_screening is not None:
        actual_end = end_idx_of_screening
        assert end_idx_of_screening < length
    else:
        actual_end = length

    if start_idx_of_screening is not None and end_idx_of_screening is not None:
        assert end_idx_of_screening > start_idx_of_screening

    if actual_start > actual_end:
        raise ValueError("Start/continuation of screening is after end. This may occur if loading backup of fully screened file.")
    #if actual_end - actual_start < window_length:
    #    raise ValueError("Time between start and end must be greater than window length!")

def backup_append_start_end(backup_filename, start_idx_of_screening, end_idx_of_screening):
    """If start and end indices of screening are specified, append these to backup filename

    Parameters
    ----------
    backup_filename: str
        Path to backup file
    start_idx_of_screening: int
        Where in file to start screening
    end_idx_of_screening: int
        Where in file to end screening

    Returns
    -------
    backup_filename: str
        Path to backup file with start and end indicies, if applicable
    """
    if start_idx_of_screening is not None:
        if start_idx_of_screening != 0:     # if loaded from backup, it will read 0 as start, and we don't want that written
            backup_filename += f"_start_{start_idx_of_screening}"
    if end_idx_of_screening is not None:
        backup_filename += f"_end_{end_idx_of_screening}"
    return backup_filename


def starts_ends_invert_and_to_pandas(starts, ends, start_idx_of_screening, end_idx_of_screening):
    """Given interval starts and ends, convert the original and inverse into pandas

    Parameters
    ----------
    starts: np.array
        Starts of intervals
    ends: np.array
        Ends of intervals
    length: int
        Length of array

    Returns
    -------
    intervals_all
        pandas dataframe of all intervals as original specified
    inverted_intervals_all
        pandas dataframe of all inverted intervals
    """
    intervals_all = pd.DataFrame({'onset': starts,'offset': ends})

    # invert and save
    mask = states_from_starts_ends(starts, ends, end_idx_of_screening-start_idx_of_screening, start_idx=start_idx_of_screening)
    inverted_starts, inverted_ends = starts_ends_from_states(np.logical_not(mask), start_idx_of_screening)

    asserts_starts_ends(inverted_starts, inverted_ends, end_idx_of_screening, start_idx=start_idx_of_screening)

    inverted_intervals_all = pd.DataFrame({'onset': inverted_starts,
                                            'offset': inverted_ends})

    return intervals_all, inverted_intervals_all

def write_seizures_nonseizures_to_file(excel_file, label, seizures, nonseizures):
    """Write seizures and nonseizures into an excel sheet with two tables

    Parameters
    ----------
    excel_file: str
        File location to save at
    label: str
        File label
    seizures: pd.DataFrame
        Seizures onsets and offsets
    nonseizures: pd.DataFrame
        Nonseizures onsets and offsets
    """
    file = f"{excel_file}detected_seizures_{label}.xlsx"
    with pd.ExcelWriter(file) as writer:
        seizures.to_excel(writer, sheet_name='seizures')
        nonseizures.to_excel(writer, sheet_name='nonseizures')


def load_backup(backup_file_original_name, start_idx_of_screening, end_idx_of_screening, all_seizure_starts, all_seizure_ends,
                length, window_length):
    """If backup file exists, prompt user to load and continue from backup, adding previous seizures and moving start

    Parameters
    ----------
    backup_file: str
        Path to backup file
    start_idx_of_screening: int
        Where in file to start screening
    end_idx_of_screening: int
        Where in file to end screening
    all_seizure_starts: np.ndarray
        Seizure starts array
    all_seizure_ends: np.ndarray
        Seizure ends array
    length: int
        Total file length

    Returns
    -------
    backup_file: str
        Path to backup file, may be modified to include screening start and end idx
    start_idx_of_screening: int
        Start idx of screening start, may be modified if continuing from backup file
    all_seizure_starts: np.ndarray
        Seizure starts array, may be modified to include seizures loaded from backup
    all_seizure_ends: np.ndarray
        Seizure ends array, may be modified to include seizures loaded from backup
    """


    backup_start_of_screening = None    # if backup loaded, stores where the screening originally started in the backup

    if backup_file_original_name is not None:
        # Try to load from backup
        # append start and end of screening to base filename
        # NOTE: if you backed up while detecting a different interval than the one specified, it will not load!
        screening_start_end_idx_asserts(start_idx_of_screening, end_idx_of_screening, length, window_length)
        backup_file = backup_append_start_end(backup_file_original_name, start_idx_of_screening, end_idx_of_screening)
        # Add extension if missing
        if not backup_file.endswith(".npz"):
            backup_file += ".npz"
        print(f"Backup specified at {backup_file}")
        if os.path.exists(backup_file):
            print("Backup file found - do you wish to load detected seizures and continue from there?")
            load = input("Enter Y to load, N to ignore: ")
            # if loading from backup
            if load.lower() == 'y':
                # if start idx specified, acknowledge backup's end will be used instead
                if start_idx_of_screening is not None:
                    print("Start index specified, but loading from backup file, so start is after last screened frame from backup. Ending from backup ignored and using ending specified here.")
                    input("Overwriting start index with end index of backup. Press enter to acknowledge")
                backup_data = np.load(backup_file)
                all_seizure_starts = backup_data["starts"]
                all_seizure_ends = backup_data["ends"]
                end_conditions = backup_data["position"]

                # extract conditions at point of backup
                backup_start_of_screening = int(end_conditions[0]) # where in backup file did the screenings start
                if start_idx_of_screening is not None:
                    assert backup_start_of_screening == start_idx_of_screening # start in backup equal to current start to prevent unintuitive behaviour
                start_idx_of_screening = int(end_conditions[1])  # where in backup file did screenings end is new start

                # verify data
                screening_start_end_idx_asserts(start_idx_of_screening, end_idx_of_screening, length, window_length)
                # all loaded seizures should be between start of screening in backup and new start (end of backup)
                asserts_starts_ends(all_seizure_starts, all_seizure_ends, start_idx_of_screening, start_idx=backup_start_of_screening)

                print(f"Sucessfully loaded backup file {backup_file}")
                print(f"Continuing screening at idx {start_idx_of_screening}, originally started at idx {backup_start_of_screening}")
                print(f"Total number of loaded seizures = {len(all_seizure_starts)}")

                print(all_seizure_starts)
                print(all_seizure_ends)
            elif load.lower() == 'n':
                print("Ignoring backup file, it will be overwritten if seizure detection performed")
                input("Press enter to acknowledge")
            else:
                raise ValueError("Invalid response")
        else:
            print("Backup file not found")
    else:
        print("No backup file specified.")

    # pass possibly modified backup file
    backup_file = backup_file_original_name

    return backup_file, start_idx_of_screening, backup_start_of_screening, all_seizure_starts, all_seizure_ends

def save_backup(backup_file, all_seizure_starts, all_seizure_ends, backup_start_of_screening, start_idx_of_screening, end_idx):
    # if continuing from backup, save backup start as overall start
    if backup_start_of_screening is not None:
        np.savez(backup_file, starts=all_seizure_starts, ends=all_seizure_ends, position=np.array([backup_start_of_screening, end_idx]))
    # if not continuing from backup, save this screening start as overall start
    else:
        np.savez(backup_file, starts=all_seizure_starts, ends=all_seizure_ends, position=np.array([start_idx_of_screening, end_idx]))

class ConstantChangerWindow:
    """Window for changing constants"""
    def __init__(self, axs):
        """Initialisation

        Parameters
        ----------
        axs: matplotlib.axes.Axes
        """
        self.changes = False
        for ax in axs:
            ax.figure.canvas.mpl_connect('key_press_event', self.on_press)

    def on_press(self, event):
        """Handle keystrokes

        Parameters
        ----------
        event : matplotlib.event.Event
        """
        sys.stdout.flush()
        # Reject constant changes
        if event.key == 'x':
            print("Rejecting constant changes")
            plt.close()
        # Accept constant changes
        if event.key == 'y':
            print("Accepting constant changes")
            self.changes = True
            plt.close()

class Constants:
    def __init__(self, base_dir, constant_saving_mode, animal_folder=None, session_folder=None):
        self.base_dir = base_dir
        self.animal_folder = animal_folder
        self.session_folder = session_folder
        self.constant_saving_mode = constant_saving_mode
        self.load()

    def _set_defaults(self):
        """Initalise default constants"""           # TODO use dictionary
        self.quantile_band = quantile_band_default  # quantile at which the minimum peak height is set
        self.quantile_hf = quantile_hf_default
        self.threshold_band = threshold_band_default  # variance
        self.threshold_hf = threshold_hf_default
        self.window_size = window_size_default  # rolling window size, even number
        self.spread = spread_default  # how much binary dilation happens around the place where rolling variance dips below the threshold, seconds
        self.edge = int(self.window_size / 2)

    def _load_constants_from_file(self):
        """Extract constants from spreadsheet given location

        Returns
        -------
        quantile_band: float
        quantile_hf: float
        threshold_band: float
        threshold_hf: float
        window_size: int
        spread: float
        edge: int
        """
        constants = pd.read_excel(self.constant_location)
        for _, row in constants.iterrows():
            const = row['Constant']
            val = row['Value']
            if const == "quantile_band":
                quantile_band = float(val)
            elif const == "quantile_hf":
                quantile_hf = float(val)
            elif const == "threshold_band":
                threshold_band = float(val)
            elif const == "threshold_hf":
                threshold_hf = float(val)
            elif const == "window_size":
                window_size = int(val)
            elif const == "spread":
                spread = float(val)
        edge = int(window_size / 2)

        self.quantile_band = quantile_band
        self.quantile_hf = quantile_hf
        self.threshold_band = threshold_band
        self.threshold_hf = threshold_hf
        self.window_size = window_size
        self.spread = spread
        self.edge = edge

    def _set_constant_file(self):
        """Get a possible constant file location depending on specified constant saving mode

        Returns none if file does not exist in expected location

        Parameters
        ----------
        base_dir: str
            Path to base directory
        animal_folder: str
            Path to animal folder, if applicable
        session_folder: str
            Path to session folder, if applicable

        Returns
        -------
        constant_location
            Location of constant saving/loading location
        """

        # Get path for constant saving mode
        if constant_saving_mode == 'global':
            constant_location = self.base_dir + "seizure_global_constants.xlsx"
        elif constant_saving_mode == 'per_animal':
            assert self.animal_folder is not None
            constant_location = self.base_dir + self.animal_folder + "/seizure_animal_constants.xlsx"
        elif constant_saving_mode == 'none':
            constant_location = ''
        elif constant_saving_mode == 'per_session_per_animal':
            assert self.animal_folder is not None and self.session_folder is not None
            constant_location = self.base_dir + self.animal_folder + "/" + self.session_folder + "/seizure_session_animal_constants.xlsx"
        elif constant_saving_mode == 'per_session':
            assert self.session_folder is not None
            # since sessions are in every animal, if we want session settings in one place, new files with session names are created in base dir
            constant_location = self.base_dir + "seizure_session_" + self.session_folder + "_constants.xlsx"
        else:
            raise ValueError("Invalid constant saving mode specified!")

        # Check if file exists
        if os.path.isfile(constant_location):
            self.constant_location = constant_location
        else:
            self.constant_location = None

    def _verify_constants(self):
        quantiles = [self.quantile_band, self.quantile_hf]
        thresholds = [self.threshold_band, self.threshold_hf]
        for quantile in quantiles:
            assert quantile > 0
            assert quantile <= 1
        for threshold in thresholds:
            assert threshold > 0
        assert self.window_size > 0
        assert self.spread > 0
        assert type(self.window_size) == int
        # window size must be even
        if self.window_size % 2 != 0:
            raise ValueError("Window size must be a multiple of 2")

    def load(self):
        self._set_constant_file()
        # if there is a constant file
        print(f"Loading constants from {self.constant_location} with mode {self.constant_saving_mode}")
        if self.constant_location is not None:
            self._load_constants_from_file()
        else:
            print("! NOTICE ! Constant file for specified constant mode not found! Using defaults - press any key to acknowledge!")
            input()
            self._set_defaults()

        self._verify_constants()

    def save(self):
        """Save constants to spreadsheet given location

        Parameters
        ----------
        dir: str
            Directory to save parameters in
        quantile_band: float
        quantile_hf: float
        threshold_band: float
        threshold_hf: float
        window_size: int
        spread: float
        """

        df = pd.DataFrame([self.quantile_band, self.quantile_hf, self.threshold_band, self.threshold_hf, self.window_size, self.spread],
                          ["quantile_band", "quantile_hf", "threshold_band", "threshold_hf", "window_size", "spread"],
                          ["Value"])
        df.index.name = 'Constant'
        if os.path.isfile(self.constant_location):
            assert self.constant_location.endswith("_constants.xlsx")
            os.remove(self.constant_location)
        df.to_excel(self.constant_location)

    def adjust_window(self):
        """Constant adjustment window slider setup

        Parameters
        ----------
        quantile_band: float
        quantile_hf: float
        threshold_band: float
        threshold_hf: float
        window_size: int
        spread: float

        Returns
        -------
        Updated input values in same order or None of no changes
        """
        fig, axs = plt.subplots(1, 6, sharex=True, figsize=(15, 15))

        quantile_band_slider = Slider(
            ax=axs[0],
            label="Bandpass quantile",
            valmin=self.quantile_band - 0.15,
            valmax=self.quantile_band + 0.15,
            valinit=self.quantile_band,
            orientation="vertical"
        )
        quantile_hf_slider = Slider(
            ax=axs[1],
            label="Highpass quantile",
            valmin=self.quantile_hf - 0.02,
            valmax=self.quantile_hf + 0.02,
            valinit=self.quantile_hf,
            orientation="vertical"
        )
        window_size_slider = Slider(
            ax=axs[2],
            label="Rolling window size",
            valmin=0,
            valmax=20,
            valinit=self.window_size,
            orientation="vertical",
            valstep=2
        )
        threshold_band_slider = Slider(
            ax=axs[3],
            label="Bandpass var threshold",
            valmin=0.0,
            valmax=self.threshold_band * 10,
            valinit=self.threshold_band,
            orientation="vertical"
        )
        threshold_hf_slider = Slider(
            ax=axs[4],
            label="Highpass var threshold",
            valmin=0.0,
            valmax=self.threshold_hf * 10,
            valinit=self.threshold_hf,
            orientation="vertical"
        )
        spread_slider = Slider(
            ax=axs[5],
            label="Spread",
            valmin=0.0,
            valmax=5.0,
            valinit=self.spread,
            orientation="vertical"
        )

        # button = Button(axs[5], 'Reset d)

        # button = Button(axs[5], 'Reset defaultsefaults', color="red")
        # button.on_clicked(reset_defaults)

        change_window = ConstantChangerWindow(axs)
        plt.show()
        if change_window.changes == True:  # todo create set method
            # update constants from sliders
            self.quantile_band = quantile_band_slider.val
            self.quantile_hf = quantile_hf_slider.val
            self.window_size = window_size_slider.val
            self.spread = spread_slider.val
            self.threshold_band = threshold_band_slider.val
            self.threshold_hf = threshold_hf_slider.val
            self.edge = int(self.window_size / 2)
            self._verify_constants()
            self.save()

            # user text output
            if self.constant_saving_mode == 'none':
                print("Adjusted constants - NOT saving to file (saving mode set to none)!")
            elif self.constant_saving_mode == 'global':
                print("Adjusted constants - saving these constants for all animals!")
            elif self.constant_saving_mode == 'per_animal':
                print("Adjusted constants - saving these constants only for this animal!")
            elif self.constant_saving_mode == 'per_session_per_animal':
                print("Adjusted constants - saving these constants only for this session for animal!")
            elif self.constant_saving_mode == 'per_session':
                print("Adjusted constants - saving these constants for this session type across all animals!")

            return True
        else:
            print("Parameter changes rejected by user")
            return False