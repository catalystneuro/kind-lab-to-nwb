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

import copy

import pandas as pd

from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.dual_band_peaks import dual_band_peaks_analysis


def detect_noisy_periods(channel):
    """Detect saturation in TAINI data

    Function made by paulrignanese and modified by domagoj-42
    """
    # Initialize variables
    noisy_periods = []
    in_noisy_period = False
    onset = None

    # Iterate through the first channel to detect noisy periods
    for i, value in enumerate(channel):
        if value < 2000 or value > 3000:  # Detect noise
            if not in_noisy_period:
                # Start of a new noisy period
                in_noisy_period = True
                onset = i
        else:
            if in_noisy_period:
                # End of a noisy period
                offset = i - 1
                noisy_periods.append({'onset': onset, 'offset': offset+1})
                in_noisy_period = False

    # Handle the case where the last period is noisy
    if in_noisy_period:
        noisy_periods.append({'onset': onset, 'offset': len(channel) - 1})

    # Convert the list of periods to a DataFrame
    noisy_df = pd.DataFrame(noisy_periods)

    return noisy_df

def dual_band_peaks_analysis_fear_cond_paradigm_integration(lfp_data_filtered, spect_data, bad_periods, sample_rate, animal_info_in, session_folder):
    """Interfaces dual_band_peaks method with fear conditioning paradigm code"""
    from params import AnalysisParams
    fear_paradigm_params = AnalysisParams()

    # to numpy
    input_data = lfp_data_filtered#.to_numpy()

    # detect spectrogram decimation
    spectrogram = spect_data["Sxx"]

    # animal folder format
    animal_info = copy.deepcopy(animal_info_in)
    animal_folder = animal_info['Folder'].to_numpy(copy=True)[0]
    animal_id = str(animal_info.iloc[0]['ID'])

    base_dir = fear_paradigm_params.base_dir

    backup_file = session_folder + "/results/seizures_backup"
    excel_save_location = f"{session_folder}/results/"


    seizures, nonseizures = dual_band_peaks_analysis(input_data, spectrogram, bad_periods, sample_rate, base_dir, animal_folder, session_folder, animal_id,
                                                     backup_file=backup_file, excel_dir=excel_save_location)

    return nonseizures

def dual_band_peaks_openephys_minimal_integration(data_array, spectrogram, sample_rate, constant_location, output_dir):
    """Interfaces dual_band_peaks method with minimal Openephys implementation"""


    backup_file = output_dir + "/seizures_backup"
    excel_save_location = output_dir

    seizures, nonseizures = dual_band_peaks_analysis(data_array, spectrogram, None, sample_rate, constant_location, "", None, None,
                                                     backup_file=backup_file, excel_dir=excel_save_location)

    return seizures

def dual_band_peaks_for_taini_integration(custom_raw, channel_name, data_file_path, constant_location, start_idx=None, end_idx=None):
    """This function loads TAINI data and overrides the param file with settings typical for TAINI files (take care when
    changing params.py, some are overriden here!)



    """
    # extract filename from path for use in backup file
    if data_file_path.endswith(".dat"):
        data_filename = data_file_path.split("/")[-1][:-4]
    else:
        raise ValueError

    base_dir = "/".join(data_file_path.split("/")[:-1]) + "/"
    backup_file = base_dir + data_filename + "_seizures_bkp"
    excel_save_location = base_dir

    all_channels_data = custom_raw.get_data()
    channel_index = custom_raw.ch_names.index(channel_name)
    single_channel_data = all_channels_data[channel_index, :]

    bad_periods = detect_noisy_periods(single_channel_data)

    seizures, nonseizures = dual_band_peaks_analysis(single_channel_data, None, bad_periods, 250.4, constant_location, "",
                                                     None, None,
                                                     excel_dir=excel_save_location, exclude_bad_from_input_sig=True,
                                                     start_idx_of_screening=start_idx, end_idx_of_screening=end_idx, merge_and_filter=True,
                                                     backup_file=backup_file)

    return seizures