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

import h5py
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.widgets import SpanSelector
from tqdm import tqdm

from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.params import AnalysisParams, seizure_channel_helper
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.raw_data_extraction import extract_cs_onsets
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_cepstral.cepstral_analysis import cepstral_analysis
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.integrations import dual_band_peaks_analysis_fear_cond_paradigm_integration
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.sleep import sleep_detection_fear_paradigm_integration

params = AnalysisParams()


def detect_noisy_periods(df):
    # Initialize variables
    noisy_periods = []
    in_noisy_period = False
    onset = None

    # Iterate through the first channel to detect noisy periods
    for i, value in tqdm(enumerate(df.iloc[0])):
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
        noisy_periods.append({'onset': onset, 'offset': len(df.iloc[0]) - 1})

    # Convert the list of periods to a DataFrame
    noisy_df = pd.DataFrame(noisy_periods)

    return noisy_df


def extract_timings_and_cleanup_data(params, session_results_paths, animal_base_folder, session_type, led_data, folder,
                                     animal_info, spectrograms_dict, areas_dict_cleaned,
                                     processed_motion, lfp_data_filtered, session_folder):

    timings = {}
    if 'Cond' in session_type:
        last_sample = processed_motion.index[-1]

        timings['cs'] = extract_cs_timings(session_results_paths['cs_timings'], led_data, params.cond_cs_lengh,
                                           params.sample_rate)
        timings['cs_raw'] = timings['cs']
        timings['shock'] = timings['cs'].copy()
        timings['shock']['onset'] += params.cond_cs_lengh * params.sample_rate

        for i in range(len(timings['cs'])):
            timings[f'cs_{i + 1}'] = pd.DataFrame(timings['cs'].loc[i, :]).transpose()
            timings[f'shock_{i + 1}'] = timings[f'cs_{i + 1}'].copy()
            timings[f'shock_{i + 1}']['onset'] += params.cond_cs_lengh * params.sample_rate
            timings[f'shock_{i + 1}']['offset'] += params.cond_shock_length * params.sample_rate

            print(f'cs_{i+1}')

        timings['noncs'] = extract_opposite_epochs(timings['cs'], last_sample)

        for i in range(params.cond_cs_n + 1):
            timings[f'noncs_{i + 1}'] = pd.DataFrame(timings['noncs'].iloc[i + 1]).transpose()

    elif 'Recall' in session_type:
        timings['cs'] = extract_cs_timings(session_results_paths['cs_timings'], led_data, params.recall_cs_length,
                                           params.sample_rate)
        timings['cs_raw'] = timings['cs'].iloc[:params.recall_cs_n]
        last_sample = timings['cs_raw']['offset'].iloc[-1] + 30 * params.sample_rate

        for i in range(params.recall_cs_n):
            timings[f'cs_{i + 1}'] = pd.DataFrame(timings['cs'].iloc[i]).transpose()
            timings[f'cs_raw_{i + 1}'] = pd.DataFrame(timings['cs'].iloc[i]).transpose()

        timings['firsts_cs'] = timings['cs'].iloc[params.recall_firsts_cs[0]:params.recall_firsts_cs[1]]
        timings['middles_cs'] = timings['cs'].iloc[params.recall_middles_cs[0]:params.recall_middles_cs[1]]
        timings['lasts_cs'] = timings['cs'].iloc[params.recall_lasts_cs[0]:params.recall_lasts_cs[1]]

        timings['noncs'] = extract_opposite_epochs(timings['cs'], last_sample)

        for i in range(params.recall_cs_n + 1):
            timings[f'noncs_{i + 1}'] = pd.DataFrame(timings['noncs'].iloc[i + 1]).transpose()

        timings['cs'] = timings['cs'].iloc[:params.recall_cs_n]
        timings['noncs'] = timings['noncs'].iloc[:params.recall_cs_n + 2]  # need to have the pre-cs and the last noncs
    else:
        last_sample = processed_motion.index[-1]

    if params.data_type == "openephys":
        timings['bad_epochs'] = extract_bad_epochs(session_results_paths['bad_epochs_timings'], folder,
                                                   spectrograms_dict, lfp_data_filtered, areas_dict_cleaned,
                                                   params.sample_rate, params.redo_bad_epochs)
    elif params.data_type == "taini":
        timings['bad_epochs'] = detect_noisy_periods(lfp_data_filtered) # This is for when the taini recording system disconnects and fills missing values by saturated signal
        timings['bad_epochs'] = merge_close_epochs( timings['bad_epochs'], 250)

    if 'Recall' in session_type:
        new_row = {'onset': [timings['noncs'].iloc[-1]['offset']], 'offset': [
            last_sample]}  # this is in case of a session with more that 12 cs, ignore the rest of the session

        if new_row['offset'] > new_row['onset']:
            timings['bad_epochs'] = pd.concat([timings['bad_epochs'], pd.DataFrame(new_row)], ignore_index=True)
    elif 'Cond' in session_type:
        timings['bad_epochs'] = pd.concat([timings['bad_epochs'], timings['shock']])# to remove shocks artifacts

    if not timings['bad_epochs'].empty:
        timings['good_epochs'] = extract_opposite_epochs(timings['bad_epochs'], last_sample)
    else:
        timings['good_epochs'] = pd.DataFrame(columns=['onset', 'offset'], data=[[0, last_sample]])
        print(timings['good_epochs'])

    cleaned_lfp_data_filtered = remove_bad_regions_lfp(lfp_data_filtered.copy(), timings['bad_epochs'])
    cleaned_processed_motion = remove_bad_regions_series(processed_motion, timings['bad_epochs'])


    spectrograms_dict = remove_bad_regions_spectrograms(spectrograms_dict, timings['bad_epochs'])

    matplotlib.use('TkAgg')

    timings['nonseizure'] = extract_nonseizures_epochs(session_results_paths['nonseizure_timings'], spectrograms_dict,
                                                                            cleaned_lfp_data_filtered, timings['bad_epochs'], areas_dict_cleaned,
                                                       params.sample_rate, animal_info, params.seizures_detection_mode,
                                                       params.redo_seizures, session_folder)

    timings['freezing'] = extract_freezing_epochs(session_results_paths['freezing_timings'],
                                                  cleaned_processed_motion, params.threshold_motion_detection)

    timings['seizure'] = extract_opposite_epochs(timings['nonseizure'], last_sample)

    timings['nonfreezing'] = extract_opposite_epochs(timings['freezing'], last_sample)

    # currently REM and awake states are blank arrays as this has not been implemented yet!!
    print(session_results_paths.keys())
    if 'Seizure' in session_type:
        timings['sleep'] = extract_sleep_epochs(session_results_paths['sleep_timings'], cleaned_lfp_data_filtered,
                                                cleaned_processed_motion, spectrograms_dict,
                                                timings['freezing'], timings['bad_epochs'],
                                                areas_dict_cleaned, params.sample_rate, animal_info,
                                                "theta_delta_ratio", params.redo_sleep, session_folder)
        if len(timings['sleep']):
            timings['awake'] = extract_opposite_epochs(timings['sleep'], last_sample)
        else:
            timings['awake'] = timings['good_epochs']

        timings['sleep'] = extract_epochs_combination(timings['good_epochs'], timings['sleep'])
        timings['awake'] = extract_epochs_combination(timings['good_epochs'], timings['awake'])
        timings['awake'] = extract_epochs_combination(timings['awake'], timings['nonseizure'])

    # timings['seizure'] = extract_epochs_combination(timings['good_epochs'], timings['seizure'])

    timings['good_epochs'] = extract_epochs_combination(timings['good_epochs'], timings['nonseizure'])

    if 'Baseline' in session_type:
        timings['stim'] = extract_stim_timings(session_results_paths['stim_timings'], led_data, params.sample_rate)
        timings['nonstim'] = extract_opposite_epochs(timings['stim'], last_sample)
        timings['freezing_within_stim'] = extract_epochs_combination(timings['stim'], timings['freezing'])
        timings['nonfreezing_within_stim'] = extract_epochs_combination(timings['stim'], timings['nonfreezing'])
        timings['freezing_within_nonstim'] = extract_epochs_combination(timings['nonstim'], timings['freezing'])
        timings['nonfreezing_within_nonstim'] = extract_epochs_combination(timings['nonstim'], timings['nonfreezing'])

        timings['stim'] = extract_epochs_combination(timings['stim'], timings['good_epochs'])
        timings['nonstim'] = extract_epochs_combination(timings['nonstim'], timings['good_epochs'])
        timings['freezing_within_stim'] = extract_epochs_combination(timings['freezing_within_stim'], timings['good_epochs'])
        timings['nonfreezing_within_stim'] = extract_epochs_combination(timings['nonfreezing_within_stim'], timings['good_epochs'])
        timings['freezing_within_nonstim'] = extract_epochs_combination(timings['freezing_within_nonstim'], timings['good_epochs'])
        timings['nonfreezing_within_nonstim'] = extract_epochs_combination(timings['nonfreezing_within_nonstim'], timings['good_epochs'])

    if 'Recall' in session_type:
        excel_timings_file_path_recall = animal_base_folder + '/' + folder + '.xlsx'

        timings = extract_fine_timings_recall_xlsx(timings, excel_timings_file_path_recall, redo=True)

    elif 'Cond' in session_type:
        excel_timings_file_path_cond = animal_base_folder + '/' + folder + '_cond.xlsx'

        timings = extract_fine_timings_cond_xlsx(timings, excel_timings_file_path_cond, redo=False)

    if len(timings[
               'bad_epochs']) > 0:  # making sure that if there are bad_epochs, those are assigned as nans in lfp and motion
        assert cleaned_lfp_data_filtered.isna().any().any(), "Expected NaNs in cleaned_lfp_data_filtered when timings['bad_epochs'] is not empty"
        assert cleaned_processed_motion.isna().any(), "Expected NaNs in cleaned_processed_motion when timings['bad_epochs'] is not empty"

    return timings, cleaned_lfp_data_filtered, cleaned_processed_motion


def clip_epochs(df, boundary):
    df_clipped = df.copy()
    df_clipped['offset'] = df_clipped['offset'].clip(upper=boundary)
    df_clipped = df_clipped[df_clipped['onset'] <= boundary]

    return df_clipped


def extract_fine_timings_recall_xlsx(timings: object, excel_timings_file_path: object, redo=False) -> object:
    if redo or not os.path.exists(excel_timings_file_path):
        # Calculate onset and offset for 'pre_cs'
        first_onset_cs = timings['cs_raw']['onset'].iloc[0]
        pre_cs_onset = first_onset_cs - params.recall_pre_cs_length * params.sample_rate
        pre_cs_offset = first_onset_cs
        # Create dataframe for 'pre_cs'
        timings['pre_cs'] = pd.DataFrame({'onset': [pre_cs_onset], 'offset': [pre_cs_offset]})
        timings['freezing_within_pre_cs'] = extract_epochs_combination(timings['pre_cs'], timings['freezing'])
        timings['nonfreezing_within_pre_cs'] = extract_epochs_combination(timings['pre_cs'], timings['nonfreezing'])
        timings['freezing_within_cs'] = extract_epochs_combination(timings['cs'], timings['freezing'])
        timings['nonfreezing_within_cs'] = extract_epochs_combination(timings['cs'], timings['nonfreezing'])

        timings['freezing_within_noncs'] = extract_epochs_combination(timings['noncs'], timings['freezing'])
        timings['nonfreezing_within_noncs'] = extract_epochs_combination(timings['noncs'], timings['nonfreezing'])

        # num_cs_conditions = 12
        # num_noncs_conditions = 11
        for i in range(1, params.recall_cs_n + 1):
            cs_key = f'cs_{i}'
            timings[f'freezing_within_{cs_key}'] = extract_epochs_combination(timings[cs_key], timings['freezing'])
            timings[f'nonfreezing_within_{cs_key}'] = extract_epochs_combination(timings[cs_key], timings['nonfreezing'])

        for i in range(1, params.recall_cs_n + 2):
            timings[f'freezing_within_noncs_{i}'] = extract_epochs_combination(timings[f'noncs_{i}'],
                                                                               timings['freezing'])
            timings[f'nonfreezing_within_noncs_{i}'] = extract_epochs_combination(timings[f'noncs_{i}'],
                                                                                  timings['nonfreezing'])

        timings['freezing_within_firsts_cs'] = extract_epochs_combination(timings['firsts_cs'], timings['freezing'])
        timings['freezing_within_middles_cs'] = extract_epochs_combination(timings['middles_cs'], timings['freezing'])
        timings['freezing_within_lasts_cs'] = extract_epochs_combination(timings['lasts_cs'], timings['freezing'])
        timings['nonfreezing_within_firsts_cs'] = extract_epochs_combination(timings['firsts_cs'], timings['nonfreezing'])
        timings['nonfreezing_within_middles_cs'] = extract_epochs_combination(timings['middles_cs'], timings['nonfreezing'])
        timings['nonfreezing_within_lasts_cs'] = extract_epochs_combination(timings['lasts_cs'], timings['nonfreezing'])

        if not timings['bad_epochs'].empty:
            timings['good_epochs'] = filter_short_epochs(timings['good_epochs'], 10000)

            for timing_type in [i for i in timings.keys() if
                                i not in ['seizure', 'good_epochs', 'cs_raw', 'cs_raw_1', 'cs_raw_2',
                                          'cs_raw_3', 'cs_raw_4',
                                          'cs_raw_5', 'cs_raw_6', 'cs_raw_7', 'cs_raw_8', 'cs_raw_9', 'cs_raw_10']]:
                timings[timing_type] = extract_epochs_combination(timings['good_epochs'], timings[timing_type])

        for timing_type in [i for i in timings.keys() if
                            i not in ['seizure', 'cs_raw', 'cs_raw_1', 'cs_raw_2', 'cs_raw_3', 'cs_raw_4', 'cs_raw_5',
                                      'cs_raw_6', 'cs_raw_7', 'cs_raw_8', 'cs_raw_9', 'cs_raw_10', 'bad_epochs']]:


            timings[timing_type] = extract_epochs_combination(timings['nonseizure'], timings[timing_type])

        excel_writer = pd.ExcelWriter(excel_timings_file_path, engine='xlsxwriter')

        for timing_type in timings.keys():

            timings[timing_type].to_excel(excel_writer, sheet_name=timing_type, index=False)

        excel_writer._save()

    else:

        excel_file = pd.ExcelFile(excel_timings_file_path)

        for sheet_name in excel_file.sheet_names:

            df = pd.read_excel(excel_file, sheet_name)
            timings[sheet_name] = df

    return timings


def extract_fine_timings_cond_xlsx(timings: object, excel_timings_file_path: object, redo=False) -> object:

    if redo or not os.path.exists(excel_timings_file_path):

        first_onset_cs = timings['cs_raw']['onset'].iloc[0]
        pre_cs_onset = first_onset_cs - params.cond_pre_cs_length * params.sample_rate
        pre_cs_offset = first_onset_cs

        # Create dataframe for 'pre_cs'
        timings['pre_cs'] = pd.DataFrame({'onset': [pre_cs_onset], 'offset': [pre_cs_offset]})

        timings['freezing_within_pre_cs'] = extract_epochs_combination(timings['pre_cs'], timings['freezing'])

        timings['nonfreezing_within_pre_cs'] = extract_epochs_combination(timings['pre_cs'], timings['nonfreezing'])

        timings['freezing_within_cs'] = extract_epochs_combination(timings['cs'], timings['freezing'])
        timings['nonfreezing_within_cs'] = extract_epochs_combination(timings['cs'], timings['nonfreezing'])

        for i in range(1, params.cond_cs_n + 1):
            cs_key = f'cs_{i}'
            timings[f'freezing_within_{cs_key}'] = extract_epochs_combination(timings[cs_key], timings['freezing'])
            timings[f'nonfreezing_within_{cs_key}'] = extract_epochs_combination(timings[cs_key], timings['nonfreezing'])

        for i in range(1, params.cond_cs_n + 1):
            timings[f'freezing_within_noncs_{i}'] = extract_epochs_combination(timings[f'noncs_{i}'],
                                                                               timings['freezing'])
            timings[f'nonfreezing_within_noncs_{i}'] = extract_epochs_combination(timings[f'noncs_{i}'],
                                                                                  timings['nonfreezing'])

        if not timings['bad_epochs'].empty:
            timings['good_epochs'] = filter_short_epochs(timings['good_epochs'], 10000)
        #     timings['good_epochs'] = merge_close_epochs(timings['good_epochs'], 6000)

            for timing_type in [i for i in timings.keys() if i not in ['good_epochs']]:
                timings[timing_type] = extract_epochs_combination(timings['good_epochs'], timings[timing_type])


        for timing_type in [i for i in timings.keys() if i not in ['good_epochs', 'nonseizure', 'seizure']]:
            timings[timing_type] = extract_epochs_combination(timings['nonseizure'], timings[timing_type])

        excel_writer = pd.ExcelWriter(excel_timings_file_path, engine='xlsxwriter')

        for timing_type in timings.keys():
            timings[timing_type].to_excel(excel_writer, sheet_name=timing_type, index=False)

        excel_writer._save()

    else:
        excel_file = pd.ExcelFile(excel_timings_file_path)

        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name)
            timings[sheet_name] = df

    return timings


def extract_cs_timings(session_cs_timings_path, led_data, cs_length, sample_rate):
    if not os.path.exists(session_cs_timings_path):
        led_events = extract_cs_onsets(led_data['led_pulses_events'])
        cs_timings = pd.DataFrame(led_events['timepoints'].values, columns=['onset'])
        cs_timings['offset'] = cs_timings['onset'] + sample_rate * cs_length
        cs_timings.to_hdf(session_cs_timings_path, key='cs', mode='w')
        if len(cs_timings) <= 10:
            raise Warning( f"There are only {len(cs_timings)} CSs extracted from ttl channel")
    else:
        cs_timings = pd.read_hdf(session_cs_timings_path, 'cs')

    return cs_timings


def extract_stim_timings(session_cs_timings_path, led_data, sample_rate):
    if not os.path.exists(session_cs_timings_path):
        led_events = extract_cs_onsets(led_data['led_pulses_events'])
        cs_timings = pd.DataFrame(led_events['timepoints'].values, columns=['onset'])
        cs_timings['offset'] = cs_timings['onset'] + sample_rate * 10  # because all cs last 30 seconds
        cs_timings.to_hdf(session_cs_timings_path, key='cs', mode='w')

    else:
        cs_timings = pd.read_hdf(session_cs_timings_path, 'cs')

    return cs_timings


def extract_bad_epochs(session_bad_epochs_timings_path, folder, spectrograms_dict, lfp_data_filtered, areas_dict, sample_rate, redo):
    if not os.path.exists(session_bad_epochs_timings_path) or redo:
        bad_epochs = plot_spectrograms_areas_acc_select_bad_epochs(spectrograms_dict, lfp_data_filtered, areas_dict, sample_rate)
        bad_epochs.to_hdf(session_bad_epochs_timings_path, key='bad_regions', mode='w')
    else:
        bad_epochs = pd.read_hdf(session_bad_epochs_timings_path, key='bad_regions')

    return bad_epochs


def extract_led_data(ttl_events, data_type):
    led_data = {}

    led_data['led_pulses_events'] = pd.Series(ttl_events.loc[ttl_events == 1].index)
    if len(led_data['led_pulses_events']) > 1:
        if led_data['led_pulses_events'][1] - led_data['led_pulses_events'][
            0] > 1000:  # this is to handle the case where there are two pulses at the beginning
            led_data['first_led_event'] = ttl_events.loc[ttl_events == 1].index[1]
            led_data_dropped = led_data['led_pulses_events'].drop(led_data['led_pulses_events'].index[0])
            led_data['led_pulses_events'] = led_data_dropped.reset_index(drop=True)
        else:
            led_data['first_led_event'] = ttl_events.loc[ttl_events == 1].index[0]

    else:
        led_data = False

    return led_data


def remove_bad_regions_lfp(data, bad_regions):

    for _, boundary in bad_regions.iterrows():
        onset = np.searchsorted(data.columns, boundary['onset'])
        offset = np.searchsorted(data.columns, boundary['offset'])

        data.iloc[:, onset:offset] = np.nan

    return data


def remove_bad_regions_series(data, bad_regions):

    for _, boundary in bad_regions.iterrows():
        onset = np.searchsorted(data.index, boundary['onset'])
        offset = np.searchsorted(data.index, boundary['offset'])
        data.iloc[onset:offset] = np.nan

    return data


def remove_bad_regions_spectrograms(spects, bad_regions):

    for area, spect in spects.items():
        for _, boundary in bad_regions.iterrows():
            onset = np.searchsorted(spect['spect_time_index'], boundary['onset'])
            offset = np.searchsorted(spect['spect_time_index'], boundary['offset'])
            spect['Sxx'][:, onset:offset] = np.nan

    return spects


def extract_nonseizures_epochs(session_nonseizures_timings_path, spects, cleaned_lfp_data_filtered, bad_periods,
                               areas_dict, sample_rate, animal_info, analysis_mode,
                               redo, session_folder, channel_helper=seizure_channel_helper):
    matplotlib.use('TkAgg')
    if not os.path.exists(session_nonseizures_timings_path) or redo:

        print('Please specify the best channel number for seizures detection : {}'.format(areas_dict))
        
        if channel_helper:
            fig, axs = plt.subplots(1, len(areas_dict), figsize=(20, 5))
            t0 = 0
            t1 = 500 #s
            for i in range(len(areas_dict)):
                sig_all = cleaned_lfp_data_filtered.iloc[i]
                t0_s = np.min([t0 * sample_rate, len(sig_all)])
                t1_s = np.min([t1 * sample_rate, len(sig_all)])
                sig = sig_all[t0_s:t1_s]
                t = np.linspace(t0_s, t1_s, t1_s-t0_s)
                axs[i].plot(t, sig)
                axs[i].set_title(str(areas_dict[i]))
                axs[i].set_xlabel('Time/samples')
            axs[0].set_ylabel('Amplitude/a.u.')
            plt.show()

        seizures_channel_idx = input()
        seizures_channel_lfp = cleaned_lfp_data_filtered.iloc[int(seizures_channel_idx)]
        seizures_spect = spects[int(seizures_channel_idx)]
        if analysis_mode == 'cepstral':
            nonseizures_epochs_onset_offset = cepstral_analysis(seizures_channel_lfp, seizures_spect, sample_rate)
        elif analysis_mode == 'deep':
            nonseizures_epochs_onset_offset = cepstral_analysis(seizures_channel_lfp, seizures_spect, sample_rate)
        elif analysis_mode == 'dual_band_peaks':
            nonseizures_epochs_onset_offset = dual_band_peaks_analysis_fear_cond_paradigm_integration(seizures_channel_lfp, seizures_spect, bad_periods,
                                                                       sample_rate, animal_info, session_folder)

        with h5py.File(session_nonseizures_timings_path, 'w') as h5file:
            h5file.create_dataset('nonseizure_timings', data=nonseizures_epochs_onset_offset.to_records(index=False))
    else:
        with h5py.File(session_nonseizures_timings_path, 'r') as h5file:

            nonseizures_epochs_onset_offset = pd.DataFrame.from_records(h5file['nonseizure_timings'][:])

    return nonseizures_epochs_onset_offset

def extract_sleep_epochs(results_path, cleaned_lfp_data_filtered, cleaned_processed_motion, spectrograms_dict, motion_epochs, bad_epochs, areas_dict,
                        sample_rate, animal_info, sleep_detection_mode, redo, session_folder):
    if not os.path.exists(results_path) or redo:

        print('Please specify the best channel number for sleep detection : {}'.format(areas_dict))

        seizures_channel_idx = input()
        signal_channel = cleaned_lfp_data_filtered.iloc[int(seizures_channel_idx)]
        spect_channel = spectrograms_dict[int(seizures_channel_idx)]

        if sleep_detection_mode == "theta_delta_ratio":
            NREM = sleep_detection_fear_paradigm_integration(signal_channel, spect_channel, cleaned_processed_motion, motion_epochs, bad_epochs, sample_rate, animal_info, session_folder)

        with h5py.File(results_path, 'w') as h5file:
            h5file.create_dataset('sleep_timings', data=NREM.to_records(index=False))

    else:
        with h5py.File(results_path, 'r') as h5file:
            NREM = pd.DataFrame.from_records(h5file['sleep_timings'][:])

    return NREM

def extract_freezing_epochs(session_freezing_epochs_boundaries_path, cleaned_processed_motion, threshold, redo_motion=False):
    if not os.path.exists(session_freezing_epochs_boundaries_path) or redo_motion:

        freezing_epochs_onset_offset = split_thresholded_acc_data(cleaned_processed_motion, threshold,4000)
        with h5py.File(session_freezing_epochs_boundaries_path, 'w') as h5file:
            h5file.create_dataset('freezing_epochs', data=freezing_epochs_onset_offset.to_records(index=False))

    else:
        with h5py.File(session_freezing_epochs_boundaries_path, 'r') as h5file:
            freezing_epochs_onset_offset = pd.DataFrame.from_records(h5file['freezing_epochs'][:])

    return freezing_epochs_onset_offset


def extract_opposite_epochs(df, len_session):
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == ['onset', 'offset']
    assert len_session > 0

    reverse_intervals = []

    if not len(df):
        reverse_intervals.append({'onset': 0, 'offset': len_session})
        reverse_df = pd.DataFrame(reverse_intervals, columns=['onset', 'offset'])
        return reverse_df

    if df['onset'].iloc[0] > 0:
        reverse_intervals.append({'onset': 0, 'offset': df['onset'].iloc[0]})

    for i in range(len(df) - 1):
        reverse_intervals.append({'onset': df['offset'].iloc[i], 'offset': df['onset'].iloc[i + 1]})

    if df['offset'].iloc[-1] < len_session:
        reverse_intervals.append({'onset': df['offset'].iloc[-1], 'offset': len_session})

    reverse_df = pd.DataFrame(reverse_intervals, columns=['onset', 'offset'])

    return reverse_df


def extract_epochs_combination(filter_df1, filter_df2):
    assert isinstance(filter_df1, pd.DataFrame)
    assert isinstance(filter_df2, pd.DataFrame)

    assert filter_df1.columns.tolist() == ['onset', 'offset']
    assert filter_df2.columns.tolist() == ['onset', 'offset']

    extracted_periods = []

    for _, filter_period1 in filter_df1.iterrows():
        filter_df1_onset, filter_df1_offset = filter_period1['onset'], filter_period1['offset']

        for _, filter_period2 in filter_df2.iterrows():
            filter_df2_onset, filter_df2_offset = filter_period2['onset'], filter_period2['offset']

            common_onset = max(filter_df1_onset, filter_df2_onset)
            common_offset = min(filter_df1_offset, filter_df2_offset)

            if common_onset <= common_offset:
                extracted_periods.append({'onset': common_onset, 'offset': common_offset})

    if extracted_periods:
        result_df = pd.DataFrame(extracted_periods)
        result_df = result_df.sort_values(by='onset').reset_index(drop=True)

        prev_offset = result_df.loc[0, 'offset']
        for idx in range(1, len(result_df)):
            if result_df.loc[idx, 'onset'] <= prev_offset:
                result_df.loc[idx, 'onset'] = prev_offset + 1
            prev_offset = result_df.loc[idx, 'offset']

        return result_df
    else:
        return pd.DataFrame(columns=['onset', 'offset'])


def filter_short_epochs(df, min_duration):
    """
    Filter out epochs with duration shorter than min_duration.

    Parameters:
        df (DataFrame): DataFrame with "onset" and "offset" columns.
        min_duration (int): Minimum duration threshold for an epoch to be kept.

    Returns:
        filtered_df (DataFrame): DataFrame with short epochs removed.
    """

    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == ['onset', 'offset']
    assert min_duration > 0

    filtered_df = df[df['offset'] - df['onset'] >= min_duration].reset_index(drop=True)
    return filtered_df


def merge_close_epochs(df, max_gap):
    """
    Merge epochs that are close enough, where the gap between consecutive epochs
    is less than or equal to max_gap.

    Parameters:
        df (DataFrame): DataFrame with "onset" and "offset" columns.
        max_gap (int): Maximum gap allowed between epochs for merging.

    Returns:
        merged_df (DataFrame): DataFrame with close epochs merged.
    """
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == ['onset', 'offset']
    assert max_gap > 0
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

        merged_df = pd.DataFrame(merged_epochs).reset_index(drop=True)
    else:
        merged_df = df
    return merged_df


def plot_spectrograms_areas_acc_select_bad_epochs(spectrograms_dict, lfp_data_filtered, areas_dict, sample_rate, save=False):

    bad_regions = []

    periods_len_minutes = 20
    print('Session is {} minutes long'.format(lfp_data_filtered.shape[1]/sample_rate/60))
    nb_periods = int(np.ceil(lfp_data_filtered.shape[1]/sample_rate/60/periods_len_minutes))
    print('breaking down in {} periods of 20 minutes max.'.format(nb_periods))
    samples_per_interval = periods_len_minutes * 60 * sample_rate

    for i in range(nb_periods):
        start_idx = int(i * samples_per_interval)
        end_idx = (i + 1) * int(samples_per_interval)

        lfp_data_filtered_interval = lfp_data_filtered.iloc[:, start_idx:end_idx]

        spect_interval_data = {}
        for channel, spect_data in spectrograms_dict.items():
            spect_interval_data[channel] = {
                "spect_time_index": spect_data["spect_time_index"][int(start_idx/1000):int(end_idx/1000)],
                "Sxx": spect_data["Sxx"][:, int(start_idx/1000):int(end_idx/1000)],
                "f": spect_data["f"]
            }

        fig, axs = plt.subplots(len(areas_dict), 1, figsize=(20, 15), facecolor='black')
        fig.set_facecolor('black')

        axs = axs.flatten()

        for i_plot, i in enumerate(areas_dict.keys()):
            ax = axs[i_plot]  # Get the current subplot

            pcm = ax.pcolormesh(spect_interval_data[i]['spect_time_index'],
                                range(spect_interval_data[i]['Sxx'].shape[0]),
                                spect_interval_data[i]['Sxx'], shading='auto', cmap='inferno')
            ax.set_xlim(spect_interval_data[i]['spect_time_index'][0],
                        spect_interval_data[i]['spect_time_index'][-1])
            ax.set_ylabel(areas_dict[i], color='white', rotation=0)
            # ax.set_ylabel('Frequency (Hz)', color='white')
            ax.tick_params(axis='y', labelcolor='white')


            ax2 = ax.twinx()

            ax2.plot(np.linspace(spect_interval_data[i]['spect_time_index'][0],
                                 spect_interval_data[i]['spect_time_index'][-1],
                                 num=len(lfp_data_filtered_interval.iloc[i])),
                     lfp_data_filtered_interval.iloc[i].values+20,
                     alpha=0.3, linewidth=0.5, color='cyan')
            ax2.set_ylabel('LFP Amplitude', color='cyan')
            ax2.tick_params(axis='y', labelcolor='cyan')

            ax.set_facecolor('black')
            ax2.set_facecolor('black')
            ax.set_yticks([-10, -5, 0, 5, 10, 15, 20])
            ax.set_yticklabels(['Motion', '', '0', '5', '10', '15', '20'], color='white')
            ax.yaxis.label.set_color('white')
            # ax.set_xlim(cs_onsets[0] - 60 * sample_rate, cs_onsets[0] + 120 * sample_rate)
            # Set axis labels and title for the subplot
            ax.set_xlabel('Time (s)')
            # ax.set_ylabel('             Frequency (Hz)')
            # ax.set_title(areas_dict[i], color='white')  # Adjust the electrode number in the title

            ax.set_facecolor('k')

            cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])  # Position of the colorbar (adjust as needed)
            cbar = fig.colorbar(pcm, cax=cbar_ax, label='Power (a. u.)')
            cbar.ax.yaxis.label.set_color('white')  # Set color for colorbar label
            cbar.ax.yaxis.set_tick_params(color='white')  # Set color for colorbar ticks

        def onselect(xmin, xmax):
            region = np.array([xmin, xmax])
            print(xmin, xmax)
            bad_regions.append(region)
            for ax in axs:
                rect = plt.Rectangle((xmin, ax.get_ylim()[0]), xmax - xmin, ax.get_ylim()[1] - ax.get_ylim()[0],
                                     facecolor='red', alpha=0.5)
                ax.add_patch(rect)
            fig.canvas.draw()

        span_selector = SpanSelector(axs[0], onselect, 'horizontal', useblit=True,
                                     props=dict(alpha=0.5, facecolor='tab:blue'))
        plt.show()

        plt.close()

    bad_regions = pd.DataFrame(bad_regions)
    if not bad_regions.empty:
        bad_regions.columns = ['onset', 'offset']
        bad_regions['onset'] = bad_regions['onset'].apply(lambda x: max(x, 0))
        bad_regions['offset'] = bad_regions['offset'].apply(
            lambda x: min(x, spectrograms_dict[i]['spect_time_index'][-1] - 1))

    return bad_regions


def split_thresholded_acc_data(accel_data, threshold, min_duration):
    assert isinstance(accel_data, pd.Series)
    assert min_duration > 0

    if threshold is not None:
        periods = []
        current_period = None
        onset = None

        for timestamp, value in accel_data.items():
            if value < threshold:
                if current_period is None:
                    onset = timestamp
                current_period = timestamp
            else:
                if current_period is not None:
                    duration = current_period - onset
                    if duration >= min_duration:
                        periods.append((onset, current_period))
                    current_period = None
                    onset = None

        if current_period is not None:
            duration = current_period - onset
            if duration >= min_duration:
                periods.append((onset, current_period))

    freezing_periods = pd.DataFrame(periods)
    if not freezing_periods.empty:
        freezing_periods.columns = ['onset', 'offset']
    else:
        freezing_periods = pd.DataFrame(columns=['onset', 'offset'], data=[[np.nan, np.nan]])
    return freezing_periods
