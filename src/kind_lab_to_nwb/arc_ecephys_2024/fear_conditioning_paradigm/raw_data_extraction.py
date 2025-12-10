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
import time

import cv2
import numpy as np
import pandas as pd

from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.OpenEphys import loadContinuous, loadEvents
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.animate_imcoh.video_processing import get_frame_timestamps


def load_dat_to_array(filename):
    '''Load a .dat file by interpreting it as int16 and then de-interlacing the 16 channels'''
    number_of_channels = 16

    sample_datatype = 'int16'
    display_decimation = 1

    print("Loading " + filename)

    dat_raw = np.fromfile(filename, dtype=sample_datatype)

    step = number_of_channels * display_decimation

    dat_chans = [dat_raw[c::step] for c in range(number_of_channels)]

    data = [lst for lst in dat_chans if lst.size > 0]

    max_length = max(map(len, data))

    padded_data = [np.pad(lst, (0, max_length - len(lst)), 'constant', constant_values=0) for lst in data]

    data = np.array(padded_data)

    return data


def extract_raw_acc_lpf_events(animal, data_type, session_folder, src):

    if data_type == 'openephys':

        data_session_folder = os.listdir(session_folder)
        data_session_folder = [i for i in data_session_folder if i.startswith(str(animal))]
        data_session_folder = [i for i in data_session_folder if '.' not in i][0]
        data_session_folder = session_folder + data_session_folder + '/'

        # src = [i for i in os.listdir(data_session_folder) if '.continuous' in i][0][:3]

        header_lfp, lfp_data = load_folder_to_array(data_session_folder, source=src)
        # header_lfp, lfp_data = loadContinuous(data_session_folder, source=src)
        if any(['AUX' in i for i in os.listdir(data_session_folder)]):
            header_acc, accelerometer_data = load_folder_to_array(data_session_folder, ch_prefix='AUX',
                                                                  source=src)
            accelerometer_data = accelerometer_data * float(header_acc['bitVolts'])

        else:
            accelerometer_data = np.nan

        events = loadEvents(data_session_folder + 'all_channels.events')
        ttl_events = np.multiply(events['eventId'], events['channel'])
        ttl_events_timestamps = events['timestamps'] - header_lfp['timestamps_0'][0]
        sample_rate = int(header_lfp['sampleRate'])
        print('sample rate: ' + str(sample_rate))
        ttl_events = pd.Series(ttl_events, index=ttl_events_timestamps)
        lfp_data = lfp_data * float(header_lfp['bitVolts'])
        lfp_data = pd.DataFrame(lfp_data).transpose()

    elif data_type == 'taini':
        filename = [i for i in os.listdir(session_folder) if i.endswith('.dat')][0]
        filename = session_folder + filename
        lfp_data = load_dat_to_array(filename)
        lfp_data = np.transpose(lfp_data)
        accelerometer_data = np.nan
        ttl_events = pd.Series([np.nan])
        sample_rate = 250.4 # WARNING Hardcoded

    else:
        print("Invalid data type!")

    return lfp_data, accelerometer_data, ttl_events, sample_rate


def extract_raw_frame_rate_numbers_blue_intensities(video_path):
    frames_timestamps = get_frame_timestamps(video_path)
    # Load the video
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    print(frame_rate)

    if frame_rate == 30:  # for files where the fps extracted is 30, the actual frame rate is 60
        frames_timestamps = [i / 2 for i in frames_timestamps]
        frame_rate = frame_rate * 2
    elif frame_rate == 15:
        frame_rate = frame_rate * 2

    # Lists to store frame number and blue channel intensity
    frames_numbers = []
    blue_intensities = []

    correlations = []

    prev_frame = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        # Calculate blue channel intensity
        blue_intensity = frame[:, :, 0].mean()  # Blue channel is the first channel (index 0)

        # Remove the blue channel from the frame
        frame_without_blue = frame.copy()
        frame_without_blue[:, :, 0] = 0
        # Convert frame to grayscale for better correlation calculation
        frame_gray = cv2.cvtColor(frame_without_blue, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # Calculate correlation between previous frame and current frame
            correlation = np.corrcoef(prev_frame.ravel(), frame_gray.ravel())[0, 1]
            correlations.append(correlation)
        prev_frame = frame_gray

        # Store frame number and blue intensity
        frames_numbers.append(cap.get(cv2.CAP_PROP_POS_FRAMES))
        blue_intensities.append(blue_intensity)
    cap.release()

    return frame_rate, frames_timestamps, frames_numbers, blue_intensities, correlations


def extract_cs_onsets(time_series):
    led_events = pd.DataFrame([time_series, time_series.diff()], index=['timepoints', 'diff']).transpose()
    led_events.loc[0, 'diff'] = 10000  # to not have nans as first values after diff

    cs_onsets = led_events.loc[led_events.loc[:, 'diff'] > 401]

    return cs_onsets


def read_header(f):
    header = {}
    h = f.read(1024).decode().replace('\n', '').replace('header.', '')
    for i, item in enumerate(h.split(';')):
        if '=' in item:
            header[item.split(' = ')[0]] = item.split(' = ')[1]
    return header


def load_events(filepath):
    data = {}

    print('loading events...')

    f = open(filepath, 'rb')
    header = read_header(f)

    if float(header[' version']) < 0.4:
        raise Exception('Loader is only compatible with .events files with version 0.4 or higher')

    data['header'] = header

    index = -1

    channel = np.zeros(int(1e6))
    timestamps = np.zeros(int(1e6))
    sample_n = np.zeros(int(1e6))
    node_id = np.zeros(int(1e6))
    event_type = np.zeros(int(1e6))
    event_id = np.zeros(int(1e6))
    recording_number = np.zeros(int(1e6))

    while f.tell() < os.fstat(f.fileno()).st_size:
        index += 1

        timestamps[index] = np.fromfile(f, np.dtype('<i8'), 1)
        sample_n[index] = np.fromfile(f, np.dtype('<i2'), 1)
        event_type[index] = np.fromfile(f, np.dtype('<u1'), 1)
        node_id[index] = np.fromfile(f, np.dtype('<u1'), 1)
        event_id[index] = np.fromfile(f, np.dtype('<u1'), 1)
        channel[index] = np.fromfile(f, np.dtype('<u1'), 1)
        recording_number[index] = np.fromfile(f, np.dtype('<u2'), 1)

    data['channel'] = channel[:index]
    data['timestamps'] = timestamps[:index]
    data['event_type'] = event_type[:index]
    data['node_id'] = node_id[:index]
    data['event_id'] = event_id[:index]
    data['recording_number'] = recording_number[:index]
    data['sample_n'] = sample_n[:index]

    return data


def get_sorted_channels(folder_path, ch_prefix='CH', session='0', source='100'):
    if ch_prefix == 'CH':
        files = [f for f in os.listdir(folder_path) if '.continuous' in f
                 and '_' + ch_prefix in f
                 and str(int(float(source))) in f]
    elif ch_prefix == 'AUX':
        files = [f for f in os.listdir(folder_path) if
                 '.continuous' in f and ch_prefix in f and str(int(float(source))) in f]

    if session == '0':
        # files = [f for f in files if len(f.split('_')) == 3]
        files = [f for f in files if len(f.split('_')) == 2]
        if ch_prefix == 'CH':

            chs = sorted([int(f.split('_' + ch_prefix)[1].split('.')[0]) for f in files])

        elif ch_prefix == 'AUX':
            chs = sorted([int(f.split(ch_prefix)[1].split('.')[0]) for f in files])
    else:
        files = [f for f in files if len(f.split('_')) == 3
                 and f.split('.')[0].split('_')[2] == session]

        chs = sorted([int(f.split('_' + ch_prefix)[1].split('_')[0]) for f in files])

    return (chs)


def load_folder_to_array(folder_path, channels='all', ch_prefix='CH',
                         dtype=float, session='0', source='101'):
    '''Load continuous files in specified folder to a single numpy array. By default all
    CH continous files are loaded in numerical order, ordering can be specified with
    optional channels argument which should be a list of channel numbers.'''

    if channels == 'all':
        channels = get_sorted_channels(folder_path, ch_prefix, session, source)

    if session == '0':
        if ch_prefix == 'CH':
            # file_list = [source + '_RhythmData_' + chprefix + x + '.continuous' for x in map(str, channels)]
            file_list = [str(int(float(source))) + '_' + ch_prefix + x + '.continuous' for x in map(str, channels)]
        elif ch_prefix == 'AUX':
            # file_list = [source + '_RhythmData_' + chprefix + x + '.continuous' for x in map(str, channels)]
            file_list = [i for i in os.listdir(folder_path) if ch_prefix in i]
    else:
        file_list = [str(int(float(source))) + '_' + ch_prefix + x + '_' + session + '.continuous' for x in
                    map(str, channels)]

    t0 = time.time()
    num_files = 1

    channel_1_data = loadContinuous(os.path.join(folder_path, file_list[0]), dtype)
    header = channel_1_data['header']
    n_samples = len(channel_1_data['data'])
    n_channels = len(file_list)

    data_array = np.zeros([n_samples, n_channels], dtype)
    data_array[:, 0] = channel_1_data['data']

    for i, f in enumerate(file_list[1:]):
        data_continuous = loadContinuous(os.path.join(folder_path, f), dtype)
        data_array[:, i + 1] = data_continuous['data']
        header['timestamps_{}'.format(i)] = data_continuous['timestamps']
        num_files += 1

    print(''.join(('Avg. Load Time: ', str((time.time() - t0) / num_files), ' sec')))
    print(''.join(('Total Load Time: ', str((time.time() - t0)), ' sec')))

    return header, data_array


def readHeader(f):
    header = {}
    h = f.read(1024).decode().replace('\n', '').replace('header.', '')
    for i, item in enumerate(h.split(';')):
        if '=' in item:
            header[item.split(' = ')[0]] = item.split(' = ')[1]
    return header


def load_continuous(filepath, dtype=float):
    # constants
    NUM_HEADER_BYTES = 1024
    SAMPLES_PER_RECORD = 1024
    BYTES_PER_SAMPLE = 2
    RECORD_SIZE = 4 + 8 + SAMPLES_PER_RECORD * BYTES_PER_SAMPLE + 10  # size of each continuous record in bytes
    RECORD_MARKER = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255])

    # constants for pre-allocating matrices:
    MAX_NUMBER_Of_SPIKES = int(1e6)
    MAX_NUMBER_Of_RECORDS = int(1e6)
    MAX_NUMBER_Of_EVENTS = int(1e6)
    assert dtype in (float, np.int16), \
        'Invalid data type specified for loadContinous, valid types are float and np.int16'

    print("Loading continuous data...")

    ch = {}

    # read in the data
    f = open(filepath, 'rb')

    fileLength = os.fstat(f.fileno()).st_size

    # calculate number of samples
    recordBytes = fileLength - NUM_HEADER_BYTES
    if recordBytes % RECORD_SIZE != 0:
        raise Exception("File size is not consistent with a continuous file: may be corrupt")
    nrec = recordBytes // RECORD_SIZE
    nsamp = nrec * SAMPLES_PER_RECORD
    # pre-allocate samples
    samples = np.zeros(nsamp, dtype)
    timestamps = np.zeros(nrec)
    recordingNumbers = np.zeros(nrec)
    indices = np.arange(0, nsamp + 1, SAMPLES_PER_RECORD, np.dtype(np.int64))

    header = readHeader(f)

    recIndices = np.arange(0, nrec)

    for recordNumber in recIndices:

        timestamps[recordNumber] = np.fromfile(f, np.dtype('<i8'), 1)  # little-endian 64-bit signed integer
        N = np.fromfile(f, np.dtype('<u2'), 1)[0]  # little-endian 16-bit unsigned integer

        # print index

        if N != SAMPLES_PER_RECORD:
            raise Exception('Found corrupted record in block ' + str(recordNumber))

        recordingNumbers[recordNumber] = (np.fromfile(f, np.dtype('>u2'), 1))  # big-endian 16-bit unsigned integer

        if dtype == float:  # Convert data to float array and convert bits to voltage.
            data = np.fromfile(f, np.dtype('>i2'), N) * float(
                header['bitVolts'])  # big-endian 16-bit signed integer, multiplied by bitVolts
        else:  # Keep data in signed 16 bit integer format.
            data = np.fromfile(f, np.dtype('>i2'), N)  # big-endian 16-bit signed integer
        samples[indices[recordNumber]:indices[recordNumber + 1]] = data

        marker = f.read(10)  # dump

    # print recordNumber
    # print index

    ch['header'] = header
    ch['timestamps'] = timestamps
    ch['data'] = samples  # OR use downsample(samples,1), to save space
    ch['recordingNumber'] = recordingNumbers
    f.close()
    return ch
