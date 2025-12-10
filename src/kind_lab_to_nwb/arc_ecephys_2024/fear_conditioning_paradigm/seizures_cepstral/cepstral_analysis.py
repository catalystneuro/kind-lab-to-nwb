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

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.fftpack import ifft

matplotlib.use('TkAgg')

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

def cepstral_analysis(eeg_data, seizures_spects, sample_rate):
    matplotlib.use('TkAgg')

    # Select the relevant part of the spectrogram
    spect = seizures_spects['Sxx'][10:47]

    # Compute the cepstrum from the spectrogram
    ceps = np.real(ifft(spect, axis=0))  # Real part of the inverse FFT
    quefrency = np.arange(ceps.shape[0]) / sample_rate

    nonseizures = get_seizures_from_cepstral(seizures_spects, eeg_data, ceps, sample_rate)

    return nonseizures


def get_seizures_from_cepstral(spect_data, lfp_data_filtered, cepstral, sample_rate):
    matplotlib.use('TkAgg')

    periods_len_minutes = 20
    print('Session is {} minutes long'.format(len(lfp_data_filtered)/sample_rate/60))
    nb_periods = int(np.ceil(len(lfp_data_filtered)/sample_rate/60/periods_len_minutes))
    print('breaking down in {} periods of 20 minutes max.'.format(nb_periods))
    samples_per_interval = periods_len_minutes * 60 * sample_rate
    nonseizure_periods = []

    # Process each interval
    for i in range(nb_periods):
        start_idx = i * samples_per_interval
        end_idx = (i + 1) * samples_per_interval

        # Extract the interval from the DataFrame
        lfp_data_filtered_interval = lfp_data_filtered.iloc[start_idx:end_idx]

        # Extract the corresponding spectrogram data
        spect_interval_data = {
            "spect_time_index": spect_data["spect_time_index"][int(start_idx/1000):int(end_idx/1000)],
            "Sxx": spect_data["Sxx"][:, int(start_idx/1000):int(end_idx/1000)],
            "f": spect_data["f"]
        }

        interval_cepstral = cepstral[0:1, int(start_idx/1000):int(end_idx/1000)]

        time_extent = [spect_interval_data['spect_time_index'][0], spect_interval_data['spect_time_index'][-1]]

        fig, axs = plt.subplots(3, sharex=True, figsize=(25, 10))
        axs[0].imshow(spect_interval_data['Sxx'], cmap='inferno', aspect='auto', origin='lower',
                      extent=[time_extent[0], time_extent[1], 0, spect_interval_data['Sxx'].shape[0]])

        ax2 = axs[1].twinx()

        ax2.plot(np.linspace(spect_interval_data['spect_time_index'][0],
                             spect_interval_data['spect_time_index'][-1],
                             num=len(lfp_data_filtered_interval)),
                 lfp_data_filtered_interval.values + 20,
                 alpha=0.5, linewidth=0.5, color='cyan')

        ax2.set_ylabel('LFP Amplitude', color='cyan')

        ax2.tick_params(axis='y', labelcolor='cyan')

        # Set the background color to black for both axes
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Freq.')

        axs[1].imshow(interval_cepstral, cmap='inferno', aspect='auto', origin='lower',
                      extent=[time_extent[0], time_extent[1], 0, interval_cepstral.shape[0]])
        axs[1].set_ylabel('Quefr.')

        axs[2].plot(spect_interval_data['spect_time_index'], interval_cepstral[0])
        selector = ThresholdSelector(axs[2], interval_cepstral.mean(axis=0))

        [ax.set_xlim(spect_interval_data['spect_time_index'][0], spect_interval_data['spect_time_index'][-1]) for ax in axs]

        plt.show()

        cepst = pd.Series(interval_cepstral[0], index=spect_interval_data['spect_time_index'])

        if selector.selected_threshold is not None:

            current_period = None
            onset = None

            for timestamp, value in cepst.items():
                if value < selector.selected_threshold:
                    if current_period is None:
                        onset = timestamp
                    current_period = timestamp
                else:
                    if current_period is not None:
                        duration = current_period - onset
                        if duration > 0.875*2:
                            nonseizure_periods.append((onset, current_period))
                        current_period = None
                        onset = None

            if current_period is not None:
                duration = current_period - onset
                if duration > 0.875*2:
                    nonseizure_periods.append((onset, current_period))

    nonseizures_periods = pd.DataFrame(nonseizure_periods)
    if not nonseizures_periods.empty:
        nonseizures_periods.columns = ['onset', 'offset']

    return nonseizures_periods
