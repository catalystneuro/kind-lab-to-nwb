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

# interface
verify_seizures = False         # event detection verification stage, show every detected seizure with choices to
                                # modify, reject, accept
save_seizure_spreadsheet = None
periods_len_minutes_seizure = 20        # duration per frame
start_idx_of_screening = None   # start of file if None
end_idx_of_screening = None # end of file if None
no_user_input = False

# Constants
live_constant_adjust = True         # allow adjusting detection constants while going through data
restart_after_adjust = True        # restart from beginning of file after adjusting constants
constant_saving_mode = 'global'     # how to save params after adjusting through interface
                                    # choose from: none, global, per_animal, per_session, per_session_per_animal

# seizure bands
f_band_low = 2          # bandpass lower frequency
f_band_high = 9         # bandpass upper frequency
f_hf_cut = 35          # highpass cutoff frequency

# merging to be done BEFORE anything is shown - WARNING: no seizures saved before this, you are losing details, but gain
# convinence when verifying seizures
merge_and_filter = False
merge_gap = 0.4 # seconds
short_cutoff = 0.4 # seconds

# power
use_power_threshold = True  # since this is a new addition to the algorithm, it may be disabled
window_power = 1.5 # seconds
threshold_power_factor = 0.75  # threshold is found as mean of high and low values in power. This factor multiplies the mean
pad_power_duration = 0.45    # how much to pad around threshold surpassing

# bad periods
exclude_bad_from_input_sig = False
remove_bad_from_detected = True

# params
quantile_band_default = 0.6
quantile_hf_default = 0.98772
threshold_band_default = 0.004
threshold_hf_default = 0.001
window_size_default = 4     # should be even integer
spread_default = 1.0

# additional dual_band_peaks algorithm params - may be useful
peak_dist = 0.035                   # minimum distance between peaks, seconds
                                    # to consider modifying if very finely seperated peaks are desirable or undesirable
verification_padding_time = 6       # seconds shown around frame during verification
remove_drift = True                 # should be true especially if gaps in data