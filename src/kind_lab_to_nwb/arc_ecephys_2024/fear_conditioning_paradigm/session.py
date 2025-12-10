import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import mne
import numpy as np
import pandas as pd

from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.oscillations import OscillationsAnalyzer
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.raw_data_extraction import load_folder_to_array, load_events, load_dat_to_array
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.single_freezing_epochs_analysis import single_freezing_epochs_analysis
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.timings import TimingsProcessor


class Session:
    def __init__(self, params, animal, session_type):
        self.params = params
        self.animal = animal
        self.session_type = session_type
        self.session_dir = self.animal.animal_base_folder + self.session_type + '/'
        self.session_results_folder = self.session_dir + 'results/'
        if not os.path.exists(self.session_results_folder):
            os.mkdir(self.session_results_folder)

        self.update_results_filenames_with_prefix()
        self.load_data()

    def process(self):
        # self.analyze_freezing_epochs()
        self.compute_oscillations_analysis()

    def bad_periods_already_processed(self, marker: str = 'PROC_BAD_BUMPS_DONE') -> bool:
        anns = self.recording.annotations
        if anns is None or len(anns) == 0:
            return False
        descs = set(anns.description)
        # Either there are existing BAD bump annotations or we already marked done
        return any(d.startswith('BAD') for d in descs) or (marker in descs)

    @staticmethod
    def _mark_bad_periods_processed(self, marker: str = 'PROC_BAD_BUMPS_DONE'):
        # Zero-duration marker at t=0; does not affect Epochs rejection
        ann = mne.Annotations(onset=[0.0], duration=[0.0], description=[marker])
        self.recording.set_annotations(
            self.recording.annotations + ann if len(self.recording.annotations) else ann)

    @staticmethod
    def _parse_idx0_from_ch_name(ch_name: str) -> int | None:
        # Extract 0-based index from ..._CH### at the end
        m = re.search(r'_CH0*([0-9]+)$', ch_name, flags=re.IGNORECASE)
        return (int(m.group(1)) - 1) if m else None

    @staticmethod
    def _lfp_picks(info):
        return mne.pick_types(info, seeg=True, ecog=True, dbs=True,
                              eeg=True, meg=False, stim=False, misc=False, exclude=[])

    @staticmethod
    def _guess_meta_cols(df: pd.DataFrame):
        # Heuristics for metadata columns you don't want to overwrite
        candidates = [c for c in ['ID', 'Genotype', 'Folder', 'source_number'] if c in df.columns]
        # Always keep the first few columns as meta (safe guard)
        keep_first_n = min(3, len(df.columns))
        meta = list(df.columns[:keep_first_n])
        for c in candidates:
            if c not in meta:
                meta.append(c)
        # Deduplicate preserving order
        seen, meta_unique = set(), []
        for c in meta:
            if c not in seen:
                seen.add(c)
                meta_unique.append(c)
        return meta_unique

    @staticmethod
    def _find_row_index(df: pd.DataFrame, animal_info_row: pd.DataFrame):
        if animal_info_row is None or animal_info_row.empty:
            return None
        row = animal_info_row.iloc[0]
        # Try matching by a priority list of keys
        for key in ['ID', 'Folder', 'Animal', 'Animal_ID']:
            if key in df.columns and key in animal_info_row.columns:
                val = row[key]
                matches = df.index[df[key] == val].tolist()
                if matches:
                    return matches[0]
        # Fallback: try exact 'Folder' if present in animal object
        if 'Folder' in df.columns and hasattr(row, 'Folder'):
            matches = df.index[df['Folder'] == row['Folder']].tolist()
            if matches:
                return matches[0]
        return None

    def update_electrode_excel_from_raw(self, sheet_name=0, electrode_cols=None):
        """
        Mark bad channels as 'bad' in the electrode columns of the Excel row for this animal.
        Does NOT rename or overwrite good channels (i.e., no AREA..._CH00X written).
        Also updates in-memory animal.bad_channels and areas_animal_clean.
        """
        path_xlsx = self.params.path_info_electrodes
        if not os.path.exists(path_xlsx):
            raise FileNotFoundError(f"Electrode info Excel not found: {path_xlsx}")

        # Load Excel
        df = pd.read_excel(path_xlsx, sheet_name=sheet_name)

        # Find row for this animal
        idx = self._find_row_index(df, getattr(self.animal, 'animal_info', pd.DataFrame()))
        if idx is None:
            raise RuntimeError("Could not locate this animal's row in the electrode Excel (check ID/Folder match).")

        # Build LFP channel list and get bad set from MNE
        picks = self._lfp_picks(self.recording.info)  # include bads in order
        ch_names = [self.recording.ch_names[p] for p in picks]
        bads = set(self.recording.info.get('bads', []))

        # Decide which columns are electrode columns to update
        if electrode_cols is None:
            meta_cols = self._guess_meta_cols(df)
            electrode_cols = [c for c in df.columns if c not in meta_cols]

        # Only mark bads; do not overwrite good entries.
        n = min(len(ch_names), len(electrode_cols))
        for i in range(n):
            if ch_names[i] in bads:
                df.at[idx, electrode_cols[i]] = "bad"
            # else: leave as-is

        # Save back to Excel
        with pd.ExcelWriter(path_xlsx, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)

        # Update in-memory animal_info row to keep session object consistent
        if hasattr(self.animal, 'animal_info') and not self.animal.animal_info.empty:
            updated_row = df.loc[[idx], df.columns]
            self.animal.animal_info = updated_row.copy()

        # Update self.animal.bad_channels as 0-based indices parsed from channel names
        bad_idx0 = []
        for ch in bads:
            idx0 = self._parse_idx0_from_ch_name(ch)
            if idx0 is not None:
                bad_idx0.append(idx0)
        self.animal.bad_channels = sorted(set(bad_idx0))

        # Rebuild areas_animal_clean to reflect the current good channels
        self._rebuild_areas_clean_from_names()

    def _rebuild_areas_clean_from_names(self):
        """
        Rebuild self.animal.areas_animal_clean = {0: 'AREA_N', 1: 'AREA_M', ...}
        using current good LFP picks (exclude='bads') and channel names 'AREA_N_CH###'.
        """
        good_picks = mne.pick_types(self.recording.info, seeg=True, ecog=True, dbs=True,
                                    eeg=False, meg=False, stim=False, misc=False, exclude='bads')
        good_names = [self.recording.ch_names[p] for p in good_picks]
        areas = []
        for ch in good_names:
            # Extract 'AREA_N' from 'AREA_N_CH###'
            m = re.match(r'^(?P<area>.+)_CH0*\d+$', ch, flags=re.IGNORECASE)
            areas.append(m.group('area') if m else None)
        # Build 0..N-1 -> area mapping
        self.animal.areas_animal_clean = {i: (a if a is not None else f'UNK_{i}') for i, a in enumerate(areas)}

    def process_bad_channels_and_bad_epochs(self, duration=60.0):
        if self.params.data_type != "openephys":
            return

        # 1) Skip if already processed (existing BAD_* or marker)
        anns = self.recording.annotations
        if len(anns):
            descs = set(anns.description)
            already = any(d.startswith('BAD') for d in descs) or ('PROC_BAD_PERIODS_DONE' in descs)
            if already:
                return

        # 2) Ensure channel types so picks behave
        types = {}
        types.update({ch: 'misc' for ch in self.recording.ch_names if ch.upper().startswith(('AUX', 'ACC', 'MOTION'))})
        types.update({ch: 'stim' for ch in self.recording.ch_names if ch.upper().startswith(('STI', 'TRIG', 'STIM'))})
        self.recording.set_channel_types(types)

        # 3) Build LFP picks (include bads in view so you can toggle them)
        lfp_picks = mne.pick_types(
            self.recording.info,
            seeg=True, ecog=True, dbs=True, eeg=True,
            meg=False, stim=False, misc=True, exclude=[]
        )
        if len(lfp_picks) == 0:
            raise RuntimeError("No LFP channels picked. Check channel types (seeg/ecog/dbs).")

        # 4) Optional: seed preset BAD_* labels in the annotation dropdown (zero-duration templates)
        existing = set(self.recording.annotations.description)
        # fix: tuple, not a string iterable
        presets_to_add = [lab for lab in ('bad_',) if lab not in existing]
        if presets_to_add:
            tmpl = mne.Annotations(
                onset=[0.0] * len(presets_to_add),
                duration=[0.0] * len(presets_to_add),
                description=presets_to_add
            )
            self.recording.set_annotations(
                self.recording.annotations + tmpl if len(self.recording.annotations) else tmpl)

        # 6) Open the browser for manual marking
        print("Browser instructions: 'b' toggles bad channel; 'a' enters annotation mode; "
              "use a bad_* label for bad segments; close window to continue.")
        self.recording.plot(picks=lfp_picks, duration=float(duration), n_channels=min(32, len(lfp_picks)),
                            scalings=dict(seeg='auto', ecog='auto', dbs='auto', misc='auto'), block=True,
                            precompute='auto')

        self.update_electrode_excel_from_raw()

        # Update bad_channels
        bad_idx0 = []
        pat_idx = re.compile(r'_CH0*([0-9]+)$', flags=re.IGNORECASE)
        for ch in self.recording.info.get('bads', []):
            m = pat_idx.search(ch)
            if m:
                bad_idx0.append(int(m.group(1)) - 1)
        self.animal.bad_channels = sorted(set(bad_idx0))

        # Optionally rebuild areas_animal_clean to match current good picks (if you renamed channels to AREA_N_CH###)
        if hasattr(self, '_rebuild_areas_clean_from_names'):
            self._rebuild_areas_clean_from_names()

        # 10) Add a processed marker (zero duration) so we can skip next time
        done = mne.Annotations(onset=[0.0], duration=[0.0], description=['PROC_BAD_PERIODS_DONE'])
        self.recording.set_annotations(self.recording.annotations + done if len(self.recording.annotations) else done)
        base = self.session_results_paths[self.session_type]['preprocessed_data']
        raw_fif = base['lfp'] + '_raw.fif'
        self.recording.save(raw_fif, fmt='single', split_size='2GB', overwrite=True)


    def notch_filter_lfp(self, copy=False, min_width_hz=0.5):
        """IIR notch filter LFP channels in self.raw using MNE's built-in notch_filter.

        - Honors raw.info['bads'] by excluding them from picks
        - Targets fundamental and harmonics up to Nyquist
        - Notch widths derived from Q (width ≈ f / Q), with optional minimum width clamp
        """
        raw = self.recording.copy() if copy else self.recording
        sfreq = float(raw.info['sfreq'])
        f0 = float(self.params.notch_frequency)
        Q = float(self.params.quality_factor)
        assert f0 < sfreq / 2, "notch_frequency must be < Nyquist"

        # Harmonics up to Nyquist
        freqs = np.arange(f0, sfreq / 2, f0)
        if freqs.size == 0:
            return raw if copy else None

        # Widths from Q (Hz); clamp to avoid ultra-narrow unstable notches
        widths = np.maximum(freqs / Q, float(min_width_hz))

        # Pick only LFP channels, exclude bads
        # Adjust depending on how you typed them ('seeg'/'dbs'/'ecog' typical for LFPs)
        lfp_picks = mne.pick_types(
            raw.info, seeg=True, dbs=True, ecog=True, eeg=False, meg=False,
            stim=False, misc=False, exclude='bads'
        )
        if len(lfp_picks) == 0:
            # fallback: name-based (e.g., channels starting with 'CH')
            lfp_picks = mne.pick_channels(raw.ch_names,
                                          include=[ch for ch in raw.ch_names if ch.startswith('CH')],
                                          exclude=raw.info['bads'])

        # IIR design parameters: Butterworth in SOS form is a good default
        iir_params = dict(order=getattr(self.params, 'iir_order', 4),
                          ftype='butter',
                          output='sos')

        raw.notch_filter(
            freqs=freqs,
            picks=lfp_picks,
            method='iir',
            iir_params=iir_params,
            notch_widths=widths,
            filter_length='auto',
            phase='zero'  # zero-phase via filtfilt-like application
        )

        self.recording = raw


    def _extract_frequencies_from_acc_data(self):
        frequencies = np.fft.fftfreq(len(self.accelerometer_data), 1 / self.params.sample_rate)
        frequencies = np.abs(frequencies)

        return frequencies

    def _lowpass_filtering_for_motion(self, frequencies):
        # Find the index corresponding to the cutoff frequency
        x_axis = self.accelerometer_data[:, 0]
        y_axis = self.accelerometer_data[:, 1]
        z_axis = self.accelerometer_data[:, 2]

        # Compute the corresponding frequencies
        fft_x = np.fft.fft(x_axis)
        fft_y = np.fft.fft(y_axis)
        fft_z = np.fft.fft(z_axis)

        # Apply cutoff frequency to filter the signal

        cutoff_index = np.abs(frequencies - self.params.motion_processing_cutoff_freq).argmin()

        # Set the magnitude spectrum values above the cutoff index to zero
        fft_x_filtered = fft_x.copy()
        fft_x_filtered[cutoff_index:] = 0
        fft_y_filtered = fft_y.copy()
        fft_y_filtered[cutoff_index:] = 0
        fft_z_filtered = fft_z.copy()
        fft_z_filtered[cutoff_index:] = 0
        # Inverse FFT to obtain the filtered signals
        x_axis_filtered = np.fft.ifft(fft_x_filtered)
        y_axis_filtered = np.fft.ifft(fft_y_filtered)
        z_axis_filtered = np.fft.ifft(fft_z_filtered)

        x_axis_filtered = np.real(x_axis_filtered)
        y_axis_filtered = np.real(y_axis_filtered)
        z_axis_filtered = np.real(z_axis_filtered)

        return x_axis_filtered, y_axis_filtered, z_axis_filtered

    def acceleration_magnitude_extraction_for_motion(self):
        if not np.isnan(self.accelerometer_data).all():
            frequencies = self._extract_frequencies_from_acc_data()

            x_axis_filtered_motion, y_axis_filtered_motion, z_axis_filtered_motion = self._lowpass_filtering_for_motion(
                frequencies)

            processed_acc_to_motion = pd.DataFrame(
                np.vstack([x_axis_filtered_motion, y_axis_filtered_motion, z_axis_filtered_motion]).transpose(),
                columns=['x_filtered', 'y_filtered', 'z_filtered'])

            # Normalize the filtered data
            processed_acc_to_motion['x_normalized'] = (processed_acc_to_motion['x_filtered'] - processed_acc_to_motion[
                'x_filtered'].mean()) / \
                                                      processed_acc_to_motion[
                                                          'x_filtered'].std()
            processed_acc_to_motion['y_normalized'] = (processed_acc_to_motion['y_filtered'] - processed_acc_to_motion[
                'y_filtered'].mean()) / \
                                                      processed_acc_to_motion[
                                                          'y_filtered'].std()
            processed_acc_to_motion['z_normalized'] = (processed_acc_to_motion['z_filtered'] - processed_acc_to_motion[
                'z_filtered'].mean()) / \
                                                      processed_acc_to_motion['z_filtered'].std()

            # Calculate the magnitude of the acceleration vector
            processed_acc_to_motion['motion'] = np.sqrt(processed_acc_to_motion['x_normalized'].diff() ** 2 +
                                                        processed_acc_to_motion['y_normalized'].diff() ** 2 +
                                                        processed_acc_to_motion['z_normalized'].diff() ** 2)

            self.processed_motion = processed_acc_to_motion['motion']

        else:
            self.processed_motion = pd.Series(np.nan)

    def extract_ttl_data(
            self,
            stim_channel=None,
            cs_len_sec=30.0,
            sep_sec=30.0,
            tol_sec=2.0,
            flash_max_gap_sec=None,  # gap to stay within a cluster; auto-estimated if None
            min_pulses=5,  # minimum pulses per CS cluster
            min_rate_hz=2.0,  # pulse rate threshold to favor LED flicker clusters
            add_annotations=True,
            store_led_within_cs=True,
            auditory_label='AUDITORY',
            cs_label='cs',
            led_label='led'
    ):
        raw = self.recording
        sfreq = float(raw.info['sfreq'])

        # 1) Stim channel
        if stim_channel is None:
            picks = mne.pick_types(raw.info, stim=True)
            if len(picks) == 0:
                # fallback by name
                cand = [ch for ch in raw.ch_names if ch.upper().startswith(('STI', 'STIM', 'TRIG'))]
                if not cand:
                    self.led_data = False
                    return pd.DataFrame(columns=['onset', 'offset']), pd.DataFrame(columns=['onset', 'offset'])
                stim_channel = cand[0]
            else:
                stim_channel = raw.ch_names[picks[0]]

        # 2) Find steps; use onset column (index 2)
        steps = mne.find_stim_steps(raw, stim_channel=stim_channel)
        if steps is None or len(steps) == 0:
            self.led_data = False
            return pd.DataFrame(columns=['onset', 'offset']), pd.DataFrame(columns=['onset', 'offset'])

        steps = np.asarray(steps)
        onset_mask = steps[:, 2].astype(bool)
        onset_samples = steps[onset_mask, 0].astype(int)
        if onset_samples.size == 0:
            self.led_data = False
            return pd.DataFrame(columns=['onset', 'offset']), pd.DataFrame(columns=['onset', 'offset'])

        t = onset_samples / sfreq
        if t.size < 2:
            self.led_data = {'led_pulses_samples': onset_samples, 'led_pulses_times': t}
            return pd.DataFrame(columns=['onset', 'offset']), pd.DataFrame(columns=['onset', 'offset'])

        # 3) Estimate within-cluster gap if needed
        inter = np.diff(t)
        if inter.size == 0:
            self.led_data = {'led_pulses_samples': onset_samples, 'led_pulses_times': t}
            return pd.DataFrame(columns=['onset', 'offset']), pd.DataFrame(columns=['onset', 'offset'])

        if flash_max_gap_sec is None:
            base = np.median(inter[inter <= np.quantile(inter, 0.5)]) if inter.size > 4 else np.median(inter)
            flash_max_gap_sec = min(cs_len_sec / 4.0, max(0.5, 4.0 * float(base)))

        # 4) Cluster pulses by within-gap threshold
        jump_idx = np.where(np.diff(t) > float(flash_max_gap_sec))[0]
        starts = np.concatenate(([0], jump_idx + 1))
        ends = np.concatenate((starts[1:] - 1, [len(t) - 1]))

        cl_onsets = t[starts]
        cl_last = t[ends]
        cl_durs = cl_last - cl_onsets
        cl_npulses = (ends - starts + 1).astype(int)
        cl_rate = np.zeros_like(cl_durs)
        valid = cl_durs > 0
        cl_rate[valid] = (cl_npulses[valid] - 1) / cl_durs[valid]

        # 5) Visual run selection by spacing ~ cs_len + sep
        expected_gap = float(cs_len_sec + sep_sec)
        if len(cl_onsets) >= 2:
            gaps_between = np.diff(cl_onsets)
            good_transition = np.abs(gaps_between - expected_gap) <= float(tol_sec)
            best_len, best_start = 0, None
            curr_len, curr_start = 0, 0
            for i, ok in enumerate(good_transition):
                if ok:
                    if curr_len == 0:
                        curr_start = i
                    curr_len += 1
                else:
                    if curr_len > best_len:
                        best_len, best_start = curr_len, curr_start
                    curr_len = 0
            if curr_len > best_len:
                best_len, best_start = curr_len, curr_start

            if best_len >= 1:
                visual_idx = np.arange(best_start, best_start + best_len + 1, dtype=int)
            else:
                visual_idx = np.arange(len(starts), dtype=int)  # fallback
        else:
            visual_idx = np.arange(len(starts), dtype=int)

        # 6) Keep only clusters that look like LED flicker (rate, count, duration)
        keep = []
        for k in visual_idx:
            if cl_npulses[k] < int(min_pulses):
                continue
            if cl_rate[k] < float(min_rate_hz):
                continue
            if cl_durs[k] < (cs_len_sec - tol_sec):
                continue
            keep.append(k)
        sel = np.array(keep, dtype=int)

        # If still ambiguous, keep those with correct spacing neighbors
        if sel.size >= 1 and len(cl_onsets) >= 2:
            ok = np.zeros_like(sel, dtype=bool)
            for i, k in enumerate(sel):
                left_ok = (k - 1 >= 0) and (abs((cl_onsets[k] - cl_onsets[k - 1]) - expected_gap) <= tol_sec)
                right_ok = (k + 1 < len(cl_onsets)) and (
                        abs((cl_onsets[k + 1] - cl_onsets[k]) - expected_gap) <= tol_sec)
                ok[i] = left_ok or right_ok
            if ok.any():
                sel = sel[ok]

        # 7) Build CS intervals from selected clusters
        cs_intervals = []
        for k in sel:
            on_samp = int(round(cl_onsets[k] * sfreq))
            off_samp = on_samp + int(round(cs_len_sec * sfreq))
            cs_intervals.append((on_samp, off_samp))
        cs_df = pd.DataFrame(cs_intervals, columns=['onset', 'offset']).sort_values('onset').reset_index(drop=True)

        # 8) Determine AUDITORY clusters as the leading/trailing clusters outside visual run
        aud_df = pd.DataFrame(columns=['onset', 'offset'])
        if sel.size > 0:
            min_sel, max_sel = sel.min(), sel.max()
            aud_idx = []
            # Leading clusters < first visual
            lead = [k for k in range(0, min_sel) if k < len(starts)]
            # Trailing clusters > last visual
            trail = [k for k in range(max_sel + 1, len(starts))]
            # We expect 0..1 leading and 0..1 trailing; keep at most one from each side
            if len(lead) > 0:
                aud_idx.append(lead[0])
            if len(trail) > 0:
                aud_idx.append(trail[-1])
            if aud_idx:
                aud_intervals = []
                for k in aud_idx:
                    on_samp = int(round(cl_onsets[k] * sfreq))
                    off_samp = on_samp + int(round(cs_len_sec * sfreq))
                    aud_intervals.append((on_samp, off_samp))
                aud_df = pd.DataFrame(aud_intervals, columns=['onset', 'offset']).sort_values('onset').reset_index(
                    drop=True)

        # 9) LED instantaneous annotations only within CS intervals
        if store_led_within_cs and len(cs_df) > 0:
            cs_on = cs_df['onset'].to_numpy() / sfreq
            cs_off = cs_df['offset'].to_numpy() / sfreq
            # Build mask of pulses inside any CS interval
            led_mask = np.zeros_like(t, dtype=bool)
            for on, off in zip(cs_on, cs_off):
                led_mask |= (t >= on) & (t < off)
            t_cs = t[led_mask]
            if add_annotations and t_cs.size > 0:
                ann_led = mne.Annotations(onset=t_cs, duration=np.zeros_like(t_cs), description=[led_label] * len(t_cs))
                raw.set_annotations(raw.annotations + ann_led if len(raw.annotations) else ann_led)

        # 10) Add CS and AUDITORY annotations
        if add_annotations:
            if len(cs_df):
                on = cs_df['onset'].to_numpy() / sfreq
                du = (cs_df['offset'] - cs_df['onset']).to_numpy() / sfreq
                ann_cs = mne.Annotations(onset=on, duration=du, description=[cs_label] * len(cs_df))
                raw.set_annotations(raw.annotations + ann_cs if len(raw.annotations) else ann_cs)

            if len(aud_df):
                on_a = aud_df['onset'].to_numpy() / sfreq
                du_a = (aud_df['offset'] - aud_df['onset']).to_numpy() / sfreq
                ann_aud = mne.Annotations(onset=on_a, duration=du_a, description=[auditory_label] * len(aud_df))
                raw.set_annotations(raw.annotations + ann_aud if len(raw.annotations) else ann_aud)

        # 11) Persist and stash for downstream if desired
        self.led_data = {
            'led_pulses_samples': onset_samples,
            'led_pulses_times': t,
            'flash_max_gap_sec': float(flash_max_gap_sec),
            'cluster_onsets_sec': cl_onsets,
            'cluster_rate_hz': cl_rate,
            'n_clusters_total': int(len(starts)),
            'n_cs': int(len(cs_df)),
            'n_auditory': int(len(aud_df)),
        }
        self.cs_raw_timings = cs_df
        self.auditory_timings = aud_df

    @staticmethod
    def _sanitize_area(area: str) -> str:
        # Keep it filename/regex friendly
        a = str(area).strip()
        a = re.sub(r'\s+', '-', a)  # spaces -> hyphen
        a = re.sub(r'[^A-Za-z0-9_\-]+', '', a)  # remove other punctuation
        return a

    def apply_area_names(self, mapping_save_path=None, area_prefix_first=True):
        """
        Rename neural channels from CH### to AREA_CH### based on self.animal.areas_animal_clean.
        areas_animal_clean is assumed to be a dict mapping 0..(n_good-1) to area names (good channels only).
        Bad channels are left as-is but still marked in recording.info['bads'].
        """
        recording = self.recording
        # Picks for neural LFP channels (adjust types if needed)
        lfp_picks = mne.pick_types(recording.info, seeg=True, dbs=True, ecog=True, eeg=True,
                                   meg=False, stim=False, misc=False, exclude='bads')
        if len(lfp_picks) == 0:
            return

        # areas_animal_clean keys are 0..n_good-1 for clean channels
        clean_map = getattr(self.animal, 'areas_animal_clean', {})
        if not clean_map:
            return

        # If the recording is already renamed (contains an underscore AREA_ prefix), skip safely
        already_area_named = any(
            '_' in ch and not ch.upper().startswith(('AUX', 'ACC', 'STI', 'MOTION')) for ch in recording.ch_names)
        if already_area_named:
            # You can still export a mapping if desired
            if mapping_save_path:
                df = pd.DataFrame({
                    'ch_name': recording.ch_names,
                    'type': recording.get_channel_types()
                })
                df.to_csv(mapping_save_path, index=False)
            return

        # Build rename mapping for good (non-bad) LFP channels
        mapping = {}
        area_map_records = []

        # The clean_map indexes correspond to the ordering of lfp_picks (good-only)
        n_assign = min(len(clean_map), len(lfp_picks))
        for k in range(n_assign):
            area = self._sanitize_area(clean_map[k])
            idx = lfp_picks[k]
            old = recording.ch_names[idx]
            if area_prefix_first:
                new = f'{area}_{old}'
            else:
                new = f'{old}_{area}'
            # Avoid duplicates
            if new in recording.ch_names or new in mapping.values():
                # Fallback: add a counter
                i = 1
                base = new
                while new in recording.ch_names or new in mapping.values():
                    new = f'{base}_{i}'
                    i += 1
            mapping[old] = new
            area_map_records.append({'old_name': old, 'new_name': new, 'area': area, 'bad': False})

        # Bad channels (keep names, but record)
        for b in recording.info.get('bads', []):
            area_map_records.append({'old_name': b, 'new_name': b, 'area': None, 'bad': True})

        # Apply rename
        recording.rename_channels(mapping)

    def raw_data_preprocessing(self):
        base = self.session_results_paths[self.session_type]['preprocessed_data']
        recording_fif = base['lfp'] + '_raw.fif'

        if not os.path.exists(recording_fif) or self.params.redo_raw_data_extraction:
            self.extract_raw_acc_lpf_events()  # sets self.raw
            # Mark bads
            self.recording.info['bads'] = [f'CH{idx + 1:03d}' for idx in getattr(self.animal, 'bad_channels', [])
                                           if f'CH{idx + 1:03d}' in self.recording.ch_names]

            self.apply_area_names(self)

            self.recording.save(recording_fif, fmt='single', split_size='2GB', overwrite=True)
        else:
            self.recording = mne.io.read_raw_fif(recording_fif, preload=False, verbose=False)

        self.extract_ttl_data()

    def arrays_to_mne_raw(self,
                          data_lfp: np.ndarray,  # shape (n_samples, n_lfp_ch), in Volts
                          sfreq: float,
                          ch_names: Optional[Sequence[str]] = None,
                          ch_types: str | Sequence[str] = 'eeg',  # 'eeg' or 'seeg' etc.
                          meas_date: Optional[datetime] = None,
                          line_freq: Optional[float] = None,
                          data_aux: Optional[np.ndarray] = None,  # shape (n_samples, n_aux_ch), same sfreq
                          aux_names: Optional[Sequence[str]] = None,
                          aux_types: str | Sequence[str] = 'misc',
                          event_samples: Optional[np.ndarray] = None,  # sample indices, int
                          event_codes: Optional[np.ndarray] = None,  # ints
                          add_stim_channel: bool = True,
                          stim_name: str = 'STI 014'
                          ) -> mne.io.Raw:
        n_samp, n_ch = data_lfp.shape
        data = data_lfp.T  # MNE expects (n_channels, n_times)
        if ch_names is None:
            ch_names = [f'CH{i:03d}' for i in range(1, n_ch + 1)]
        if isinstance(ch_types, str):
            ch_types = [ch_types] * n_ch

        info = mne.create_info(ch_names, sfreq=sfreq, ch_types=ch_types)
        if line_freq is not None:
            info['line_freq'] = float(line_freq)

        raw = mne.io.RawArray(data, info, verbose=False)

        # measurement date (optional)
        if meas_date is not None:
            if meas_date.tzinfo is None:
                meas_date = meas_date.replace(tzinfo=timezone.utc)
            raw.set_meas_date(meas_date)

        # Optional AUX (must have same sfreq)
        if data_aux is not None:
            if data_aux.shape[0] != n_samp:
                raise ValueError("AUX data must have same number of samples as LFP.")
            n_aux = data_aux.shape[1]
            aux_names = aux_names or [f'AUX{i + 1}' for i in range(n_aux)]
            if isinstance(aux_types, str):
                aux_types = [aux_types] * n_aux
            aux_info = mne.create_info(aux_names, sfreq=sfreq, ch_types=aux_types)
            raw_aux = mne.io.RawArray(data_aux.T, aux_info, verbose=False)
            raw.add_channels([raw_aux], force_update_info=True)

        # Optional stim channel from TTLs
        if event_samples is not None and event_codes is not None:
            event_samples = np.asarray(event_samples, dtype=int)
            event_codes = np.asarray(event_codes, dtype=int)
            if add_stim_channel:
                stim = np.zeros((1, raw.n_times), dtype=np.int32)
                valid = (event_samples >= 0) & (event_samples < raw.n_times)
                stim[0, event_samples[valid]] = event_codes[valid]
                stim_info = mne.create_info([stim_name], sfreq=sfreq, ch_types=['stim'])
                raw_stim = mne.io.RawArray(stim, stim_info, verbose=False)
                raw.add_channels([raw_stim], force_update_info=True)
            else:
                onsets = event_samples / float(sfreq)
                ann = mne.Annotations(onset=onsets, duration=np.zeros_like(onsets),
                                      description=[str(c) for c in event_codes])
                raw.set_annotations(ann)

        return raw

    def get_sorted_channels(self, folder_path, ch_prefix='CH', session='0', source='100'):
        folder = Path(folder_path)
        src = str(int(float(source)))
        pat = re.compile(
            rf"^{re.escape(src)}_(?:{re.escape(ch_prefix)})(\d+)(?:_{re.escape(session)})?\.continuous$"
            if session != '0'
            else rf"^{re.escape(src)}_{re.escape(ch_prefix)}(\d+)\.continuous$"
        )
        chs = []
        for f in folder.glob("*.continuous"):
            m = pat.match(f.name)
            if m:
                chs.append(int(m.group(1)))
        return sorted(chs)

    def extract_raw_acc_lpf_events(self):

        if self.params.data_type == 'openephys':

            data_session_folder = os.listdir(self.session_dir)
            data_session_folder = [i for i in data_session_folder if i.startswith(str(self.animal.animal_name))]
            data_session_folder = [i for i in data_session_folder if '.' not in i][0]
            data_session_folder = self.session_dir + data_session_folder + '/Record Node {}/'.format(self.animal.src)

            # Load LFP
            header_lfp, lfp_data = load_folder_to_array(data_session_folder, source=self.animal.src)
            sfreq = float(header_lfp['sampleRate'])
            bit_volts = float(header_lfp['bitVolts'])
            lfp_v = lfp_data * bit_volts  # convert to Volts

            # Load AUX if present
            acc_v = None
            if any('AUX' in i for i in os.listdir(data_session_folder)):
                header_acc, accelerometer_data = load_folder_to_array(
                    data_session_folder, ch_prefix='AUX', source=self.animal.src
                )
                acc_sfreq = float(header_acc['sampleRate'])
                acc_bit = float(header_acc['bitVolts'])
                accelerometer_data = accelerometer_data * acc_bit
                if np.isclose(acc_sfreq, sfreq):
                    acc_v = accelerometer_data  # can add to same Raw
                else:
                    # Keep as a separate Raw (or resample to sfreq if you prefer to merge)
                    self.acc_recording = self.arrays_to_mne_raw(
                        data_lfp=accelerometer_data,
                        sfreq=acc_sfreq,
                        ch_names=[f'AUX{i + 1}' for i in range(accelerometer_data.shape[1])],
                        ch_types='misc'
                    )

            # Build event samples from Open Ephys events
            events = load_events(os.path.join(data_session_folder, 'all_channels.events'))
            # Sample indices relative to first timestamp of LFP
            event_samples = (events['timestamps'] - header_lfp['timestamps_0'][0]).astype(int)

            # Choose event codes
            # You currently do event_id * channel. Consider using a mapping, or pack bitfields if TTL lines differ.
            evt_codes = events['event_id'].astype(int)

            # Channel names (if you want numerically sorted CH names)
            lfp_chs = self.get_sorted_channels(data_session_folder, ch_prefix='CH', source=self.animal.src)
            lfp_names = [f'CH{c:03d}' for c in lfp_chs]

            raw = self.arrays_to_mne_raw(
                data_lfp=lfp_v,
                sfreq=sfreq,
                ch_names=lfp_names,
                ch_types='eeg',  # or 'seeg'/'ecog' depending on your probes
                line_freq=50,
                data_aux=acc_v,
                aux_names=[f'AUX{i + 1}' for i in range(acc_v.shape[1])] if acc_v is not None else None,
                aux_types='misc',
                event_samples=event_samples,
                event_codes=evt_codes,
                add_stim_channel=True,
                stim_name='STI 014'
            )

            self.recording = raw

        elif self.params.data_type == 'taini':
            filename = [i for i in os.listdir(self.session_dir) if i.endswith('.dat')][0]
            filename = self.session_dir + filename
            lfp_data = load_dat_to_array(filename)
            self.lfp_data = np.transpose(lfp_data)
            self.accelerometer_data = np.nan
            self.ttl_events = pd.Series([np.nan])
            self.sample_rate = 250.4  # WARNING Hardcoded

        else:
            print("Invalid data type!")

    def compute_motion_from_aux(self, cutoff_hz=None, aux_names=None,
                                source='auto', add_as_channel=True, out_name='motion'):
        """
        Compute a scalar motion trace from 3-axis AUX/ACC channels. If a motion channel
        already exists in the chosen Raw, skip computation and load it instead.
        """

        # Helper: find motion channel case-insensitively in a Raw
        def _find_motion_channel(raw, name):
            if raw is None:
                return None
            name_l = name.lower()
            for ch in raw.ch_names:
                if ch.lower() == name_l:
                    return ch  # return actual name in Raw
            return None

        motion_ch = _find_motion_channel(self.recording, out_name)

        if motion_ch is not None:
            # Populate processed_motion from existing channel and return
            data = self.recording.get_data(picks=[motion_ch]).ravel()
            times = self.recording.times
            self.processed_motion = pd.Series(data, index=times, name=motion_ch)
            return

        # 2) Determine three AUX axis channels
        if aux_names is None:
            aux_names = [ch for ch in self.recording.ch_names if ch.upper().startswith(('AUX', 'ACC'))]

        if not aux_names or len(aux_names) < 3:
            self.processed_motion = pd.Series(np.nan, name='motion')
            print('No auxillary channels found or they are ess than three')
            return

        aux_raw = self.recording.copy().pick_channels(aux_names).load_data()

        # Compute motion: filter the three axis of the accelerometer
        sfreq = float(aux_raw.info['sfreq'])
        if cutoff_hz is None:
            cutoff_hz = getattr(self.params, 'motion_processing_cutoff_freq', None)
        if cutoff_hz is not None:
            aux_raw.filter(l_freq=None, h_freq=float(cutoff_hz),
                           picks='all', method='fir', phase='zero', fir_window='hamming',
                           verbose=False)

        # Compute motion: z-score per axis, diff, magnitude
        data = aux_raw.get_data()  # (3, n_times)
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        normed = (data - mean) / std
        diff = np.diff(normed, axis=1, prepend=normed[:, :1])
        motion = np.sqrt((diff ** 2).sum(axis=0))  # (n_times,)

        times = aux_raw.times
        self.processed_motion = pd.Series(motion, index=times, name=out_name)

        if add_as_channel:
            info_m = mne.create_info([out_name], sfreq=sfreq, ch_types='misc')
            raw_motion = mne.io.RawArray(motion[np.newaxis, :], info_m, verbose=False)

            # Choose which Raw to attach to (mirror the source)

            if not self.recording.preload:
                self.recording.load_data()
            # Resample motion to target rate if needed
            if not np.isclose(self.recording.info['sfreq'], sfreq):
                raw_motion = raw_motion.copy().resample(self.recording.info['sfreq'])
                print('resampling aux signal to match neuronal signal sampling rate')
            # Align lengths
            n = min(raw_motion.n_times, self.recording.n_times)
            if raw_motion.n_times != n:
                raw_motion = mne.io.RawArray(raw_motion.get_data()[:, :n],
                                             mne.create_info([out_name], self.recording.info['sfreq'], ch_types='misc'))
            if self.recording.n_times != n:
                self.recording.crop(tmax=(n - 1) / self.recording.info['sfreq'])
            # Finally add
            self.recording.add_channels([raw_motion], force_update_info=True)

            self.recording.set_channel_types({out_name: 'misc'})
            base = self.session_results_paths[self.session_type]['preprocessed_data']
            raw_fif = base['lfp'] + '_raw.fif'
            self.recording.save(raw_fif, fmt='single', split_size='2GB', overwrite=True)

    def load_data(self):

        assert os.path.isdir(self.session_dir), ""

        self.raw_data_preprocessing()

        self.process_bad_channels_and_bad_epochs()

        self.compute_motion_from_aux()

        TimingsProcessor(self)

    def analyze_freezing_epochs(self):
        single_freezing_epochs_analysis(
            self.cleaned_good_lfp_filtered, self.animal.animal_name, self.animal.areas_animal_clean,
            self.timings['freezing'], self.params, self.session_results_folder
        )

    def compute_oscillations_analysis(self):
        # Initialize the ConnectivityAnalyzer
        connectivity_analyzer = OscillationsAnalyzer(self)
        connectivity_analyzer.process_pan_sessions()

        # if self.session_type == 'Baseline_tone_flash_hab':
        #     connectivity_analyzer.process_baseline()
        #
        if self.session_type == 'Recall':
            connectivity_analyzer.process_recall()
        #
        # elif self.session_type == 'Cond':
        #     connectivity_analyzer.process_cond()

    def _add_path_prefix(self, data, prefix):
        if isinstance(data, dict):
            return {key: self._add_path_prefix(value, prefix) for key, value in data.items()}
        elif isinstance(data, str):
            return os.path.join(prefix, data)
        else:
            raise ValueError("Expected a dictionary or string.")

    def update_results_filenames_with_prefix(self):
        self.session_results_paths = self._add_path_prefix(self.params.filenames, self.session_results_folder)
