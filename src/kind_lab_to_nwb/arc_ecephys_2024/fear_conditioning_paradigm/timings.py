import os
import re

import h5py
import mne
import numpy as np
import pandas as pd

from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.integrations import dual_band_peaks_analysis_fear_cond_paradigm_integration
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.seizures_sleep_detect.sleep import sleep_detection_fear_paradigm_integration
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.timings_processing import extract_opposite_epochs


class TimingsProcessor:
    def __init__(self, session):
        self.session = session
        self.sfreq = float(self.session.recording.info['sfreq'])

        if self.session.recording is not None:
            if not self.session.recording.preload:
                self.session.recording.load_data()

        if not self._all_base_timings_already_processed():

            self.session.timings = {}
            self.session.last_sample = int(self.session.recording.n_times - 1)

            self.process_freezing()
            self.process_seizures()

            if session.session_type == 'Seizure_screening':
                self.process_sleep()

            # For Recall: mark outside-of-interest periods as BAD_
            if session.session_type == 'Recall':
                self._add_recall_bounds_into_bad()
                self._annotate_complement(self.session.recording, base_label='cs', comp_label='noncs')
                self.annotate_recall_numbered(
                    cs_label='cs',
                    pre_len_sec=float(getattr(self.session.params, 'recall_pre_cs_length', 180.0)),
                    noncs_len_sec=30.0,
                    limit_n=int(getattr(self.session.params, 'recall_cs_n', 0)) or None
                )

            self._annotate_complement(self.session.recording, base_label='freezing', comp_label='nonfreezing')
            self._annotate_complement(self.session.recording, base_label='BAD_', comp_label='good')

            self.cleanup_annotations(unwanted_labels={'a', 'b', '_', 'd'})
            self.drop_duplicate_annotations(label='cs')  # or 'cs' if lower-case

            ann = mne.Annotations(onset=[0.0], duration=[0.0], description=['PROC_BASE_TIMINGS_DONE'])
            self.session.recording.set_annotations(
                self.session.recording.annotations + ann if len(self.session.recording.annotations) else ann
            )

            # Save so annotations persist
            out_path = self.session.session_results_paths[self.session.session_type]['preprocessed_data'][
                           'lfp'] + '_raw.fif'
            self.session.recording.save(out_path, overwrite=True)

        else:
            return

    def drop_duplicate_annotations(self, label: str, tol_sec=0.0):
        """
        Remove duplicate annotations with the same label and same (onset, duration),
        within an optional tolerance (tol_sec).
        """
        ann = self.session.recording.annotations
        if ann is None or len(ann) == 0:
            return
        desc = np.asarray(ann.description, dtype=object)
        on = np.asarray(ann.onset, float)
        du = np.asarray(ann.duration, float)

        mask = (desc == label) & (du > 0)
        if not mask.any():
            return

        # Build a key per annotation; include rounding to samples if you want strict sample-level dedup
        sf = float(self.session.recording.info['sfreq'])
        if tol_sec > 0:
            # round to nearest multiple of tol_sec
            k_on = np.round(on[mask] / tol_sec) * tol_sec
            k_du = np.round(du[mask] / tol_sec) * tol_sec
        else:
            # sample-accurate keys
            k_on = np.round(on[mask] * sf) / sf
            k_du = np.round(du[mask] * sf) / sf

        keys = np.array([f"{o:.9f}|{d:.9f}" for o, d in zip(k_on, k_du)], dtype=object)

        # keep first occurrence of each key
        _, keep_local_idx = np.unique(keys, return_index=True)
        keep_global = np.ones(len(ann), dtype=bool)
        # mark duplicates (within label) as False
        dup_local_idx = np.setdiff1d(np.arange(keys.size), keep_local_idx)
        # map back to global indices
        idx_global = np.flatnonzero(mask)
        keep_global[idx_global[dup_local_idx]] = False

        self.session.recording.set_annotations(ann[keep_global])

    @staticmethod
    def _show_epochs_on_raw(raw: mne.io.BaseRaw, epochs: mne.Epochs,
                            label='EPOCH', colorize=True, block=True,
                            picks=None, duration=100.0, n_channels=16):
        """
        Overlay the epoch windows as shaded annotations on the Raw viewer.
        Does not permanently modify the file (restores original annotations after plotting).
        """
        sf = float(raw.info['sfreq'])
        # Onset of each epoch on the raw time axis
        on_sec = epochs.events[:, 0] / sf + float(epochs.tmin)
        dur_sec = float(epochs.tmax - epochs.tmin)
        if dur_sec <= 0:
            raise ValueError("Epochs must have tmax > tmin to visualize windows.")

        # Clip onsets to [0, raw_duration - dur]
        on_sec = np.clip(on_sec, 0, max(0, raw.times[-1] - dur_sec))
        lab = [label] * len(on_sec)

        # Add temporary annotations
        ann_new = mne.Annotations(onset=on_sec, duration=np.full_like(on_sec, dur_sec, dtype=float),
                                  description=lab)
        ann_old = raw.annotations
        raw.set_annotations(ann_old + ann_new if len(ann_old) else ann_new)

        # Plot
        if picks is None:
            picks = mne.pick_types(raw.info, seeg=True, ecog=True, dbs=True, eeg=True,
                                   meg=False, stim=False, misc=False,
                                   exclude=['bad'])  # include misc if you want motion
        raw.plot(picks=picks, duration=float(duration),
                 n_channels=min(n_channels, len(picks)),
                 scalings=dict(seeg='auto', ecog='auto', dbs='auto', misc='auto'),
                 block=block)

        # Restore original annotations
        raw.set_annotations(ann_old)

    def _get_label_intervals_sec(self, raw: mne.io.BaseRaw, label: str) -> np.ndarray:
        """Return sorted Nx2 array of [onset_sec, offset_sec] for a given label (case-insensitive)."""
        ann = raw.annotations
        if ann is None or len(ann) == 0:
            return np.empty((0, 2), float)
        desc = np.asarray(ann.description, dtype=object)
        on = np.asarray(ann.onset, float)
        du = np.asarray(ann.duration, float)
        mask = (np.char.lower(desc.astype(str)) == label.lower()) & (du > 0)
        if not mask.any():
            return np.empty((0, 2), float)
        iv = np.c_[on[mask], on[mask] + du[mask]]
        iv = iv[np.argsort(iv[:, 0])]
        # merge overlaps/adjacent
        merged = []
        s0, e0 = iv[0]
        for s, e in iv[1:]:
            if s <= e0:
                e0 = max(e0, e)
            else:
                merged.append((s0, e0))
                s0, e0 = s, e
        merged.append((s0, e0))
        return np.asarray(merged, float)

    def _merge_intervals_sec(self, intervals: np.ndarray) -> np.ndarray:
        """Merge overlapping/adjacent intervals in seconds; returns Nx2 array."""
        if intervals is None or intervals.size == 0:
            return np.empty((0, 2), float)
        iv = intervals[np.argsort(intervals[:, 0])]
        merged = []
        s0, e0 = iv[0]
        for s, e in iv[1:]:
            if s <= e0:
                e0 = max(e0, e)
            else:
                merged.append((s0, e0))
                s0, e0 = s, e
        merged.append((s0, e0))
        return np.asarray(merged, float)

    def _get_label_intervals_sec(self, raw: mne.io.BaseRaw, label: str) -> np.ndarray:
        """Return sorted Nx2 [onset_sec, offset_sec] for a label (case-insensitive), merged."""
        ann = raw.annotations
        if ann is None or len(ann) == 0:
            return np.empty((0, 2), float)
        desc = np.asarray(ann.description, dtype=object).astype(str)
        on = np.asarray(ann.onset, float)
        du = np.asarray(ann.duration, float)
        mask = (np.char.lower(desc) == label.lower()) & (du > 0)
        if not mask.any():
            return np.empty((0, 2), float)
        iv = np.c_[on[mask], on[mask] + du[mask]]
        iv = iv[np.argsort(iv[:, 0])]
        # merge overlaps/adjacent
        merged = []
        s0, e0 = iv[0]
        for s, e in iv[1:]:
            if s <= e0:
                e0 = max(e0, e)
            else:
                merged.append((s0, e0))
                s0, e0 = s, e
        merged.append((s0, e0))
        return np.asarray(merged, float)

    def _remove_numbered_cs_noncs_annotations(self, raw: mne.io.BaseRaw):
        """Remove existing CS_n, NONCS_n and PRE_CS annotations to avoid duplicates."""
        ann = raw.annotations
        if ann is None or len(ann) == 0:
            return
        desc = np.asarray(ann.description, dtype=object).astype(str)
        # regex for CS_#, NONCS_#
        drop_mask = np.array([bool(re.fullmatch(r'(cs|noncs)_[0-9]+', d)) or d == 'pre_cs' for d in desc], bool)
        keep = ~drop_mask
        raw.set_annotations(ann[keep])

    def annotate_recall_numbered(self,
                                 cs_label='cs',
                                 pre_len_sec=None,  # defaults to self.session.params.recall_pre_cs_length
                                 noncs_len_sec=30.0,  # fixed NONCS length after each CS
                                 limit_n=None,  # defaults to self.session.params.recall_cs_n
                                 trim_epsilon='half-sample'  # tiny trim to avoid visual overlap
                                 ):
        raw = self.session.recording
        sf = float(raw.info['sfreq'])
        rec_len = raw.n_times / sf

        if pre_len_sec is None:
            pre_len_sec = float(getattr(self.session.params, 'recall_pre_cs_length', 180.0))
        if limit_n is None:
            limit_n = int(getattr(self.session.params, 'recall_cs_n', 0)) or None

        # Fetch and possibly limit CS intervals
        cs_iv = self._get_label_intervals_sec(raw, cs_label)
        if cs_iv.size == 0:
            return  # nothing to do

        if limit_n is not None:
            cs_iv = cs_iv[:min(limit_n, len(cs_iv))]

        # Remove previous numbered annotations to keep operation idempotent
        self._remove_numbered_cs_noncs_annotations(raw)

        # Optional epsilon to avoid visual overlap in the browser
        if trim_epsilon == 'half-sample':
            eps = 0.5 / sf
        elif trim_epsilon is None:
            eps = 0.0
        else:
            eps = float(trim_epsilon)

        new_on, new_du, new_desc = [], [], []

        # PRE_CS: [max(0, first_on - pre_len), first_on)
        first_on = float(cs_iv[0, 0])
        pre_on = max(0.0, first_on - pre_len_sec)
        pre_off = first_on
        if pre_off - pre_on > 0:
            # trim to avoid touching the CS rectangle
            pre_off_t = max(pre_on, pre_off - eps)
            pre_on_t = pre_on
            new_on.append(pre_on_t);
            new_du.append(pre_off_t - pre_on_t);
            new_desc.append('pre_cs')

        # CS_n and NONCS_n after each CS
        for i, (on_cs, off_cs) in enumerate(cs_iv, start=1):
            # CS_i exactly as base (optionally trimmed a bit to be visually disjoint from NONCS_i)
            on_cs_t = on_cs
            off_cs_t = max(on_cs_t, off_cs - eps) if eps > 0 else off_cs
            if off_cs_t - on_cs_t > 0:
                new_on.append(on_cs_t);
                new_du.append(off_cs_t - on_cs_t);
                new_desc.append(f'cs_{i}')

            # NONCS_i: 30 s immediately after CS_i
            on_nc = off_cs
            off_nc = min(rec_len, on_nc + float(noncs_len_sec))
            # trim start by eps to make it visually non-overlapping from CS
            on_nc_t = min(off_nc, on_nc + eps) if eps > 0 else on_nc
            if off_nc - on_nc_t > 0:
                new_on.append(on_nc_t);
                new_du.append(off_nc - on_nc_t);
                new_desc.append(f'noncs_{i}')

        # Add them as annotations
        if new_on:
            ann_new = mne.Annotations(onset=np.array(new_on, float),
                                      duration=np.array(new_du, float),
                                      description=new_desc)
            raw.set_annotations(raw.annotations + ann_new if len(raw.annotations) else ann_new)

    def _add_recall_bounds_into_bad(self):
        """
        For Recall sessions: extend existing BAD_ annotations by adding:
          - [0, first_CS_onset - recall_pre_cs_length]
          - [CS_N_offset + 30s, recording_end]
        This merges into one bad_label BAD label (prefer 'BAD', else first BAD_*), without creating a new label.
        """
        raw = self.session.recording
        sf = float(raw.info['sfreq'])
        rec_len = raw.n_times / sf

        # Parameters
        pre_len = float(getattr(self.session.params, 'recall_pre_cs_length', 180.0))  # seconds
        n_cs_req = int(getattr(self.session.params, 'recall_cs_n', 0))
        post_gap = 30.0  # seconds

        # CS intervals (seconds)
        cs_iv = self._get_label_intervals_sec(raw, 'cs')
        if cs_iv.size == 0:
            return  # nothing to do if no CS

        # Leading segment: [0, first_cs_on - pre_len]
        first_on = float(cs_iv[0, 0])
        lead_on, lead_off = 0.0, max(0.0, first_on - pre_len)

        # Trailing segment: [Nth_cs_off + post_gap, end]
        idx_last = min(max(n_cs_req, 1), len(cs_iv)) - 1  # 1-based -> 0-based; clamp
        nth_off = float(cs_iv[idx_last, 1])
        trail_on, trail_off = min(rec_len, nth_off + post_gap), rec_len

        new_iv = []
        if lead_off - lead_on > 0:
            new_iv.append((lead_on, lead_off))
        if trail_off - trail_on > 0:
            new_iv.append((trail_on, trail_off))
        if not new_iv:
            return
        new_iv = np.asarray(new_iv, float)

        # Choose a BAD label to update
        ann = raw.annotations
        desc = np.asarray(ann.description, dtype=object) if len(ann) else np.array([], object)

        bad_label = 'BAD_'  # create generic BAD if none exists

        # Collect existing intervals for the bad_label label (seconds)
        if len(ann):
            mask_chosen = np.array([isinstance(d, str) and d == bad_label for d in desc], bool)
            chosen_iv = np.c_[ann.onset[mask_chosen], ann.onset[mask_chosen] + ann.duration[
                mask_chosen]] if mask_chosen.any() else np.empty((0, 2), float)
        else:
            chosen_iv = np.empty((0, 2), float)

        # Merge new intervals with existing bad_label BAD intervals
        merged = self._merge_intervals_sec(np.vstack([chosen_iv, new_iv]) if chosen_iv.size else new_iv)

        # Replace existing bad_label BAD intervals with merged set (leave other BAD_* labels intact)
        if len(ann):
            keep = ~(np.array([isinstance(d, str) and d == bad_label for d in desc], bool))
            raw.set_annotations(ann[keep])

        if merged.size:
            on = merged[:, 0]
            du = merged[:, 1] - merged[:, 0]
            ann_new = mne.Annotations(onset=on, duration=du, description=[bad_label] * len(on))
            raw.set_annotations(raw.annotations + ann_new if len(raw.annotations) else ann_new)

    def _find_channel_casefold(self, name: str):
        if self.session.recording is None:
            return None
        name_l = name.lower()
        for ch in self.session.recording.ch_names:
            if ch.lower() == name_l:
                return ch
        return None

    def _freezing_already_processed(self) -> bool:
        anns = self.session.recording.annotations
        if anns is None or len(anns) == 0:
            return False
        descs = set(anns.description)
        return ('freezing' in descs) or ('PROC_FREEZING_DONE' in descs)

    def _mark_freezing_processed(self):
        ann = mne.Annotations(onset=[0.0], duration=[0.0], description=['PROC_FREEZING_DONE'])
        self.session.recording.set_annotations(
            self.session.recording.annotations + ann if len(self.session.recording.annotations) else ann)

    def _freezing_from_annotations(self) -> pd.DataFrame:
        anns = self.session.recording.annotations
        if anns is None or len(anns) == 0:
            return pd.DataFrame(columns=['onset', 'offset'])
        sfreq = float(self.session.recording.info['sfreq'])
        mask = np.array([d == 'freezing' for d in anns.description])
        if not mask.any():
            return pd.DataFrame(columns=['onset', 'offset'])
        on_samp = np.round(anns.onset[mask] * sfreq).astype(int)
        off_samp = np.round((anns.onset[mask] + anns.duration[mask]) * sfreq).astype(int)
        return (pd.DataFrame({'onset': on_samp, 'offset': off_samp})
                .sort_values('onset').reset_index(drop=True))

    def _detect_freezing_from_motion(self, motion: np.ndarray, sfreq: float,
                                     threshold: float, min_dur_sec: float) -> pd.DataFrame:
        """
        motion: 1D array, length n_times
        threshold: freeze if motion < threshold
        min_dur_sec: minimum duration in seconds to keep a segment
        Returns DataFrame with onset/offset in SAMPLES.
        """
        n = motion.size
        if n == 0 or threshold is None:
            return pd.DataFrame(columns=['onset', 'offset'])
        mask = motion < float(threshold)  # True where freezing candidate
        if not mask.any():
            return pd.DataFrame(columns=['onset', 'offset'])

        # Find contiguous True segments
        diff = np.diff(mask.astype(int), prepend=0, append=0)
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1) - 1
        assert len(starts) == len(ends)

        # Enforce min duration
        min_len = int(np.round(min_dur_sec * sfreq))
        keep = (ends - starts + 1) >= max(1, min_len)
        starts, ends = starts[keep], ends[keep]

        if len(starts) == 0:
            return pd.DataFrame(columns=['onset', 'offset'])
        return pd.DataFrame({'onset': starts.astype(int), 'offset': ends.astype(int)})

    def process_freezing(self):
        sfreq = self.sfreq

        # Skip if already processed
        if self._freezing_already_processed():
            # Load from annotations
            freezing = self._freezing_from_annotations()
            self.session.timings['freezing'] = freezing
            self.session.timings['nonfreezing'] = extract_opposite_epochs(freezing, self.session.last_sample)
            return

        motion_name = self._find_channel_casefold('motion')

        if motion_name is None:
            # No motion available; produce empty timing
            self.session.timings['freezing'] = pd.DataFrame(columns=['onset', 'offset'])
            return

        # 2) Get motion data
        motion = self.session.recording.get_data(picks=[motion_name]).ravel()

        # 3) Parameters: threshold and minimum duration
        threshold = float(self.session.params.threshold_motion_detection)
        # min duration: prefer seconds param; else fall back from samples; else default
        min_dur_sec = getattr(self.session.params, 'min_freezing_duration_sec', None)
        if min_dur_sec is None:
            min_dur_samp = getattr(self.session.params, 'min_freezing_duration', 4000)  # old default
            min_dur_sec = float(min_dur_samp) / sfreq

        # 4) Detect freezing intervals (in samples)
        freezing = self._detect_freezing_from_motion(motion, sfreq, threshold, min_dur_sec)

        # 5) Save as annotations (freezing)
        if len(freezing) > 0:
            onsets = freezing['onset'].to_numpy() / sfreq
            durations = (freezing['offset'] - freezing['onset']).to_numpy() / sfreq
            ann = mne.Annotations(onset=onsets, duration=durations, description=['freezing'] * len(freezing))
            self.session.recording.set_annotations(
                self.session.recording.annotations + ann if len(self.session.recording.annotations) else ann)

        # 6) Keep pandas tables for downstream
        self.session.timings['freezing'] = freezing
        self.session.timings['nonfreezing'] = extract_opposite_epochs(freezing, self.session.last_sample)

        # 7) Optional: persist a sidecar HDF (if you still want it)
        try:
            path = self.session.session_results_paths[self.session.session_type]['timings']['freezing']
            if path:
                # store as a compact table of samples
                freezing.to_hdf(path, key='freezing', mode='w')
        except Exception:
            pass

        # 8) Mark processed and save FIF so annotations persist
        self._mark_freezing_processed()
        if not self.session.recording.preload:
            self.session.recording.load_data()  # bring data into RAM
        self.session.recording.save(
            self.session.session_results_paths[self.session.session_type]['preprocessed_data']['lfp'] + '_raw.fif',
            overwrite=True)

    def _add_intervals_as_annotations(self, df: pd.DataFrame, label: str):
        """Add intervals given in SAMPLES to raw as Annotations with description=label."""
        if df is None or len(df) == 0:
            return
        sfreq = self.session.params.sample_rate
        onsets = (df['onset'].to_numpy() / sfreq).astype(float)
        durations = ((df['offset'] - df['onset']).to_numpy() / sfreq).astype(float)
        ann = mne.Annotations(onset=onsets, duration=durations, description=[label] * len(df))
        raw = self.session.recording
        raw.set_annotations(raw.annotations + ann if len(raw.annotations) else ann)

    @staticmethod
    def _annotate_complement(raw: mne.io.BaseRaw,
                             base_label: str,
                             comp_label: str,
                             clear_existing: bool = True,
                             trim_epsilon: float | None = 'half-sample') -> None:
        """
        Create disjoint complement annotations: comp_label = NOT(base_label),
        covering the entire recording exactly once when combined with base_label.

        - clear_existing: remove any previous comp_label annotations before adding new ones
        - trim_epsilon: shrink complement segments by a tiny epsilon to avoid visual overlap at boundaries.
          * 'half-sample'  -> use 0.5 / sfreq seconds per end
          * float value    -> use that many seconds per end
          * None           -> no trimming (may look overlapped due to float boundary)
        """
        sf = float(raw.info['sfreq'])
        n = int(raw.n_times)
        dur_total = n / sf

        # Optionally remove any existing comp_label annotations
        if clear_existing and len(raw.annotations):
            desc = np.asarray(raw.annotations.description, dtype=object)
            keep = desc != comp_label
            raw.set_annotations(raw.annotations[keep])

        # Extract base intervals (seconds), merge overlaps
        if len(raw.annotations) == 0:
            base_intervals = np.empty((0, 2), float)
        else:
            desc = np.asarray(raw.annotations.description, dtype=object)
            on = np.asarray(raw.annotations.onset, float)
            du = np.asarray(raw.annotations.duration, float)
            mask = (desc == base_label) & (du > 0)
            if not mask.any():
                base_intervals = np.empty((0, 2), float)
            else:
                iv = np.c_[on[mask], on[mask] + du[mask]]
                iv = iv[np.argsort(iv[:, 0])]
                # merge overlaps
                merged = []
                s0, e0 = iv[0]
                for s, e in iv[1:]:
                    if s <= e0:  # overlap/adjacent
                        e0 = max(e0, e)
                    else:
                        merged.append((s0, e0))
                        s0, e0 = s, e
                merged.append((s0, e0))
                base_intervals = np.array(merged, float)

        # Build complement of base over [0, dur_total]
        compl = []
        cur = 0.0
        for s, e in base_intervals:
            if cur < s:
                compl.append((cur, s))
            cur = max(cur, e)
        if cur < dur_total:
            compl.append((cur, dur_total))
        compl = np.array(compl, float) if compl else np.empty((0, 2), float)

        # Optional: trim complement ends by epsilon to avoid visual boundary overlap
        if trim_epsilon is not None and compl.size:
            eps = 0.5 / sf if trim_epsilon == 'half-sample' else float(trim_epsilon)
            # trim each side by eps
            compl[:, 0] += eps
            compl[:, 1] -= eps
            # drop non-positive durations
            keep = (compl[:, 1] - compl[:, 0]) > 0
            compl = compl[keep]

        # Add comp_label annotations (replace existing already handled above)
        if compl.size:
            on = compl[:, 0]
            du = compl[:, 1] - compl[:, 0]
            ann = mne.Annotations(onset=on, duration=du, description=[comp_label] * len(on))
            raw.set_annotations(raw.annotations + ann if len(raw.annotations) else ann)


    def add_overlap_metadata(self, epochs: mne.Epochs,
                             overlap_labels: dict[str, float]) -> None:
        """
        For each label in overlap_labels, compute per-epoch fraction of time overlapping that annotation.
        overlap_labels: {label: min_fraction_for_true}, e.g., {'FREEZING': 0.0} or {'FREEZING': 0.5}
        Adds columns: overlap_<label> (float in [0,1]) and <label>_any (bool per min_fraction) to epochs.metadata.
        """
        if epochs is None or len(epochs) == 0 or not overlap_labels:
            return
        sf = float(self.session.recording.info['sfreq'])
        # Pre-extract intervals per label in seconds
        desc = np.asarray(self.session.recording.annotations.description, dtype=object)
        on = np.asarray(self.session.recording.annotations.onset, float)
        du = np.asarray(self.session.recording.annotations.duration, float)
        label_intervals = {}
        for lab in overlap_labels:
            mask = (desc == lab) & (du > 0)
            if not mask.any():
                label_intervals[lab] = np.empty((0, 2), float)
                continue
            iv = np.c_[on[mask], on[mask] + du[mask]]
            iv = iv[np.argsort(iv[:, 0])]
            # merge overlaps
            merged = []
            s0, e0 = iv[0]
            for s, e in iv[1:]:
                if s <= e0:
                    e0 = max(e0, e)
                else:
                    merged.append((s0, e0));
                    s0, e0 = s, e
            merged.append((s0, e0))
            label_intervals[lab] = np.array(merged, float)

        # Compute per-epoch overlaps
        e_on = epochs.events[:, 0] / sf
        e_off = e_on + (epochs.tmax - epochs.tmin)
        meta = epochs.metadata.copy() if epochs.metadata is not None else pd.DataFrame(index=np.arange(len(epochs)))
        for lab, thr in overlap_labels.items():
            iv = label_intervals[lab]
            frac = np.zeros(len(epochs), float)
            for i, (a, b) in enumerate(zip(e_on, e_off)):
                if iv.size == 0:
                    frac[i] = 0.0
                    continue
                # sum of overlaps over all intervals of lab
                overlap = 0.0
                # two-pointer over sorted segments
                j = 0
                while j < len(iv) and iv[j, 1] <= a:
                    j += 1
                k = j
                while k < len(iv) and iv[k, 0] < b:
                    overlap += max(0.0, min(b, iv[k, 1]) - max(a, iv[k, 0]))
                    k += 1
                frac[i] = overlap / (b - a) if (b - a) > 0 else 0.0
            meta[f'overlap_{lab}'] = frac
            meta[f'{lab}_any'] = frac >= float(thr)
        epochs.metadata = meta

    @staticmethod
    def _summarize_annotation_excluding_within(
            raw: mne.io.BaseRaw,
            label: str,
            match: str = 'exact',  # 'exact' | 'prefix' | 'regex' (case-insensitive)
            drop_zero_duration: bool = True,
            exclude_prefixes: tuple[str, ...] = ('BAD_', 'seizure'),
            exclude_labels: tuple[str, ...] = (),  # extra exact labels to exclude
            percent_relative_to: str = 'recording'  # 'recording' | 'good' (good = recording minus excluded spans)
    ):
        """
        Summarize a label or an intersection 'A_within_B' on Raw, excluding BAD_* (or given exclusions).

        Returns a dict with:
          - label_found (bool): at least one matching span before exclusion
          - exists (bool): at least one span remains after exclusion
          - n_periods (int): number of non-overlapping periods after exclusion
          - total_duration_sec (float): total duration (merged, after exclusion)
          - coverage_sec (float): same as total_duration_sec (merged, disjoint)
          - percent_of_recording (float): coverage / (recording or 'good' time)
          - first_onset_sec, last_offset_sec
          - dur_stats_sec: dict(min/median/mean/max) after exclusion
          - intervals_sec: Nx2 array of [onset, offset] seconds after exclusion
          - intervals_samples: DataFrame onset/offset in samples (offset exclusive)
          - source: dict with optional source intervals (seconds) for diagnostics: {'A': iv_a, 'B': iv_b}
        """
        sf = float(raw.info['sfreq'])
        rec_len_sec = raw.n_times / sf
        ann = raw.annotations

        empty = dict(
            label_found=False, exists=False, n_periods=0,
            total_duration_sec=0.0, coverage_sec=0.0, percent_of_recording=0.0,
            first_onset_sec=None, last_offset_sec=None, dur_stats_sec={},
            intervals_sec=np.empty((0, 2), float),
            intervals_samples=pd.DataFrame(columns=['onset', 'offset']),
            source={'A': np.empty((0, 2), float), 'B': np.empty((0, 2), float)}
        )

        if ann is None or len(ann) == 0:
            return empty

        # utilities
        def merge_intervals(iv: np.ndarray) -> np.ndarray:
            if iv is None or iv.size == 0:
                return np.empty((0, 2), float)
            iv = iv[np.argsort(iv[:, 0])]
            out = []
            s0, e0 = iv[0]
            for s, e in iv[1:]:
                if s <= e0:
                    e0 = max(e0, e)
                else:
                    out.append((s0, e0))
                    s0, e0 = s, e
            out.append((s0, e0))
            return np.asarray(out, float)

        def subtract_intervals(base: np.ndarray, cut: np.ndarray) -> np.ndarray:
            """Return base \ cut (both merged, seconds)."""
            if base.size == 0:
                return np.empty((0, 2), float)
            if cut.size == 0:
                return base.copy()
            i = j = 0
            out = []
            while i < len(base):
                a, b = base[i]
                cur = a
                while j < len(cut) and cut[j, 1] <= a:
                    j += 1
                k = j
                while k < len(cut) and cut[k, 0] < b:
                    s, e = cut[k]
                    if s > cur:
                        out.append((cur, min(b, s)))
                    cur = max(cur, e)
                    if cur >= b:
                        break
                    k += 1
                if cur < b:
                    out.append((cur, b))
                i += 1
            if not out:
                return np.empty((0, 2), float)
            out = np.asarray(out, float)
            out = out[(out[:, 1] - out[:, 0]) > 0]
            return merge_intervals(out)

        def intervals_for_label(lab: str, how: str) -> np.ndarray:
            desc = np.asarray(ann.description, dtype=object).astype(str)
            on = np.asarray(ann.onset, float)
            du = np.asarray(ann.duration, float)
            desc_l = np.char.lower(desc)
            lab_l = lab.lower()
            if how == 'exact':
                mask = (desc_l == lab_l)
            elif how == 'prefix':
                mask = np.array([d.startswith(lab_l) for d in desc_l], dtype=bool)
            elif how == 'regex':
                rx = re.compile(lab, re.IGNORECASE)
                mask = np.array([bool(rx.fullmatch(d)) for d in desc], dtype=bool)
            else:
                raise ValueError("match must be 'exact', 'prefix', or 'regex'")
            if drop_zero_duration:
                du_mask = du > 0
                mask &= du_mask
            if not mask.any():
                return np.empty((0, 2), float)
            iv = np.c_[on[mask], on[mask] + du[mask]]
            return merge_intervals(iv)

        def intersect(iv1: np.ndarray, iv2: np.ndarray) -> np.ndarray:
            if iv1.size == 0 or iv2.size == 0:
                return np.empty((0, 2), float)
            i = j = 0
            out = []
            while i < len(iv1) and j < len(iv2):
                a1, b1 = iv1[i];
                a2, b2 = iv2[j]
                a = max(a1, a2);
                b = min(b1, b2)
                if b > a:
                    out.append((a, b))
                if b1 < b2:
                    i += 1
                else:
                    j += 1
            if not out:
                return np.empty((0, 2), float)
            return merge_intervals(np.asarray(out, float))

        # parse label (support A_within_B)
        src_A = np.empty((0, 2), float)
        src_B = np.empty((0, 2), float)
        if '_within_' in label.lower():
            a_lab, b_lab = label.split('_within_', 1)
            src_A = intervals_for_label(a_lab, match)
            src_B = intervals_for_label(b_lab, match)
            base_iv = intersect(src_A, src_B)
            label_found = (src_A.size > 0 and src_B.size > 0 and base_iv.size > 0)
        else:
            base_iv = intervals_for_label(label, match)
            label_found = base_iv.size > 0

        if base_iv.size == 0:
            return empty | dict(label_found=bool(label_found), source={'A': src_A, 'B': src_B})

        # build union of exclude intervals (seconds)
        desc = np.asarray(ann.description, dtype=object).astype(str)
        on = np.asarray(ann.onset, float);
        du = np.asarray(ann.duration, float)
        desc_l = np.char.lower(desc)
        ex_mask = np.zeros(len(desc), dtype=bool)
        for p in exclude_prefixes:
            p_l = p.lower()
            ex_mask |= np.array([d.startswith(p_l) for d in desc_l], dtype=bool)
        if exclude_labels:
            ex_mask |= np.isin(desc_l, [x.lower() for x in exclude_labels])
        if drop_zero_duration:
            ex_mask &= (du > 0)
        exclude_iv = merge_intervals(
            np.c_[on[ex_mask], on[ex_mask] + du[ex_mask]] if ex_mask.any() else np.empty((0, 2), float))

        # subtract excluded spans
        good_iv = subtract_intervals(base_iv, exclude_iv)
        if good_iv.size == 0:
            return dict(
                label_found=bool(label_found), exists=False, n_periods=0,
                total_duration_sec=0.0, coverage_sec=0.0, percent_of_recording=0.0,
                first_onset_sec=None, last_offset_sec=None, dur_stats_sec={},
                intervals_sec=np.empty((0, 2), float),
                intervals_samples=pd.DataFrame(columns=['onset', 'offset']),
                source={'A': src_A, 'B': src_B}
            )

        # stats
        durs = good_iv[:, 1] - good_iv[:, 0]
        total = float(durs.sum())  # merged -> no overlaps
        # denominator for percent
        if percent_relative_to == 'good':
            # total good time = recording minus excluded (global)
            good_total_iv = subtract_intervals(np.array([[0.0, rec_len_sec]], float), exclude_iv)
            good_total = float(np.sum(good_total_iv[:, 1] - good_total_iv[:, 0])) if good_total_iv.size else 0.0
            denom = good_total if good_total > 0 else rec_len_sec
        else:
            denom = rec_len_sec if rec_len_sec > 0 else 1.0
        pct = (total / denom) * 100.0 if denom > 0 else 0.0

        first_on = float(good_iv[0, 0]);
        last_off = float(good_iv[-1, 1])
        dur_stats = dict(min=float(np.min(durs)), median=float(np.median(durs)),
                         mean=float(np.mean(durs)), max=float(np.max(durs)))

        # sample-based (offset exclusive)
        on_samp = np.floor(good_iv[:, 0] * sf).astype(int)
        off_samp = np.ceil(good_iv[:, 1] * sf).astype(int)
        intervals_samples = pd.DataFrame({'onset': on_samp, 'offset': off_samp})

        return dict(
            label_found=True,
            exists=True,
            n_periods=int(len(good_iv)),
            total_duration_sec=total,
            coverage_sec=total,
            percent_of_recording=float(pct),
            first_onset_sec=first_on,
            last_offset_sec=last_off,
            dur_stats_sec=dur_stats,
            intervals_sec=good_iv,
            intervals_samples=intervals_samples,
            source={'A': src_A, 'B': src_B}
        )

    @staticmethod
    def _epochs_from_label(raw: mne.io.BaseRaw, label: str, window_sec: float | None,
                           picks=None, preload=True, reject_by_annotation=True,
                           min_duration_sec: float = 0.0,
                           event_repeated: str = 'drop',
                           exclude_labels: tuple[str, ...] = ('seizure',)
                           ) -> mne.Epochs | None:
        """
        Build Epochs directly from annotations.

        - label can be a base label (e.g., 'CS') or a composite 'A_within_B' (e.g., 'FREEZING_within_CS').
          In the latter case, epochs are defined on the intersection A ∩ B.
        - window_sec=None: use the native (constant) duration; raises if durations differ.
        - window_sec>0: tile each (base or intersected) interval into fixed windows of length window_sec.
        - min_duration_sec: drop intervals shorter than this (applied only to intersection/base segments before tiling).
        - reject_by_annotation: MNE will drop epochs overlapping BAD_* annotations.
        - exclude_labels: in addition, drop any epoch whose window overlaps these labels (case-insensitive),
          without modifying annotations (e.g., exclude 'seizure').
        """
        sf = float(raw.info['sfreq'])

        def _merge(iv: np.ndarray) -> np.ndarray:
            if iv is None or iv.size == 0:
                return np.empty((0, 2), float)
            iv = iv[np.argsort(iv[:, 0])]
            out = []
            s0, e0 = iv[0]
            for s, e in iv[1:]:
                if s <= e0:
                    e0 = max(e0, e)
                else:
                    out.append((s0, e0)); s0, e0 = s, e
            out.append((s0, e0))
            return np.asarray(out, float)

        def _intervals_for_label(lab: str) -> np.ndarray:
            ann = raw.annotations
            if ann is None or len(ann) == 0:
                return np.empty((0, 2), float)
            desc = np.asarray(ann.description, dtype=object).astype(str)
            on = np.asarray(ann.onset, float)
            du = np.asarray(ann.duration, float)
            mask = (np.char.lower(desc) == lab.lower()) & (du > 0)
            if not mask.any():
                return np.empty((0, 2), float)
            iv = np.c_[on[mask], on[mask] + du[mask]]
            return _merge(iv)

        def _intersect(iv1: np.ndarray, iv2: np.ndarray) -> np.ndarray:
            if iv1.size == 0 or iv2.size == 0:
                return np.empty((0, 2), float)
            i = j = 0;
            out = []
            while i < len(iv1) and j < len(iv2):
                a1, b1 = iv1[i];
                a2, b2 = iv2[j]
                a = max(a1, a2);
                b = min(b1, b2)
                if b > a: out.append((a, b))
                if b1 < b2:
                    i += 1
                else:
                    j += 1
            return _merge(np.asarray(out, float)) if out else np.empty((0, 2), float)

        # Base or composite intervals
        if '_within_' in label.lower():
            a_lab, b_lab = label.split('_within_', 1)
            iv_a = _intervals_for_label(a_lab)
            iv_b = _intervals_for_label(b_lab)
            iv = _intersect(iv_a, iv_b)
        else:
            iv = _intervals_for_label(label)

        # Drop very short intervals
        if iv.size and min_duration_sec > 0:
            iv = iv[(iv[:, 1] - iv[:, 0]) >= float(min_duration_sec)]
        if iv.size == 0:
            return None

        # Compute tmin/tmax and event onsets
        if window_sec is None:
            durs = iv[:, 1] - iv[:, 0]
            d0 = durs[0]
            if not np.allclose(durs, d0, atol=1.0 / sf):
                raise ValueError(f"Label '{label}' yields variable durations; set window_sec to tile.")
            tmin, tmax = 0.0, float(d0)
            onsets_sec = iv[:, 0]
        else:
            win = float(window_sec)
            tmin, tmax = 0.0, win
            onsets_sec = []
            for a, b in iv:
                L = b - a
                if L < win:
                    continue
                # non-overlapping tiling
                t = np.arange(a, b - win + 1e-12, win)
                onsets_sec.append(t)
            if len(onsets_sec) == 0:
                return None
            onsets_sec = np.concatenate(onsets_sec, axis=0)

        if onsets_sec.size == 0:
            return None

        # Build events with unique onsets
        on_samp = np.floor(onsets_sec * sf).astype(int)
        _, keep_idx = np.unique(on_samp, return_index=True)
        if keep_idx.size != on_samp.size:
            keep_idx = np.sort(keep_idx)
            on_samp = on_samp[keep_idx]
            onsets_sec = onsets_sec[keep_idx]

        if on_samp.size == 0:
            return None

        # Extra exclusion: drop events whose epoch window overlaps exclude_labels (e.g., 'seizure')
        if exclude_labels:
            # Build union of exclude intervals (seconds)
            ex_list = []
            for lab in exclude_labels:
                ex_iv = _intervals_for_label(lab)
                if ex_iv.size:
                    ex_list.append(ex_iv)
            if ex_list:
                ex_all = _merge(np.vstack(ex_list))
                # Epoch windows [start, end) in seconds
                starts = onsets_sec + float(tmin)
                ends = onsets_sec + float(tmax)
                # Clip starts to 0 for comparison (optional)
                starts = np.maximum(starts, 0.0)
                keep = np.ones_like(starts, dtype=bool)
                # Simple O(E*M) overlap test
                for (a, b) in ex_all:
                    # overlap if start < b and end > a
                    keep &= ~((starts < b) & (ends > a))
                if not np.any(keep):
                    return None
                on_samp = on_samp[keep]
                onsets_sec = onsets_sec[keep]

        if on_samp.size == 0:
            return None

        events = np.c_[on_samp, np.zeros_like(on_samp), np.ones_like(on_samp)]
        event_id = {label: 1}

        # Build Epochs — BAD_* will be dropped automatically
        return mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax,
                          picks=picks, preload=preload, reject_by_annotation=reject_by_annotation,
                          baseline=None, event_repeated=event_repeated) or None

    def extract_opposite_epochs(self, df: pd.DataFrame, len_session: int) -> pd.DataFrame:
        assert isinstance(df, pd.DataFrame)
        assert df.columns.tolist() == ['onset', 'offset']
        assert len_session > 0

        reverse_intervals = []
        if len(df) == 0:
            reverse_intervals.append({'onset': 0, 'offset': len_session})
            return pd.DataFrame(reverse_intervals, columns=['onset', 'offset'])

        if df['onset'].iloc[0] > 0:
            reverse_intervals.append({'onset': 0, 'offset': int(df['onset'].iloc[0])})

        for i in range(len(df) - 1):
            reverse_intervals.append({
                'onset': int(df['offset'].iloc[i]),
                'offset': int(df['onset'].iloc[i + 1])
            })

        if df['offset'].iloc[-1] < len_session:
            reverse_intervals.append({'onset': int(df['offset'].iloc[-1]), 'offset': int(len_session)})

        return pd.DataFrame(reverse_intervals, columns=['onset', 'offset'])


    def extract_sleep_epochs(self):

        if not os.path.exists(self.session.session_results_paths[self.session.session_type]['timings'][
                                  'sleep']) or self.session.params.redo_sleep:

            print('Please specify the best channel number for sleep detection : {}'.format(
                self.session.animal.areas_animal_clean))

            seizures_channel_idx = input()
            signal_channel = self.session.cleaned_lfp_data_filtered.iloc[int(seizures_channel_idx)]
            spect_channel = self.session.spectrograms_dict[int(seizures_channel_idx)]

            if self.session.params.sleep_detection_mode == "theta_delta_ratio":
                NREM = sleep_detection_fear_paradigm_integration(signal_channel, spect_channel,
                                                                 self.session.cleaned_processed_motion,
                                                                 self.session.timings['freezing'],
                                                                 self.session.timings['bad_epochs'],
                                                                 self.session.params.sample_rate,
                                                                 self.session.animal.animal_info,
                                                                 self.session.session_dir)

            with h5py.File(self.session.session_results_paths[self.session.session_type]['timings']['sleep'],
                           'w') as h5file:
                h5file.create_dataset('timings_sleep', data=NREM.to_records(index=False))

        else:
            with h5py.File(self.session.session_results_paths[self.session.session_type]['timings']['sleep'],
                           'r') as h5file:
                NREM = pd.DataFrame.from_records(h5file['timings_sleep'][:])

        return NREM

    def process_sleep(self):
        self.session.timings['sleep'] = self.extract_sleep_epochs()
        self.session.timings['woke'] = extract_opposite_epochs(self.session.timings['sleep'],
                                                               self.session.last_sample)
        if len(self.session.timings['sleep']) > 0:
            onsets = self.session.timings['sleep']['onset'].to_numpy() / self.sfreq
            durations = (self.session.timings['sleep']['offset'] - self.session.timings['sleep'][
                'onset']).to_numpy() / self.sfreq
            self.session.recording.annotations.delete(
                np.where(self.session.recording.annotations.description == 'sleep'))
            ann = mne.Annotations(onset=onsets, duration=durations,
                                  description=['sleep'] * len(self.session.timings['sleep']))
            self.session.recording.set_annotations(
                self.session.recording.annotations + ann if len(self.session.recording.annotations) else ann)

            onsets = self.session.timings['woke']['onset'].to_numpy() / self.sfreq
            durations = (self.session.timings['woke']['offset'] - self.session.timings['woke'][
                'onset']).to_numpy() / self.sfreq
            self.session.recording.annotations.delete(
                np.where(self.session.recording.annotations.description == 'woke'))
            ann = mne.Annotations(onset=onsets, duration=durations,
                                  description=['woke'] * len(self.session.timings['woke']))
            self.session.recording.set_annotations(
                self.session.recording.annotations + ann if len(self.session.recording.annotations) else ann)
        pass

    def get_or_choose_seizure_channel(self, prefer_types=('seeg', 'dbs', 'ecog')) -> tuple[int, str]:
        """
        Returns (pick_index, ch_name) for the chosen seizure channel.

        - Tries to load a previously stored selection from annotations: SEIZURE_CH=CHxxx
        - Otherwise lists good LFP channels (exclude bads), asks for integer index,
          and stores the choice as an annotation for next time.
        """
        # 1) Check if we already stored the choice as an annotation
        if len(self.session.recording.annotations):
            for desc in self.session.recording.annotations.description:
                if isinstance(desc, str) and desc.startswith('SEIZURE_CH='):
                    ch_name = desc.split('=', 1)[1]
                    if ch_name in self.session.recording.ch_names:
                        pick = mne.pick_channels(self.session.recording.ch_names, include=[ch_name])[0]
                        return pick, ch_name

        # 2) Build good LFP picks (exclude bads). Adjust types to your data.
        picks = mne.pick_types(
            self.session.recording.info,
            seeg=('seeg' in prefer_types),
            dbs=('dbs' in prefer_types),
            ecog=('ecog' in prefer_types),
            eeg=True, meg=False, stim=False, misc=False,
            exclude='bads'
        )
        if len(picks) == 0:
            raise RuntimeError("No LFP channels picked. Check channel types or bads.")

        good_names = [self.session.recording.ch_names[p] for p in picks]

        # 3) Present the enumerated list to the user (0..N-1)
        print("Choose seizure channel index from the good channels below:")
        for i, name in enumerate(good_names):
            area = self.session.animal.areas_animal_clean.get(i, '')  # your dict maps 0..N-1 -> area label
            print(f"[{i:02d}] {name}  {area}")

        # 4) Get user input (or pass it in programmatically)
        user_idx = int(input("Seizure channel index (0-based): ").strip())
        if not (0 <= user_idx < len(picks)):
            raise ValueError(f"Index {user_idx} out of range (0..{len(picks) - 1}).")

        pick = picks[user_idx]
        ch_name = self.session.recording.ch_names[pick]

        # 5) Store selection as an annotation so you won’t be asked again
        ann = mne.Annotations(onset=[0.0], duration=[0.0], description=['SEIZURE_CH=' + ch_name])
        self.session.recording.set_annotations(
            self.session.recording.annotations + ann if len(self.session.recording.annotations) else ann)

        return pick, ch_name

    def bad_periods_from_annotations(self,
                                     include_prefixes=('BAD_',),
                                     include_labels=None) -> pd.DataFrame:
        """
        Extract 'bad' time segments from Raw.annotations as a DataFrame of sample indices.

        - include_prefixes: tuple of prefixes to include (e.g., ('BAD',)).
        - include_labels: explicit list of labels to include (e.g., ['freezing']).
          If provided, these are included in addition to any prefixes.
        Returns: DataFrame with columns ['onset', 'offset'] in samples (offset is exclusive).
        """
        anns = self.session.recording.annotations
        if anns is None or len(anns) == 0:
            return pd.DataFrame(columns=['onset', 'offset'])

        sfreq = float(self.session.recording.info['sfreq'])
        n_times = int(self.session.recording.n_times)

        descs = np.array(anns.description, dtype=object)
        dur = np.array(anns.duration, dtype=float)
        onset_sec = np.array(anns.onset, dtype=float)

        # Select labels
        mask = np.zeros(len(descs), dtype=bool)
        if include_prefixes:
            for p in include_prefixes:
                mask |= np.array([d.startswith(p) if isinstance(d, str) else False for d in descs], dtype=bool)
        if include_labels:
            wanted = set(include_labels)
            mask |= np.array([d in wanted for d in descs], dtype=bool)

        # Exclude zero-duration markers
        mask &= (dur > 0)

        if not mask.any():
            return pd.DataFrame(columns=['onset', 'offset'])

        # Convert to samples (onset: floor, length: ceil; offset exclusive)
        onset_samp = np.floor(onset_sec[mask] * sfreq).astype(int)
        length_samp = np.ceil(dur[mask] * sfreq).astype(int)
        length_samp[length_samp <= 0] = 1  # ensure at least 1 sample
        offset_samp = onset_samp + length_samp

        # Clip to recording bounds
        onset_samp = np.clip(onset_samp, 0, n_times)
        offset_samp = np.clip(offset_samp, 0, n_times)

        df = pd.DataFrame({'onset': onset_samp, 'offset': offset_samp}).sort_values('onset').reset_index(drop=True)

        # Merge overlapping/adjacent intervals
        if len(df) == 0:
            return df

        merged = []
        cur_on, cur_off = int(df.loc[0, 'onset']), int(df.loc[0, 'offset'])
        for i in range(1, len(df)):
            on, off = int(df.loc[i, 'onset']), int(df.loc[i, 'offset'])
            if on <= cur_off:  # overlap or adjacency
                cur_off = max(cur_off, off)
            else:
                merged.append({'onset': cur_on, 'offset': cur_off})
                cur_on, cur_off = on, off
        merged.append({'onset': cur_on, 'offset': cur_off})

        return pd.DataFrame(merged, columns=['onset', 'offset'])

    def extract_nonseizures_epochs(self):

        if not os.path.exists(self.session.session_results_paths[self.session.session_type]['timings'][
                                  'seizure']) or self.session.params.redo_seizures:

            print('Please specify the best channel number for seizure detection : {}'.format(
                self.session.animal.areas_animal_clean))

            pick, ch_name = self.get_or_choose_seizure_channel(
                prefer_types=('seeg', 'eeg', 'ecog'))  # adjust to your LFP types

            # Get the signal for detection
            signal_channel = self.session.recording.get_data(picks=[pick]).ravel()
            # Build bad_periods from annotations in self.session.raw
            bad_periods = self.bad_periods_from_annotations(include_prefixes=('bad', 'BAD'))

            if self.session.params.seizures_detection_mode == "dual_band_peaks":
                nonseizures = dual_band_peaks_analysis_fear_cond_paradigm_integration(signal_channel, {'Sxx': None},
                                                                                   bad_periods,
                                                                                   self.session.params.sample_rate,
                                                                                   self.session.animal.animal_info,
                                                                                   self.session.session_dir)

            with h5py.File(self.session.session_results_paths[self.session.session_type]['timings']['seizure'],
                           'w') as h5file:
                h5file.create_dataset('timings_seizure', data=nonseizures.to_records(index=False))

        else:
            with h5py.File(self.session.session_results_paths[self.session.session_type]['timings']['seizure'],
                           'r') as h5file:
                nonseizures = pd.DataFrame.from_records(h5file['timings_seizure'][:])

        return nonseizures

    @staticmethod
    def _merge_intervals_df(df: pd.DataFrame) -> pd.DataFrame:
        """Merge overlapping or adjacent [onset, offset) intervals in samples."""
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=['onset', 'offset'])
        df = df.dropna()[['onset', 'offset']].astype(int).sort_values('onset').reset_index(drop=True)
        merged = []
        cur_on, cur_off = int(df.loc[0, 'onset']), int(df.loc[0, 'offset'])
        for i in range(1, len(df)):
            on, off = int(df.loc[i, 'onset']), int(df.loc[i, 'offset'])
            if on <= cur_off:  # overlap or touch
                cur_off = max(cur_off, off)
            else:
                merged.append({'onset': cur_on, 'offset': cur_off})
                cur_on, cur_off = on, off
        merged.append({'onset': cur_on, 'offset': cur_off})
        return pd.DataFrame(merged, columns=['onset', 'offset'])

    def remove_annotations_by_label(self, labels: list[str]) -> None:
        """Remove all annotations whose description matches any in labels."""
        if self.session.recording.annotations is None or len(self.session.recording.annotations) == 0:
            return
        labset = set(labels)
        keep_mask = np.array([d not in labset for d in self.session.recording.annotations.description], dtype=bool)
        self.session.recording.set_annotations(self.session.recording.annotations[keep_mask])

    def add_intervals_as_annotations(self,
                                     df: pd.DataFrame,
                                     label: str,
                                     merge=True,
                                     clip=True,
                                     replace_existing=True) -> None:
        """
        Add intervals (samples) as annotations with given label.
        - df: DataFrame with columns ['onset', 'offset'] in samples (offset exclusive)
        - replace_existing: remove existing annotations with same label first
        """
        if df is None or len(df) == 0:
            return
        sfreq = float(self.session.recording.info['sfreq'])
        n_times = int(self.session.recording.n_times)

        df_use = df[['onset', 'offset']].dropna().astype(int)
        if merge:
            df_use = self._merge_intervals_df(df_use)
        if clip:
            df_use['onset'] = df_use['onset'].clip(lower=0, upper=n_times - 1)
            df_use['offset'] = df_use['offset'].clip(lower=0, upper=n_times)
            df_use = df_use[df_use['offset'] > df_use['onset']]

        on_sec = (df_use['onset'].to_numpy() / sfreq).astype(float)
        dur_sec = ((df_use['offset'] - df_use['onset']).to_numpy() / sfreq).astype(float)

        if replace_existing:
            self.remove_annotations_by_label([label])

        ann = mne.Annotations(onset=on_sec, duration=dur_sec, description=[label] * len(on_sec))
        self.session.recording.set_annotations(
            self.session.recording.annotations + ann if len(self.session.recording.annotations) else ann)

    def _seizures_already_processed(self) -> bool:
        anns = self.session.recording.annotations
        if anns is None or len(anns) == 0:
            return False
        descs = set(anns.description)
        return ('seizure' in descs) or ('PROC_SEIZURES_DONE' in descs)

    def _all_base_timings_already_processed(self) -> bool:
        anns = self.session.recording.annotations
        if anns is None or len(anns) == 0:
            return False
        descs = set(anns.description)
        return ('PROC_BASE_TIMINGS_DONE' in descs)


    def process_seizures(self):

        # Compute intervals in samples (you already do this)
        self.session.timings['nonseizure'] = self.extract_nonseizures_epochs()
        self.session.timings['seizure'] = extract_opposite_epochs(self.session.timings['nonseizure'],
                                                                      self.session.last_sample)
        if self._seizures_already_processed():

            if len(self.session.timings['nonseizure']) > 0:
                onsets = self.session.timings['nonseizure']['onset'].to_numpy() / self.sfreq
                durations = (self.session.timings['nonseizure']['offset'] - self.session.timings['nonseizure'][
                    'onset']).to_numpy() / self.sfreq
                self.session.recording.annotations.delete(
                    np.where(self.session.recording.annotations.description == 'nonseizures'))
                ann = mne.Annotations(onset=onsets, duration=durations,
                                      description=['nonseizure'] * len(self.session.timings['nonseizure']))
                self.session.recording.set_annotations(
                    self.session.recording.annotations + ann if len(self.session.recording.annotations) else ann)

            if len(self.session.timings['seizure']) > 0:
                onsets = self.session.timings['seizure']['onset'].to_numpy() / self.sfreq
                durations = (self.session.timings['seizure']['offset'] - self.session.timings['seizure'][
                    'onset']).to_numpy() / self.sfreq
                ann = mne.Annotations(onset=onsets, duration=durations,
                                      description=['seizure'] * len(self.session.timings['seizure']))
                self.session.recording.set_annotations(
                    self.session.recording.annotations + ann if len(self.session.recording.annotations) else ann)
            # Optional: add a processed marker
            marker = mne.Annotations(onset=[0.0], duration=[0.0], description=['PROC_SEIZURES_DONE'])
            self.session.recording.set_annotations(
                self.session.recording.annotations + marker if len(self.session.recording.annotations) else marker)
        else:
            return

    def save_timings_to_excel(self, path):
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            for timing_type, df in self.session.timings.items():
                if isinstance(df, pd.DataFrame) and {'onset', 'offset'}.issubset(df.columns):
                    df.to_excel(writer, sheet_name=timing_type[:31], index=False)  # Excel sheet name limit

    def cleanup_annotations(self,
                            unwanted_labels=None,  # exact labels to drop
                            drop_single_char=False,  # drop 1-char labels (e.g., 'a','b','_')
                            drop_zero_duration_for=None,  # drop zero-duration occurrences for given labels
                            drop_regex=None  # optional regex (or list) to drop
                            ) -> None:
        """
        Remove unwanted annotations from self.session.raw in-place.
        """
        raw = self.session.recording
        ann = raw.annotations
        if ann is None or len(ann) == 0:
            return

        desc = np.asarray(ann.description, dtype=object)
        dur = np.asarray(ann.duration, dtype=float)
        keep = np.ones(len(desc), dtype=bool)

        # Exact labels to drop (your case)
        if unwanted_labels:
            keep &= ~np.isin(desc, list(unwanted_labels))

        # Optional: drop any single-character labels
        if drop_single_char:
            keep &= np.array([not (isinstance(d, str) and len(d) == 1) for d in desc])

        # Optional: drop zero-duration presets for specific labels
        if drop_zero_duration_for:
            keep &= ~(np.isin(desc, list(drop_zero_duration_for)) & (dur <= 0))

        # Optional: drop by regex pattern(s)
        if drop_regex:
            patterns = drop_regex if isinstance(drop_regex, (list, tuple)) else [drop_regex]
            bad_re = np.zeros(len(desc), dtype=bool)
            for pat in patterns:
                bad_re |= np.array([bool(re.fullmatch(pat, d)) if isinstance(d, str) else False for d in desc])
            keep &= ~bad_re

        raw.set_annotations(ann[keep])
