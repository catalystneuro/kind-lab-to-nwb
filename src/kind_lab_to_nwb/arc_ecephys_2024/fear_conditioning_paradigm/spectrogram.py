from pathlib import Path

import mne
import numpy as np
from mne.time_frequency import read_tfrs


class SpectrogramExtractor:
    def __init__(self, session, time_frequency_analysis_timings):
        """
        time_frequency_analysis_timings: iterable of tuples
          (label, event_type, t_before, t_after)
          - label: e.g. 'CS', 'NONCS', 'FREEZING', or composite 'FREEZING_within_CS'
          - event_type: 'onset' or 'offset'
          - t_before, t_after: in samples (integers)
        """
        self.session = session
        self.raw = session.recording
        self.params = session.params
        self.sfreq = float(self.raw.info['sfreq'])
        self.time_frequency_analysis_timings = time_frequency_analysis_timings

    def _lfp_picks(self, info):
        # Adjust seeg/ecog/dbs flags to your channel typing
        return mne.pick_types(info, seeg=True, ecog=True, dbs=True,
                              eeg=True, meg=False, stim=False, misc=False,
                              exclude='bads')

    # ---------- intervals and intersections ----------
    @staticmethod
    def _merge_intervals(iv: np.ndarray) -> np.ndarray:
        """iv: Nx2 [on, off] in seconds; returns merged & sorted."""
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

    @staticmethod
    def _intersect(iv1: np.ndarray, iv2: np.ndarray) -> np.ndarray:
        """Intersection of two merged Nx2 arrays (seconds)."""
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
        return SpectrogramExtractor._merge_intervals(np.asarray(out, float)) if out else np.empty((0, 2), float)

    def _intervals_for_label(self, label: str) -> np.ndarray:
        """Return merged [on, off] intervals in seconds for a simple annotation label (case-insensitive)."""
        ann = self.raw.annotations
        if ann is None or len(ann) == 0:
            return np.empty((0, 2), float)
        desc = np.asarray(ann.description, dtype=object).astype(str)
        on = np.asarray(ann.onset, float)
        du = np.asarray(ann.duration, float)
        mask = (np.char.lower(desc) == label.lower()) & (du > 0)
        if not mask.any():
            return np.empty((0, 2), float)
        iv = np.c_[on[mask], on[mask] + du[mask]]
        return self._merge_intervals(iv)

    def _intervals_for_label_or_within(self, label: str) -> np.ndarray:
        """Supports 'A', or 'A_within_B' (case-insensitive)."""
        if '_within_' in label.lower():
            a_lab, b_lab = label.split('_within_', 1)
            iv_a = self._intervals_for_label(a_lab)
            iv_b = self._intervals_for_label(b_lab)
            return self._intersect(iv_a, iv_b)
        else:
            return self._intervals_for_label(label)

    def _events_from_label(self, label: str, event_type: str) -> np.ndarray:
        """
        Build events at onsets or offsets of label intervals (base or intersection).
        Returns events array (n_events, 3) with value code 1.
        """
        iv = self._intervals_for_label_or_within(label)
        if iv.size == 0:
            return np.empty((0, 3), int)

        sf = self.sfreq
        if event_type.lower() == 'onset':
            ev_samp = np.floor(iv[:, 0] * sf).astype(int)
        elif event_type.lower() == 'offset':
            ev_samp = np.ceil(iv[:, 1] * sf).astype(int)
        else:
            raise ValueError("event_type must be 'onset' or 'offset'")

        # Deduplicate equal sample points
        if ev_samp.size == 0:
            return np.empty((0, 3), int)
        _, keep = np.unique(ev_samp, return_index=True)
        ev_samp = ev_samp[np.sort(keep)]

        events = np.c_[ev_samp, np.zeros_like(ev_samp), np.ones_like(ev_samp)]
        return events

    # ---------- main API ----------
    def process(self):
        for (label, event_type, t_before_samp, t_after_samp) in self.time_frequency_analysis_timings:
            print('Processing spectrogram: --------- ' + label)

            self._compute_and_save_spectrogram_for_label(
                label=label,
                event_type=event_type,
                t_before_samp=int(t_before_samp),
                t_after_samp=int(t_after_samp),
            )

    def _compute_and_save_spectrogram_for_label(self, label: str, event_type: str,
                                                t_before_samp: int, t_after_samp: int):
        # Paths
        out_dir = self.session.session_results_paths[self.session.session_type]['time_frequency_analysis'][
            'spectrogram']
        formatted = f"{label}_{event_type}_{t_before_samp}_{t_after_samp}"
        out_path = Path(out_dir[formatted])

        # Picks
        picks = self._lfp_picks(self.raw.info)

        # Build events from annotations (base or composite) first
        events = self._events_from_label(label, event_type)
        if events.size == 0:
            # nothing to do
            return

        raw_copy = self.raw.copy()
        raw_copy.notch_filter(self.params.notch_frequency)

        # Epoch around event onsets/offsets with your sample-based window
        tmin = -float(t_before_samp) / self.sfreq
        tmax = float(t_after_samp) / self.sfreq

        epochs = mne.Epochs(raw_copy, events, event_id={label: 1}, tmin=tmin, tmax=tmax,
                            picks=picks, preload=True, reject_by_annotation=True, baseline=None)
        if len(epochs) == 0:
            return

        # Frequencies and tapers
        freqs = np.arange(self.params.spect_min_freq,
                          self.params.spect_max_freq,
                          self.params.spect_freq_resolution,
                          dtype=float)

        tfr = epochs.compute_tfr(freqs=freqs, n_cycles=self.params.spect_n_cycles, verbose=True, output='power',
                                 average=True, method='multitaper', time_bandwidth=self.params.spects_time_bandwidth,
                                 n_jobs=-1)

        # Save natively (MNE TFR HDF5). Use .h5 extension for compatibility.
        tfr.save(str(out_path), overwrite=True)

    # Optional: load back later
    @staticmethod
    def load_spectrogram(path: str):
        # Returns a list; AverageTFR or EpochsTFR at index 0
        return read_tfrs(path)[0]
