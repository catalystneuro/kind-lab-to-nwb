import gzip
import pickle
from pathlib import Path

import mne
import numpy as np
from mne_connectivity import spectral_connectivity_epochs


class CoherogramExtractor:
    """
    Time-frequency spectral connectivity (“coherograms”) per condition.

    time_frequency_analysis_timings: iterable of tuples (label, event_type, t_before, t_after)
      - label: e.g., 'CS', 'NONCS', 'FREEZING', or composite 'FREEZING_within_CS'
      - event_type: 'onset' or 'offset' (epoch anchor)
      - t_before, t_after: window in samples (integers)
    """

    def __init__(self, session, time_frequency_analysis_timings):
        self.session = session
        self.raw = session.recording
        self.params = session.params
        self.sfreq = float(self.raw.info['sfreq'])
        self.time_frequency_analysis_timings = time_frequency_analysis_timings

    # ---------- picks ----------
    def _lfp_picks(self, info):
        # Adjust typing to your data (seeg/ecog/dbs); exclude bads
        return mne.pick_types(info, seeg=True, ecog=True, dbs=True,
                              eeg=True, meg=False, stim=False, misc=False,
                              exclude='bads')

    # ---------- interval helpers ----------
    @staticmethod
    def _merge_intervals(iv: np.ndarray) -> np.ndarray:
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
        return CoherogramExtractor._merge_intervals(np.asarray(out, float)) if out else np.empty((0, 2), float)

    def _intervals_for_label(self, label: str) -> np.ndarray:
        """Merged [on, off] in seconds for a simple annotation label (case-insensitive)."""
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
        """Support 'A' or composite 'A_within_B' (case-insensitive)."""
        if '_within_' in label.lower():
            a_lab, b_lab = label.split('_within_', 1)
            return self._intersect(self._intervals_for_label(a_lab),
                                   self._intervals_for_label(b_lab))
        else:
            return self._intervals_for_label(label)

    def _events_from_label(self, label: str, event_type: str,
                           t_before_samp: int, t_after_samp: int):
        """
        Build events from annotation intervals (base or intersection).
        Keep only intervals that can accommodate t_before + t_after.
        Returns (events, tmin, tmax) or (None, None, None).
        """
        iv = self._intervals_for_label_or_within(label)
        if iv.size == 0:
            return None, None, None

        min_len_sec = (t_before_samp + t_after_samp) / self.sfreq
        iv = iv[(iv[:, 1] - iv[:, 0]) >= min_len_sec]
        if iv.size == 0:
            return None, None, None

        if event_type.lower() == 'onset':
            ev_sec = iv[:, 0]
        elif event_type.lower() == 'offset':
            ev_sec = iv[:, 1]
        else:
            raise ValueError("event_type must be 'onset' or 'offset'")

        ev_samp = np.floor(ev_sec * self.sfreq).astype(int)
        if ev_samp.size == 0:
            return None, None, None
        _, keep = np.unique(ev_samp, return_index=True)
        ev_samp = ev_samp[np.sort(keep)]
        if ev_samp.size == 0:
            return None, None, None

        events = np.c_[ev_samp, np.zeros_like(ev_samp), np.ones_like(ev_samp)]
        tmin = -float(t_before_samp) / self.sfreq
        tmax = float(t_after_samp) / self.sfreq
        return events, tmin, tmax

    # ---------- main ----------
    def process(self):
        for (label, event_type, t_before, t_after) in self.time_frequency_analysis_timings:
            print('Processing coherogram: --------- ' + label)

            self._compute_and_save_coherogram_for_label(
                label=label, event_type=event_type,
                t_before_samp=int(t_before), t_after_samp=int(t_after)
            )

    def _compute_and_save_coherogram_for_label(self, label: str, event_type: str,
                                               t_before_samp: int, t_after_samp: int):
        # Resolve output base path
        out_map = self.session.session_results_paths[self.session.session_type]['time_frequency_analysis']['coherogram']
        formatted = f"{label}_{event_type}_{t_before_samp}_{t_after_samp}"
        out_base = Path(out_map[formatted]).with_suffix('')  # we will add per-method suffix

        # Picks
        picks = self._lfp_picks(self.raw.info)

        # Build events first to decide if we should process
        events, tmin, tmax = self._events_from_label(label, event_type, t_before_samp, t_after_samp)
        if events is None or events.size == 0:
            return

        # Notch-filter raw copy only if we will build epochs
        raw_copy = self.raw.copy()
        raw_copy.notch_filter(self.params.notch_frequency)
        # Build epochs anchored on events; BAD_* excluded automatically
        epochs = mne.Epochs(raw_copy, events=events, event_id={label: 1},
                            tmin=tmin, tmax=tmax, picks=picks,
                            preload=True, reject_by_annotation=True, baseline=None)
        if len(epochs) == 0:
            return

        # Optional: resample epochs for speed
        decim = int(getattr(self.params, 'coheros_decimation', 1))
        if decim > 1:
            epochs.resample(self.sfreq / decim, npad='auto')

        # Frequencies for coherogram (wavelet-based)
        freqs = np.arange(self.params.coheros_min_freq,
                          self.params.coheros_max_freq,
                          self.params.coheros_freq_resolution,
                          dtype=float)
        n_cycles = float(getattr(self.params, 'coheros_n_cycles', 7.0))

        # Methods you want to compute (univariate undirected)
        methods = getattr(self.params, 'coheros_methods', ['coh', 'imcoh', 'dpli'])
        methods = [m.lower() for m in methods]

        # Compute time-frequency connectivity (averaged across epochs)
        conn = spectral_connectivity_epochs(
            epochs,
            method=methods,  # e.g., ['coh','imcoh','dpli']
            mode='cwt_morlet',
            cwt_freqs=freqs,
            cwt_n_cycles=n_cycles,
            # By default, averages across epochs; if you need per-epoch, see keep_trials in your mne-connectivity version
            n_jobs=getattr(self.params, 'n_jobs', -1),
            verbose=True
        )
        for i, m in enumerate(methods):
            # Fallback to pickle if NetCDF not available
            out_pkl = str(out_base) + f'_{m}.pkl.gz'
            with gzip.open(out_pkl, 'wb', compresslevel=6) as f:
                pickle.dump(conn[i], f, protocol=pickle.HIGHEST_PROTOCOL)
