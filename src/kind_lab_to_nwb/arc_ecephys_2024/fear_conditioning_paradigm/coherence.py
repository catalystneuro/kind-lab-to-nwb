import gzip
import os
import pickle

import mne
import numpy as np
from mne_connectivity import spectral_connectivity_epochs

from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.timings import TimingsProcessor


class CoherenceAnalyzer:
    """
     Spectral connectivity for conditions using mne-connectivity.

     methods: iterable of measures to compute (e.g. ['coh', 'imcoh', 'dpli', 'wpli', 'gc']).

     Expected results_paths structure:
       results_paths['epoch_analysis']['conn'][method][cond] -> filepath (we will save .nc there)
     """

    def __init__(self, session, epochs_frequency_analysis_timings, methods=('coh', 'imcoh', 'dpli', 'wpli', 'gc')):
        self.session = session
        self.params = session.params
        self.results_paths = session.session_results_paths[self.session.session_type]
        self.raw = session.recording
        self.sfreq = float(self.raw.info['sfreq'])
        self.epochs_frequency_analysis_timings = [str(c).lower() for c in epochs_frequency_analysis_timings]
        self.methods = [m.lower() for m in methods]
        self.process()

    # ---------- Picks ----------
    def _lfp_picks(self, info):
        # Adjust seeg/ecog/dbs based on your channel typing
        return mne.pick_types(info, seeg=True, dbs=True, ecog=True,
                              eeg=True, meg=False, stim=False, misc=False,
                              exclude='bads')

    # ---------- Main loop ----------
    def process(self):
        for cond in self.epochs_frequency_analysis_timings:
            print('Processing coherence: --------- ' + cond)

            for method in self.methods:
                out_path = self.results_paths['epoch_analysis']['coherence'][cond]
                self._compute_and_save_condition_connectivity(
                    cond, method, out_path, redo=self.params.redo_cohe)


    def _get_output_path(self, cond: str, method: str) -> str:
        try:
            return self.results_paths['epoch_analysis']['conn'][method][cond]
        except Exception as e:
            raise KeyError("Add results_paths['epoch_analysis']['conn'][method][cond] to your session paths.") from e

    @staticmethod
    def _save_connectivity_pkl(conn, save_path: str, compress_level: int = 6):
        """
        Save the Connectivity object as a pickle (.pkl or .pkl.gz if compress=True).

        Returns the final path used.
        """

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with gzip.open(save_path, 'wb', compresslevel=int(compress_level)) as f:
            pickle.dump(conn, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load_connectivity_pkl(path: str):
        import gzip
        import pickle
        opener = gzip.open if path.endswith('.gz') else open
        with opener(path, 'rb') as f:
            conn = pickle.load(f)
        return conn


    def _compute_and_save_condition_connectivity(self, cond: str, method: str, results_path: str, redo: bool = False):
        out_pkl = results_path[:-3] + f'_{method}.pkl.gz'
        if (not redo) and os.path.exists(out_pkl):
            return

        label = cond  # matches your annotation label names
        recording_raw = self.raw.copy()

        # Picks for neural channels
        picks = self._lfp_picks(recording_raw.info)

        win = float(getattr(self.params, 'psds_window_sec', 5.0))
        info_label = TimingsProcessor._summarize_annotation_excluding_within(recording_raw, label)
        if not info_label['exists'] or info_label['dur_stats_sec'].get('max', 0.0) < win:
            return

        # Notch filter Raw only now that we know we'll have epochs
        if getattr(self.params, 'psds_apply_notch', True):
            recording_raw.notch_filter(getattr(self.params, 'notch_frequency', 50.0))

        # Build epochs from annotations
        epochs = TimingsProcessor._epochs_from_label(recording_raw, label, window_sec=win, picks=picks)
        if (epochs is None) or (len(epochs) == 0):
            print(f"[ConnectivityAnalyzer] No epochs for condition '{cond}', skipping.")
            return

        # Bounds and options
        fmin = float(self.params.cohe_min_freq)
        fmax = float(self.params.cohe_max_freq)
        mode = 'multitaper'
        n_jobs = getattr(self.params, 'n_jobs', -1)

        # Indices: only for 'gc' (directed), as multivariate lists-of-arrays
        indices = None
        if method.lower() == 'gc':
            n_nodes = len(epochs.ch_names)
            if n_nodes < 2:
                print("[ConnectivityAnalyzer] Not enough channels for GC.")
                return

            # All ordered pairs i -> j (i != j) as singleton “nodes”
            ii, jj = np.where(~np.eye(n_nodes, dtype=bool))
            seeds = [np.array([i], dtype=int) for i in ii]
            targets = [np.array([j], dtype=int) for j in jj]
            indices = (seeds, targets)

        # Compute spectral connectivity
        conn = spectral_connectivity_epochs(
            epochs,
            method=method,
            mode=mode,
            fmin=fmin,
            fmax=fmax,
            indices=indices,  # None for undirected measures, tuple for 'gc'
            n_jobs=n_jobs,
            verbose=True
        )

        save_path = out_pkl
        self._save_connectivity_pkl(conn, save_path)