import os

import mne
import numpy as np

from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.timings import TimingsProcessor


class PSDsAnalyzer:
    def __init__(self, session, epochs_frequency_analysis_timings):
        """
        epochs_frequency_analysis_timings: list of condition keys you want to analyze,
          e.g. ['cs', 'noncs', 'freezing', 'nonfreezing']
        """
        self.session = session
        self.params = session.params
        self.results_paths = session.session_results_paths[self.session.session_type]
        self.raw = session.recording
        self.sfreq = float(self.raw.info['sfreq'])
        self.epochs_frequency_analysis_timings = [str(c).lower() for c in epochs_frequency_analysis_timings]
        self.process()

    # ---------- Core public loop ----------
    def process(self):
        for cond in self.epochs_frequency_analysis_timings:
            out_path = self.results_paths['epoch_analysis']['psd'][cond]
            self._compute_and_save_condition_psd(cond, out_path, redo=self.params.redo_pwr)

    # ---------- Picks helpers ----------
    def _lfp_picks(self, info):
        # Adjust seeg/ecog/dbs based on your channel typing
        return mne.pick_types(info, seeg=True, dbs=True, ecog=True,
                              eeg=True, meg=False, stim=False, misc=False,
                              exclude='bads')

    # ---------- Notch filtering on Epochs (true notch) ----------
    def notch_epochs_iir(self, epochs: mne.Epochs, f0: float, Q: float, order: int = 4):
        sf = epochs.info['sfreq']
        assert f0 < sf / 2, "notch_frequency must be < Nyquist"
        freqs = np.arange(f0, sf / 2, f0)
        if freqs.size == 0:
            return
        widths = np.maximum(freqs / Q, 0.5)  # Hz
        iir_params = dict(order=int(order), ftype='butter', output='sos')
        lfp_picks = self._lfp_picks(epochs.info)
        epochs.notch_filter(freqs=freqs, method='iir', iir_params=iir_params,
                            notch_widths=widths, picks=lfp_picks, phase='zero')


    def annotate_complement(self, base_label: str, comp_label: str):
        """Create comp_label = complement of base_label across the recording (if not already present)."""
        if any(d == comp_label for d in self.raw.annotations.description):
            return
        sf, n = self.raw.info['sfreq'], self.raw.n_times
        desc = np.asarray(self.raw.annotations.description, dtype=object)
        onset = np.asarray(self.raw.annotations.onset)
        dur = np.asarray(self.raw.annotations.duration)
        mask = (desc == base_label) & (dur > 0)
        if not mask.any():
            self.raw.set_annotations(
                self.raw.annotations + mne.Annotations([0.0], [n / sf], [comp_label]) if len(self.raw.annotations) else
                mne.Annotations([0.0], [n / sf], [comp_label]))
            return
        segs = np.c_[np.floor(onset[mask] * sf).astype(int),
        (np.floor(onset[mask] * sf) + np.ceil(dur[mask] * sf).astype(int))]
        segs = segs[np.argsort(segs[:, 0])]
        # merge
        merged = []
        s0, e0 = segs[0]
        for s, e in segs[1:]:
            if s <= e0:
                e0 = max(e0, e)
            else:
                merged.append((s, e0));
                s0, e0 = s, e
        merged.append((s0, e0))
        # complement
        outs, oute, cur = [], [], 0
        for s, e in merged:
            if cur < s: outs.append(cur); oute.append(s)
            cur = max(cur, e)
        if cur < n: outs.append(cur); oute.append(n)
        if outs:
            on = np.array(outs) / sf
            du = (np.array(oute) - np.array(outs)) / sf
            comp = mne.Annotations(on, du, [comp_label] * len(on))
            self.raw.set_annotations(self.raw.annotations + comp if len(self.raw.annotations) else comp)

    def epochs_from_label(self, label: str, window_sec: float | None,
                          picks=None, preload=True, reject_by_annotation=True) -> mne.Epochs | None:
        """Build Epochs directly from a single annotation label (already uppercase)."""
        event_id = {label: 1}
        if window_sec is None:
            # Use full annotation duration; must be (approximately) constant
            durs = np.asarray(
                [d for d, l in zip(self.raw.annotations.duration, self.raw.annotations.description) if
                 l == label and d > 0])
            if durs.size == 0:
                return None
            sf = self.raw.info['sfreq']
            d0 = durs[0]
            if not np.allclose(durs, d0, atol=1.0 / sf):
                raise ValueError(f"Label '{label}' has variable durations; pass window_sec to tile.")
            tmin, tmax = 0.0, float(d0)
            events, ev_id = mne.events_from_annotations(self.raw, event_id=event_id, chunk_duration=None)
        else:
            tmin, tmax = 0.0, float(window_sec)
            events, ev_id = mne.events_from_annotations(self.raw, event_id=event_id, chunk_duration=float(window_sec))

        if events.size == 0:
            return None
        epochs = mne.Epochs(self.raw, events=events, event_id=ev_id, tmin=tmin, tmax=tmax,
                            picks=picks, preload=preload, reject_by_annotation=reject_by_annotation, baseline=None)
        return epochs if len(epochs) > 0 else None

    def _compute_and_save_condition_psd(self, cond: str, results_path: str, redo: bool = False):
        if (not redo) and os.path.exists(results_path):
            return
        print('Processing PSD : --------- ' + cond)
        label = cond  # already uppercase like your annotations
        recording_raw = self.raw.copy()
        # Picks
        picks = self._lfp_picks(recording_raw.info)

        win = float(getattr(self.params, 'psds_window_sec', 5.0))
        info_label = TimingsProcessor._summarize_annotation_excluding_within(recording_raw, label)
        if info_label['exists']:
            if info_label['dur_stats_sec']['max'] >= win:
                if getattr(self.params, 'psds_apply_notch', True):
                    recording_raw.notch_filter(self.params.notch_frequency)  # .plot_psd()
                print('found epochs !')

                epochs = TimingsProcessor._epochs_from_label(recording_raw, label, window_sec=win, picks=picks)

                # TimingsProcessor._show_epochs_on_raw(recording_raw, epochs)
                if epochs is not None:
                    pwr_spectrum = epochs.compute_psd(
                        method='multitaper',
                        fmin=self.params.psds_min_freq, fmax=self.params.psds_max_freq, n_jobs=-1, verbose=True,
                    )

                    pwr_spectrum.save(results_path, overwrite=True)
        else:
            print('no epochs !')
            return
