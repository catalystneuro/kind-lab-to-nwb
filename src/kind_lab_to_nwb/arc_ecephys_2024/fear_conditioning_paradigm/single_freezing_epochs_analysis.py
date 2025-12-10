import os
import pickle

import emd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mne.time_frequency import tfr_array_multitaper
from mne_connectivity import spectral_connectivity_epochs, seed_target_multivariate_indices
from scipy.signal import hilbert
from tqdm import tqdm

from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.signal_processing import extract_signal_within_timings, extract_signal_around_timing_df


def _freq_edges_from_centers(freqs):
    freqs = np.asarray(freqs, float)
    if freqs.size < 2:
        raise ValueError("Need at least 2 frequency bins.")
    df = float(np.median(np.diff(freqs)))
    edges = np.concatenate([[max(freqs[0] - df / 2, 0.0)], freqs + df / 2])
    return edges

def _emd_imfs_analytic(x, max_imfs=8):
    imfs = emd.sift.sift(x, max_imfs=max_imfs)  # (n_times, n_imfs) or (n_times,)
    if imfs.ndim == 1:
        imfs = imfs[:, None]
    imfs = imfs.T  # (n_imfs, n_times)
    z = hilbert(imfs, axis=-1)  # complex analytic IMFs
    return z  # (n_imfs, n_times)

def _inst_freq_from_phase(phase, sfreq):
    return np.diff(phase, axis=-1) * (sfreq / (2 * np.pi))

def _hht_channel_worker(x, sfreq, f_edges, n_bins, n_times, max_imfs=8, edge_trim=None):
    """Compute per-channel HHT spectrogram; returns (n_freqs, n_times) for one channel."""
    if edge_trim is None:
        edge_trim = int(round(0.1 * sfreq))

    z = _emd_imfs_analytic(x, max_imfs=max_imfs)  # (K, T)
    amp = np.abs(z)
    phase = np.unwrap(np.angle(z), axis=-1)
    ifreq = _inst_freq_from_phase(phase, sfreq)  # (K, T-1)
    amp2 = amp[:, :-1] ** 2  # align with IF

    # Trim ends to avoid IF edge artifacts
    t0 = edge_trim
    t1 = ifreq.shape[-1] - edge_trim
    if t1 <= t0:
        t0, t1 = 0, ifreq.shape[-1]
    ifreq_seg = ifreq[:, t0:t1]
    amp2_seg = amp2[:, t0:t1]

    H = np.zeros((n_bins, ifreq_seg.shape[-1]), float)
    # Bin by instantaneous frequency at each timepoint
    for k in range(ifreq_seg.shape[0]):
        idx = np.digitize(ifreq_seg[k], f_edges) - 1
        valid = (idx >= 0) & (idx < n_bins) & np.isfinite(amp2_seg[k]) & np.isfinite(ifreq_seg[k])
        if not np.any(valid):
            continue
        for b in range(n_bins):
            mask = valid & (idx == b)
            if np.any(mask):
                H[b, mask] += np.sum(amp2_seg[k, mask])

    # Insert into full-length time axis (pad NaNs at edges)
    H_full = np.full((n_bins, n_times), np.nan, float)
    H_full[:, 1 + t0:1 + t1] = H  # +1 aligns diff to samples 1..T-1
    return H_full


def hht_spectrogram_epoch_per_channel(
        X_ch_time,
        sfreq,
        freqs,
        max_imfs=8,
        edge_trim=None,
        n_jobs=1,
        prefer="processes",
        max_nbytes="10M",
):
    """
    Parallel HHT spectrogram per channel (no channel averaging).

    Inputs
    ------
    X_ch_time : array (n_channels, n_times) or DataFrame (channels as rows)
    sfreq : float
    freqs : 1D array of frequency centers
    max_imfs, edge_trim : EMD/HHT params
    n_jobs : int, number of workers (-1 for all)
    prefer : 'processes' (default) or 'threads'
    max_nbytes : joblib memmap threshold to reduce memory copies

    Returns
    -------
    H_all : array (n_channels, n_freqs, n_times)
    """
    X = X_ch_time.values if hasattr(X_ch_time, "values") else np.asarray(X_ch_time)
    if X.ndim != 2:
        raise ValueError("X_ch_time must be (n_channels, n_times).")
    n_ch, n_times = X.shape
    freqs = np.asarray(freqs, float)
    n_bins = len(freqs)
    f_edges = _freq_edges_from_centers(freqs)

    # Parallel per channel
    results = Parallel(n_jobs=n_jobs, backend="loky" if prefer == "processes" else "threading",
                       max_nbytes=max_nbytes)(
        delayed(_hht_channel_worker)(
            X[ch], sfreq, f_edges, n_bins, n_times, max_imfs, edge_trim
        )
        for ch in range(n_ch)
    )
    H_all = np.stack(results, axis=0)  # (n_ch, n_bins, n_times)
    return H_all

def _emd_decompose_channel_worker(x, sfreq, max_imfs):
    z = _emd_imfs_analytic(x, max_imfs=max_imfs)  # (K, T)
    ph = np.unwrap(np.angle(z), axis=-1)
    ifreq = _inst_freq_from_phase(ph, sfreq)  # (K, T-1)
    return z, ifreq

def _build_zbin_channel_worker(z, ifreq, f_edges, t0, t1, n_bins):
    """Return z_bin for one channel: (n_bins, T_valid)."""
    T_valid = t1 - t0
    zc_if = z[:, :-1]  # align to IF timeline
    zc_if = zc_if[:, t0:t1]  # apply trim
    ifc_seg = ifreq[:, t0:t1]  # (K, T_valid)

    z_bin_ch = np.zeros((n_bins, T_valid), dtype=np.complex128)
    for k in range(ifc_seg.shape[0]):
        idx = np.digitize(ifc_seg[k], f_edges) - 1
        valid = (idx >= 0) & (idx < n_bins) & np.isfinite(ifc_seg[k])
        if not np.any(valid):
            continue
        for b in range(n_bins):
            mask = valid & (idx == b)
            if np.any(mask):
                z_bin_ch[b, mask] += zc_if[k, mask]
    return z_bin_ch

def _pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _window_metrics_all(zi, zj, measures, debiased_wpli):
    """
    Compute all requested metrics for a given window.
    Returns dict with scalars for requested measures.
    """
    out = {}
    cross = zi * np.conj(zj)  # (T,)
    sxx = np.mean(np.abs(zi) ** 2)
    syy = np.mean(np.abs(zj) ** 2)
    denom = np.sqrt(sxx * syy) + 1e-15

    if 'coh' in measures or 'imcoh' in measures:
        coh_c = np.mean(cross) / denom
        if 'coh' in measures:
            out['coh'] = np.abs(coh_c)
        if 'imcoh' in measures:
            out['imcoh'] = np.imag(coh_c)

    if 'dpli' in measures or 'wpli' in measures:
        dphi = np.angle(zi) - np.angle(zj)
        s = np.sin(dphi)
        imag_cross = np.imag(cross)

        if 'dpli' in measures:
            out['dpli'] = np.mean(s > 0.0)  # [0,1]

        if 'wpli' in measures:
            if debiased_wpli:
                im = imag_cross
                num = (np.sum(im)) ** 2 - np.sum(im ** 2)
                den = (np.sum(np.abs(im)) ** 2) - np.sum(im ** 2) + 1e-15
                val = num / den if den > 0 else np.nan
                out['wpli'] = float(np.clip(val, 0.0, 1.0))
            else:
                num = np.abs(np.sum(imag_cross))
                den = np.sum(np.abs(imag_cross)) + 1e-15
                out['wpli'] = num / den
    return out

def _coh_bin_worker(
        bi, z_bin_b, sfreq, f_center, n_cycles, step_frac, measures, debiased_wpli, min_samples
):
    """
    Compute coherogram metrics for a single frequency bin.
    z_bin_b: array (n_channels, T_valid) complex for bin bi
    Returns times_b (n_win,), and dict of values per measure: (n_pairs, n_win)
    """
    n_ch, T_valid = z_bin_b.shape
    if f_center <= 0:
        raise ValueError("Frequencies must be > 0.")
    win_len = max(int(round((n_cycles * sfreq) / f_center)), int(min_samples))
    win_len = min(win_len, T_valid)
    step = max(int(round(step_frac * win_len)), 1)
    n_win = 1 + max(0, (T_valid - win_len) // step)

    pairs = _pairs(n_ch)
    n_pairs = len(pairs)

    times_b = np.empty((n_win,), float)
    values_b = {m: np.full((n_pairs, n_win), np.nan, float) for m in measures}

    for wi in range(n_win):
        s0 = wi * step
        s1 = s0 + win_len
        if s1 > T_valid:
            break
        t_center = (s0 + s1) / 2.0
        times_b[wi] = t_center  # still on trimmed IF timeline; caller will convert to sec

        for pi, (i, j) in enumerate(pairs):
            zi = z_bin_b[i, s0:s1]
            zj = z_bin_b[j, s0:s1]
            if not np.any(np.abs(zi) > 0) or not np.any(np.abs(zj) > 0):
                continue
            mets = _window_metrics_all(zi, zj, measures, debiased_wpli)
            for m, v in mets.items():
                values_b[m][pi, wi] = v

    return times_b, values_b


def emd_coherogram_epoch(
        X_ch_time,
        sfreq,
        freqs,
        n_cycles=6,
        step_frac=0.25,
        measures=('coh', 'imcoh', 'dpli', 'wpli'),
        max_imfs=8,
        edge_trim=None,
        min_samples=8,
        debiased_wpli=False,
        n_jobs=1,
        prefer="processes",
        max_nbytes="10M",
):
    """
    Parallel EMD-based coherogram on a single epoch.

    Returns
    -------
    out : dict
        'freqs': (n_freqs,)
        'times': (n_freqs, n_windows_max) in seconds (NaN-padded)
        'pairs': list of (i,j)
        'values': dict of arrays (n_pairs, n_freqs, n_windows_max)
    """
    X = X_ch_time.values if hasattr(X_ch_time, "values") else np.asarray(X_ch_time)
    if X.ndim != 2:
        raise ValueError("X_ch_time must be (n_channels, n_times).")
    n_ch, n_times = X.shape
    freqs = np.asarray(freqs, float)
    n_bins = len(freqs)
    f_edges = _freq_edges_from_centers(freqs)
    if edge_trim is None:
        edge_trim = int(round(0.1 * sfreq))

    # 1) Parallel EMD decomposition per channel
    decomp = Parallel(n_jobs=n_jobs, backend="loky" if prefer == "processes" else "threading",
                      max_nbytes=max_nbytes)(
        delayed(_emd_decompose_channel_worker)(X[ch], sfreq, max_imfs)
        for ch in range(n_ch)
    )
    Z = [d[0] for d in decomp]  # list of (K_c, T)
    IF = [d[1] for d in decomp]  # list of (K_c, T-1)

    # Shared IF timeline length (use min to be safe)
    T_if = min(ifreq.shape[-1] for ifreq in IF)
    # Apply edge trim
    t0 = min(edge_trim, max(T_if // 4, 0))
    t1 = T_if - t0
    if t1 <= t0:
        t0, t1 = 0, T_if
    T_valid = t1 - t0

    # 2) Parallel z-bin construction per channel
    z_bin_list = Parallel(n_jobs=n_jobs, backend="loky" if prefer == "processes" else "threading",
                          max_nbytes=max_nbytes)(
        delayed(_build_zbin_channel_worker)(Z[ch], IF[ch][:, :T_if], f_edges, t0, t1, n_bins)
        for ch in range(n_ch)
    )
    # Stack into (n_ch, n_bins, T_valid)
    z_bin = np.stack(z_bin_list, axis=0)

    # 3) Parallel per-frequency-bin windowed connectivity
    # To reduce memory transfer, feed each worker only the per-bin slice (n_ch, T_valid)
    backend = "loky" if prefer == "processes" else "threading"
    bin_results = Parallel(n_jobs=n_jobs, backend=backend, max_nbytes=max_nbytes)(
        delayed(_coh_bin_worker)(
            bi,
            z_bin[:, bi, :],  # (n_ch, T_valid) complex
            sfreq,
            freqs[bi],
            n_cycles,
            step_frac,
            measures,
            debiased_wpli,
            min_samples
        )
        for bi in range(n_bins)
    )

    # 4) Assemble outputs (pad to max windows across bins)
    n_pairs = len(_pairs(n_ch))
    max_windows = max(len(times_b) for times_b, _ in bin_results)
    times = np.full((n_bins, max_windows), np.nan, float)
    values = {m: np.full((n_pairs, n_bins, max_windows), np.nan, float) for m in measures}

    # Convert times from trimmed IF index to seconds: (t_center + 1 + t0) / sfreq
    for bi, (times_b, vals_b) in enumerate(bin_results):
        times_sec = (times_b + 1 + t0) / sfreq
        times[bi, :len(times_sec)] = times_sec
        for m in measures:
            arr = vals_b[m]  # (n_pairs, n_win)
            values[m][:, bi, :arr.shape[1]] = arr

    out = dict(freqs=freqs, times=times, pairs=_pairs(n_ch), values=values)
    return out


def single_freezing_epochs_analysis(lfp, animal_name, areas_animal_clean, timings, timing_label, params,
                                    redo=False):
    freqs_pwr = np.arange(3, 6,
                          0.25)  # Example frequency range
    freqs_cohe = np.arange(3, 6,
                           0.2)
    results_folder_path = '/mnt/308A3DD28A3D9576/single_freezing_analysis/' + animal_name + '/results/'
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)

    if not os.path.exists(results_folder_path) or redo:
        if len(timings):

            mean = np.nanmean(lfp.values, axis=1, keepdims=True)
            std = np.nanstd(lfp.values, axis=1, keepdims=True)
            zscored = (lfp.values - mean) / std
            zscored_lfp = pd.DataFrame(zscored, index=lfp.index,
                                       columns=lfp.columns)  # need to do it like this because scipy does not have the option to ignore nans for zscoring

            for i, (onset, offset) in tqdm(timings.iterrows()):

                single_timing_df_for_psds = pd.DataFrame(columns=timings.columns,
                                                         data=pd.DataFrame(columns=timings.columns,
                                                                           data=[[onset, offset]]))
                within_timings_lfp_for_psds = extract_signal_within_timings(zscored_lfp, single_timing_df_for_psds,
                                                                            min_len_epoch=4000, decimate=False,
                                                                            zscore_each=False)
                if len(within_timings_lfp_for_psds):
                    single_timing_psds = tfr_array_multitaper(within_timings_lfp_for_psds, sfreq=params.sample_rate,
                                                              freqs=freqs_pwr,
                                                              n_cycles=params.psds_n_cycles,
                                                              time_bandwidth=params.psds_time_bandwidth, n_jobs=-1,
                                                              output='avg_power', verbose=True)

                    timings_avg_psds = pd.DataFrame(single_timing_psds.mean(axis=2))

                    results_key_psds = '/single_{}_epoch_psds_{}_len_{}.h5'.format(timing_label, i, int(offset - onset))
                    psds_results_path = results_folder_path + results_key_psds

                    if os.path.exists(psds_results_path):
                        os.remove(psds_results_path)
                        print(f"File {psds_results_path} deleted, writing a new one.")
                    else:
                        print(f"File {psds_results_path} not found, writing a new one.")

                        timings_avg_psds.to_hdf(psds_results_path, results_key_psds)

                within_timings_lfp_for_cohe = extract_signal_within_timings(zscored_lfp, single_timing_df_for_psds,
                                                                            min_len_epoch=4000, decimate=False,
                                                                            zscore_each=False)
                if len(within_timings_lfp_for_cohe):

                    seeds, targets = seed_target_multivariate_indices([[i] for i in range(len(areas_animal_clean))],
                                                                      [[i] for i in range(len(areas_animal_clean))])

                    univariate_measures = ['coh', 'imcoh', 'dpli']

                    avg_within_timings_univariate_coherence_measures = spectral_connectivity_epochs(
                        within_timings_lfp_for_cohe,
                        method=univariate_measures,
                        cwt_freqs=freqs_cohe,
                        sfreq=params.sample_rate,
                        names=list(areas_animal_clean.values()),
                        mode='cwt_morlet', n_jobs=-1, cwt_n_cycles=1.0,
                        )

                    avg_within_timings_univariate_coherence_wpli = spectral_connectivity_epochs(
                        within_timings_lfp_for_cohe,
                        method=['wpli'],
                        cwt_freqs=freqs_cohe,
                        sfreq=params.sample_rate,
                        indices=(
                            [i[0] for i in seeds],
                            [i[0] for i in targets]),
                        mode='cwt_morlet',
                        n_jobs=-1, cwt_n_cycles=1.0,
                        )

                    avg_within_timings_univariate_coherence_measures.append(
                        avg_within_timings_univariate_coherence_wpli)
                    univariate_measures = ['coh', 'imcoh', 'dpli', 'wpli']

                    results_key_cohe = '/single_{}_epoch_cohe_{}_len_{}.pkl'.format(timing_label, i,
                                                                                    int(offset - onset))
                    cohe_results_path = results_folder_path + results_key_cohe

                    for i, univariate_measure in enumerate(univariate_measures):
                        avg_within_timings_univariate_coherence_measures[i].save(cohe_results_path + univariate_measure)

                    # for multivariate_measure in multivariate_measures:
                    #     avg_within_timings_multivariate_coherence_measures.save(cohe_results_path + multivariate_measure)
    pass


def single_freezing_time_frequency_analysis(lfp, animal_name, areas_animal_clean, timings, timing_label, params,
                                            redo=True):
    freqs_pwr = np.arange(1, 5,
                          0.2)  # Example frequency range
    freqs_cohe = np.arange(1, 5,
                           0.25)
    results_folder_path = '/media/prignane/data_fast/single_freezing_analysis/' + animal_name + '/results/'
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)

    if not os.path.exists(results_folder_path) or redo:
        if len(timings):

            mean = np.nanmean(lfp.values, axis=1, keepdims=True)
            std = np.nanstd(lfp.values, axis=1, keepdims=True)
            zscored = (lfp.values - mean) / std
            zscored_lfp = pd.DataFrame(zscored, index=lfp.index,
                                       columns=lfp.columns)  # need to do it like this because scipy does not have the option to ignore nans for zscoring

            for i, (onset, offset) in tqdm(timings.iterrows()):

                within_timings_lfp_for_spect = extract_signal_around_timing_df(zscored_lfp, onset,
                                                                               offset, 4000, 4000)
                if within_timings_lfp_for_spect.shape[1] >= 12000 and not within_timings_lfp_for_spect.isna().any(
                        axis=1).all():
                    print('has data')
                    single_timing_spect = tfr_array_multitaper(np.array([within_timings_lfp_for_spect]),
                                                               sfreq=params.sample_rate,
                                                               freqs=freqs_pwr,
                                                               n_cycles=params.psds_n_cycles,
                                                               time_bandwidth=params.psds_time_bandwidth, n_jobs=-1,
                                                               output='power', verbose=True)

                    results_key_spect = '/single_{}_epoch_spect_{}_len_{}.h5'.format(timing_label, i,
                                                                                     int(offset - onset))
                    spect_results_path = results_folder_path + results_key_spect

                    if os.path.exists(spect_results_path):
                        os.remove(spect_results_path)
                        print(f"File {spect_results_path} deleted, writing a new one.")
                    else:
                        print(f"File {spect_results_path} not found, writing a new one.")

                    with open(spect_results_path, 'wb') as f:
                        pickle.dump(single_timing_spect[0].mean(axis=1), f)

                    spect_base = f'/single_{timing_label}_epoch_spect_{i}_len_{int(offset - onset)}.h5'
                    spect_results_path = results_folder_path + spect_base
                    if os.path.exists(spect_results_path):
                        os.remove(spect_results_path)
                    with open(spect_results_path, 'wb') as f:
                        # NOTE: This averages across frequencies (axis=1), NOT across channels.
                        # Result shape: (n_channels, n_times)
                        pickle.dump(single_timing_spect[0].mean(axis=1), f)

                    # EMD/HHT spectrogram (per channel, no channel averaging)
                    X = within_timings_lfp_for_spect.values  # (n_channels, n_times)
                    hht = hht_spectrogram_epoch_per_channel(
                        X, sfreq=params.sample_rate, freqs=freqs_pwr,
                        edge_trim=int(0.1 * params.sample_rate)
                    )  # (n_channels, n_freqs, n_times)

                    spect_results_path_emd = results_folder_path + spect_base.replace('.h5', '_emd.h5')
                    if os.path.exists(spect_results_path_emd):
                        os.remove(spect_results_path_emd)

                    with open(spect_results_path_emd, 'wb') as f:
                        # To match your current saved shape (n_channels, n_times),
                        # we average over frequencies (axis=1), NOT channels.
                        # If you prefer to save the full per-channel spectrogram,
                        # replace the next line with: pickle.dump(hht, f)
                        pickle.dump(hht, f)

                else:
                    print('no data, only with nans')
                within_timings_lfp_for_coherogram = extract_signal_around_timing_df(zscored_lfp, onset,
                                                                                    offset, 4000, 4000)

                if within_timings_lfp_for_coherogram.shape[
                    1] >= 12000 and not within_timings_lfp_for_coherogram.isna().any(axis=1).all():

                    univariate_measures = ['coh', 'imcoh', 'dpli']
                    time_bandwidth_product = (params.coheros_n_tapers + 1) / 2

                    avg_within_timings_univariate_coherogram = spectral_connectivity_epochs(
                        np.array([within_timings_lfp_for_coherogram]),
                        method=univariate_measures,
                        names=list(areas_animal_clean.values()),
                        sfreq=params.sample_rate,
                        cwt_freqs=freqs_cohe,
                        cwt_n_cycles=params.coheros_n_cycles,
                        mode='cwt_morlet',
                        n_jobs=-1,
                        # need those jobs and block size settings to avoid memory error
                        mt_bandwidth=time_bandwidth_product, faverage=True)

                    results_key_cohe = '/single_{}_epoch_coherogram_{}_len_{}.pkl'.format(timing_label, i,
                                                                                          int(offset - onset))
                    coherogram_results_path = results_folder_path + results_key_cohe

                    for i, univariate_measure in enumerate(univariate_measures):
                        avg_within_timings_univariate_coherogram[i].save(coherogram_results_path + univariate_measure)

                    # X_epoch: DataFrame or ndarray of shape (n_channels, n_times)
                    emd_coh = emd_coherogram_epoch(
                        pd.DataFrame(within_timings_lfp_for_coherogram), sfreq=params.sample_rate,
                        freqs=np.arange(1, 5, 0.25),
                        n_cycles=params.coheros_n_cycles,
                        step_frac=0.25,
                        measures=('coh', 'imcoh', 'dpli', 'wpli'),
                        max_imfs=8,
                        debiased_wpli=False  # set True if you want debiased wPLI
                    )

                    # Save one file per measure, e.g., ..._emd.pkl + measure name
                    base = f"{results_folder_path}/single_{timing_label}_epoch_coherogram_{i}_len_{int(offset - onset)}_emd.pkl"
                    for meas, arr in emd_coh['values'].items():
                        payload = dict(
                            freqs=emd_coh['freqs'],
                            times=emd_coh['times'],
                            pairs=emd_coh['pairs'],
                            values=arr
                        )
                        with open(base + meas, 'wb') as f:
                            pickle.dump(payload, f)

    pass

