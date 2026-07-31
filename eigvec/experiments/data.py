"""Dataset-agnostic signal matrix utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from eigvec.core import decompose_fourier


@dataclass
class SignalMatrix:
    """A matrix of observations by time with one metadata row per observation."""

    X: np.ndarray
    rows: pd.DataFrame
    fs: float
    condition: str | None = None

    def __post_init__(self):
        if self.X.ndim != 2:
            raise ValueError("X must be 2D: rows=observations, columns=time.")
        if len(self.rows) != self.X.shape[0]:
            raise ValueError("rows must have one entry per matrix row.")

    def take(self, indices, reset_index: bool = True) -> "SignalMatrix":
        rows = self.rows.iloc[indices]
        if reset_index:
            rows = rows.reset_index(drop=True)
        return SignalMatrix(self.X[indices], rows, self.fs, self.condition)


def standardize_rows(X: np.ndarray, eps: float = 1e-12, demean: bool = False) -> np.ndarray:
    """Center and scale each observation row without temporal filtering."""
    X = np.asarray(X, dtype=np.float32).copy()
    X -= X.mean(axis=1, keepdims=True)
    X /= X.std(axis=1, keepdims=True) + eps
    if demean:
        X -= X.mean(axis=0, keepdims=True)
    return X


def window_traces(
    traces: np.ndarray,
    fs: float,
    window_seconds: float,
    step_seconds: float | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Turn continuous traces into channel/subject x time-window rows.

    Parameters
    ----------
    traces
        Array shaped ``n_traces x n_samples``.
    fs
        Sampling rate in Hz.
    window_seconds, step_seconds
        Window and step size. Incomplete tail samples are dropped.
    """

    traces = np.asarray(traces)
    if traces.ndim != 2:
        raise ValueError("traces must be 2D: n_traces x n_samples.")

    step_seconds = window_seconds if step_seconds is None else step_seconds
    window_samples = int(round(window_seconds * fs))
    step_samples = int(round(step_seconds * fs))
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("window_seconds and step_seconds must be positive.")

    blocks = []
    rows = []
    n_traces, n_samples = traces.shape
    for window_index, start in enumerate(range(0, n_samples - window_samples + 1, step_samples)):
        stop = start + window_samples
        blocks.append(traces[:, start:stop])
        for trace_index in range(n_traces):
            rows.append(
                {
                    "trace_index": trace_index,
                    "window_index": window_index,
                    "window_start_s": start / fs,
                    "window_stop_s": stop / fs,
                    "window_seconds": window_samples / fs,
                    "step_seconds": step_samples / fs,
                }
            )

    if not blocks:
        raise RuntimeError("No complete windows could be extracted.")
    return np.vstack(blocks), pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Multimodal eigenspectrum notebook helpers

# Heavy imports used by the multimodal notebook loaders/fitting helpers.
import json
from io import BytesIO
import gzip
import hashlib
import os
import re
import sys
import tarfile
import zipfile
from math import gcd
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, urlretrieve

Path('/tmp/matplotlib').mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import h5py
import matplotlib.pyplot as plt
import mne
from scipy.io import loadmat
from scipy.signal import find_peaks, resample_poly
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from mne.datasets import eegbci, epilepsy_ecog, sleep_physionet, ssvep
from timescales.autoreg import ARPSD
from timescales.autoreg.fit import _ar_spectrum

from eigvec.core import mirror_pos_to_full, reconstruct
from .row_selection import filter_signal_rows_for_stationary_oscillation
from .spectral import RobustAperiodicAR, mask_peaks

mne.set_log_level('WARNING')


def _module_settings():
    return sys.modules[__name__].__dict__


# Default notebook settings. Override from a notebook with an explicit settings mapping.

WORKDIR = Path.cwd()
if (WORKDIR / 'notebooks').exists():
    NOTEBOOK_DIR = WORKDIR / 'notebooks'
elif WORKDIR.name == 'notebooks':
    NOTEBOOK_DIR = WORKDIR
elif WORKDIR.parent.name == 'notebooks':
    NOTEBOOK_DIR = WORKDIR.parent
else:
    NOTEBOOK_DIR = WORKDIR

DATA_ROOT = NOTEBOOK_DIR / 'data' / 'multimodal_cache'
ALLEN_CACHE = NOTEBOOK_DIR / 'data' / 'ecephys_cache_dir'

# Select rows for the final figure. Defaults are sustained oscillatory states;
# transient/phase-locked examples are preserved in DATASET_REGISTRY as optional.
# DATASET_REGISTRY = {
#     'human_eegbci_alpha': load_human_eegbci,
#     'human_lemon_alpha': load_human_lemon_alpha,
#     'human_pardo_valencia_stn_beta': load_human_pardo_valencia_stn_beta,
#     'human_bciciii_ecog_motor_beta': load_human_bciciii_ecog_motor_beta,
#     'human_miller_fingerflex_beta': load_human_miller_fingerflex_beta,
#     'human_sleep_spindles': load_human_sleep_spindles,
#     'human_ssvep_12hz': load_human_ssvep,
#     'human_epilepsy_ecog': load_human_epilepsy_ecog,
#     'mouse_neuropixels_natural_movie': load_mouse_neuropixels_natural_movie,
#     'rat_hc3_ca1_ca3': load_hc3_hippocampal_eeg,
#     'rat_medial_septum_theta': load_rat_medial_septum_theta,
#     'macaque_visual_grating_ecog': load_macaque_visual_grating_ecog,
#     'macaque_neurotycho_ecog': lambda: load_neurotycho_signal('ECoG'),
# }

DATASET_KEYS = [
    'human_lemon_alpha',
    # Human STN LFP beta benchmark with 1000 beta-dominant 1 s rows.
    'human_pardo_valencia_stn_beta',
    # 'human_cole_m1_beta',
    #'human_ssvep_12hz',
    'macaque_visual_grating_ecog',
    'rat_medial_septum_theta',
    'rat_hc3_ca1_ca3',
    'mouse_neuropixels_natural_movie',
]
DATASET_KEYS = DATASET_KEYS[::-1]

# Duration of each row in X. Larger values increase frequency resolution but
# reduce the number of windows/trials available to average into the eigenspectrum.
WINDOW_S = 1.0

# Number of example rows plotted per dataset.
N_PLOT = 5

# Minimum acceptable circulant score. Kappa close to 1 means the covariance is
# close to diagonal in the Fourier basis, so the decomposition is better posed.
MIN_KAPPA = 0.9

# Row selection searches PSD-deviance thresholds and keeps the largest subset
# that reaches this target; if none reaches it, the highest-kappa subset is used.
KAPPA_TARGET = 0.95

# Aperiodic ARPSD order. Use 5 or 10; AR(5) is usually less likely to absorb the
# oscillatory bump, while AR(10) is more flexible for curved aperiodic spectra.
PSD_ORDER = 5

# Signal-level periodic-component definition. Quadrature is the general
# default because it exactly preserves the fitted aperiodic power, positive
# power excess, and signal additivity. Band-specific lag balancing is opt-in.
PERIODIC_DECOMPOSITION_MODE = 'quadrature'

# Candidate oscillations must fall in this non-line-noise band. The lower bound
# avoids selecting slow drift; the upper bound keeps gamma while ignoring Nyquist edge effects.
PERIODIC_SCORE_BAND = (6, 120)

# Base line-noise frequencies whose harmonics are ignored for fitting and row selection.
# This masks line-noise examples without filtering the underlying signals.
LINE_NOISE_BASES = (50, 60)

# Iterative PSD-shape row cleaning threshold. Smaller values make rows more
# spectrally homogeneous before the kappa-selection pass.
ROW_QC_PSD_DEVIANCE_Z = 3.0

# Shared preprocessing gate borrowed from the hippocampal theta notebook.
# Rows are kept only when the loader's target rhythm is sustained and spectrally stable.
APPLY_SIGNAL_QC = False
STANDARDIZE_FOR_MODEL = True
SIGNAL_QC_PSD_RANGE = (1, 120)
SIGNAL_QC_MIN_OSC_PROMINENCE = 2.0
SIGNAL_QC_MIN_OSC_FRACTION = 0.08
SIGNAL_QC_MIN_HALF_PSD_CORR = 0.35
SIGNAL_QC_MIN_PSD_REFERENCE_CORR = 0.40
SIGNAL_QC_MAX_MEAN_ROBUST_Z = 8.0
SIGNAL_QC_MAX_LOG_STD_ROBUST_Z = 6.0
SIGNAL_QC_MIN_ROWS_PER_DATASET = 250

DATA_ROOT.mkdir(parents=True, exist_ok=True)

# Implementation defaults kept out of the main settings cell.
# ROW_QC_MEAN_Z = 8.0
# ROW_QC_STD_LOG_Z = 6.0
ROW_QC_MEAN_Z = 5.0
ROW_QC_STD_LOG_Z = 5.0
ROW_QC_MAX_ITER = 8
ROW_QC_FREQ_RANGE = (1, 120)
ROW_QC_CHUNK_ROWS = 4096
LINE_NOISE_BASE_WIDTH = 6.0
LINE_NOISE_HARMONIC_WIDTH = 4.0
NEUROTYCHO_WINDOWS = 10_000


def download_to_cache(url, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f'Using cached {out_path}')
    else:
        print(f'Downloading {url}')
        urlretrieve(url, out_path)
    return out_path


def zscore_rows(X, eps=1e-12, dtype=np.float32, drop_nonfinite=True):
    X = np.asarray(X, dtype=dtype)
    if drop_nonfinite:
        X = X[np.isfinite(X).all(axis=1)]
    X = X - X.mean(axis=1, keepdims=True)
    X = X / (X.std(axis=1, keepdims=True) + eps)
    return X.astype(dtype, copy=False)


def zscore_global(X, eps=1e-12, dtype=np.float32, drop_nonfinite=True):
    X = np.asarray(X, dtype=dtype)
    if drop_nonfinite:
        X = X[np.isfinite(X).all(axis=1)]
    X = X - X.mean()
    X = X / (X.std() + eps)
    return X.astype(dtype, copy=False)


def flatten_epochs_channels(data, normalize='rows'):
    data = np.asarray(data)
    X = data.reshape(-1, data.shape[-1])
    if normalize in (None, 'none'):
        return X
    return zscore_global(X) if normalize == 'global' else zscore_rows(X)


def continuous_to_windows(data, fs, window_s=WINDOW_S, max_windows=None, normalize='rows'):
    data = np.asarray(data)
    n_win = int(round(window_s * fs))
    n_windows_available = data.shape[-1] // n_win
    n_windows = n_windows_available if max_windows is None else min(max_windows, n_windows_available)
    if n_windows < 1:
        raise ValueError('Not enough samples for one analysis window.')

    data = data[:, : n_windows * n_win]
    data = data.reshape(data.shape[0], n_windows, n_win)
    data = np.transpose(data, (1, 0, 2))
    return flatten_epochs_channels(data, normalize=normalize), n_windows


def finite_positive_mask(freqs, powers):
    return np.isfinite(freqs) & np.isfinite(powers) & (freqs > 0) & (powers > 0)


def robust_zscore(values, eps=1e-12):
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        scale = np.nanstd(values)
    return (values - median) / (scale + eps)


def robust_center_scale(values, eps=1e-12):
    values = np.asarray(values, dtype=float)
    center = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - center))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        scale = np.nanstd(values)
    return center, scale + eps


def merge_frequency_bands(bands):
    bands = sorted((max(0.0, float(low)), float(high)) for low, high in bands if high > 0)
    if not bands:
        return tuple()
    merged = [list(bands[0])]
    for low, high in bands[1:]:
        if low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return tuple((low, high) for low, high in merged)


def line_noise_bands(fs):
    max_freq = fs / 2
    bands = []
    for base in LINE_NOISE_BASES:
        harmonic = 1
        while base * harmonic <= max_freq:
            center = base * harmonic
            width = LINE_NOISE_BASE_WIDTH if harmonic == 1 else LINE_NOISE_HARMONIC_WIDTH
            bands.append((center - width, center + width))
            harmonic += 1
    return merge_frequency_bands(bands)


def mask_frequency_bands(freqs, bands):
    mask = np.ones_like(freqs, dtype=bool)
    for low, high in bands:
        mask &= ~((freqs >= low) & (freqs <= high))
    return mask


def line_noise_frequency_mask(freqs, fs):
    return ~mask_frequency_bands(freqs, line_noise_bands(fs))


def row_qc_frequency_mask(n, fs):
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    low, high = ROW_QC_FREQ_RANGE
    high = min(high, 0.45 * fs)
    mask = (freqs >= low) & (freqs <= high)
    mask &= ~line_noise_frequency_mask(freqs, fs)
    return freqs, mask


def row_log_psd_center(X, fs, row_mask, freq_mask, chunk_rows=ROW_QC_CHUNK_ROWS, eps=1e-12):
    X = np.asarray(X)
    center = np.zeros(int(freq_mask.sum()), dtype=float)
    n_rows = 0
    active = np.flatnonzero(row_mask)
    for start in range(0, len(active), chunk_rows):
        inds = active[start:start + chunk_rows]
        powers = np.abs(np.fft.rfft(X[inds], axis=1)) ** 2
        log_powers = np.log10(powers[:, freq_mask] + eps)
        center += log_powers.sum(axis=0)
        n_rows += len(inds)
    center /= max(n_rows, 1)
    return center


def row_log_psd_distances(X, center, freq_mask, chunk_rows=ROW_QC_CHUNK_ROWS, eps=1e-12):
    X = np.asarray(X)
    distances = np.full(X.shape[0], np.nan, dtype=float)
    for start in range(0, X.shape[0], chunk_rows):
        chunk = X[start:start + chunk_rows]
        powers = np.abs(np.fft.rfft(chunk, axis=1)) ** 2
        log_powers = np.log10(powers[:, freq_mask] + eps)
        distances[start:start + len(chunk)] = np.sqrt(
            np.mean((log_powers - center[None, :]) ** 2, axis=1)
        )
    return distances


def iterative_psd_deviance_qc(X, fs):
    freqs, freq_mask = row_qc_frequency_mask(X.shape[1], fs)
    keep = np.ones(X.shape[0], dtype=bool)
    history = []
    if not np.any(freq_mask):
        return keep, {
            'row_qc_psd_iterations': 0,
            'row_qc_psd_deviance_z': float(ROW_QC_PSD_DEVIANCE_Z),
            'row_qc_psd_distance_median': np.nan,
            'row_qc_psd_distance_q95': np.nan,
            'row_qc_psd_distance_max': np.nan,
            'row_qc_frequency_range_hz': str(ROW_QC_FREQ_RANGE),
            'row_qc_line_noise_bands_hz': str(line_noise_bands(fs)),
        }

    final_distances = None
    for iteration in range(ROW_QC_MAX_ITER):
        center = row_log_psd_center(X, fs, keep, freq_mask)
        distances = row_log_psd_distances(X, center, freq_mask)
        active_distances = distances[keep]
        dev_center, dev_scale = robust_center_scale(active_distances)
        deviance_z = (distances - dev_center) / dev_scale
        next_keep = keep & np.isfinite(deviance_z) & (deviance_z <= ROW_QC_PSD_DEVIANCE_Z)
        dropped = int(keep.sum() - next_keep.sum())
        history.append({
            'iteration': iteration + 1,
            'rows_before': int(keep.sum()),
            'rows_after': int(next_keep.sum()),
            'dropped': dropped,
            'distance_median': float(np.nanmedian(active_distances)),
            'distance_q95': float(np.nanquantile(active_distances, 0.95)),
            'distance_max': float(np.nanmax(active_distances)),
        })
        keep = next_keep
        final_distances = distances
        if dropped == 0:
            break
        if keep.sum() < N_PLOT:
            raise ValueError('Too few rows after iterative PSD row QC.')

    kept_distances = final_distances[keep]
    summary = {
        'row_qc_psd_iterations': int(len(history)),
        'row_qc_psd_deviance_z': float(ROW_QC_PSD_DEVIANCE_Z),
        'row_qc_psd_distance_median': float(np.nanmedian(kept_distances)),
        'row_qc_psd_distance_q95': float(np.nanquantile(kept_distances, 0.95)),
        'row_qc_psd_distance_max': float(np.nanmax(kept_distances)),
        'row_qc_psd_history': str(history),
        'row_qc_frequency_range_hz': str(ROW_QC_FREQ_RANGE),
        'row_qc_line_noise_bands_hz': str(line_noise_bands(fs)),
    }
    return keep, summary


def bound_frequency_range(freq_range, fs, lower_bound=0.0):
    low, high = map(float, freq_range)
    low = max(float(lower_bound), low)
    high = min(high, 0.45 * float(fs))
    if not np.isfinite(high) or high <= low:
        high = min(float(fs) / 2, low + max(1.0, 0.1 * float(fs)))
    return (low, high)


def _stationary_qc_summary(
    enabled,
    applied,
    rows_before,
    rows_after,
    *,
    osc_band=None,
    psd_range=None,
    skip_reason=None,
    min_rows_per_dataset=None,
):
    rows_before = int(rows_before)
    rows_after = int(rows_after)
    if min_rows_per_dataset is None:
        min_rows_per_dataset = SIGNAL_QC_MIN_ROWS_PER_DATASET
    return {
        'signal_qc_enabled': bool(enabled),
        'signal_qc_applied': bool(applied),
        'signal_qc_skip_reason': skip_reason or '',
        'signal_qc_osc_band_hz': str(tuple(osc_band) if osc_band is not None else ()),
        'signal_qc_psd_range_hz': str(tuple(psd_range) if psd_range is not None else ()),
        'signal_qc_rows_before': rows_before,
        'signal_qc_rows_after': rows_after,
        'signal_qc_rows_dropped': int(rows_before - rows_after),
        'signal_qc_fraction_kept': float(rows_after / rows_before) if rows_before else np.nan,
        'signal_qc_min_osc_prominence': float(SIGNAL_QC_MIN_OSC_PROMINENCE),
        'signal_qc_min_osc_fraction': float(SIGNAL_QC_MIN_OSC_FRACTION),
        'signal_qc_min_half_psd_corr': float(SIGNAL_QC_MIN_HALF_PSD_CORR),
        'signal_qc_min_psd_reference_corr': float(SIGNAL_QC_MIN_PSD_REFERENCE_CORR),
        'signal_qc_max_mean_robust_z': float(SIGNAL_QC_MAX_MEAN_ROBUST_Z),
        'signal_qc_max_log_std_robust_z': float(SIGNAL_QC_MAX_LOG_STD_ROBUST_Z),
        'signal_qc_min_rows_per_dataset': int(min_rows_per_dataset),
    }


def stationary_oscillation_row_qc(X, fs, row_indices, osc_band=None, signal_qc_min_rows=None):
    X = np.asarray(X, dtype=np.float32)
    row_indices = np.asarray(row_indices)
    osc_band = bound_frequency_range(PERIODIC_SCORE_BAND if osc_band is None else osc_band, fs)
    psd_range = bound_frequency_range(SIGNAL_QC_PSD_RANGE, fs)
    rows_before = X.shape[0]
    min_rows_goal = SIGNAL_QC_MIN_ROWS_PER_DATASET if signal_qc_min_rows is None else int(signal_qc_min_rows)

    if not APPLY_SIGNAL_QC:
        summary = _stationary_qc_summary(False, False, rows_before, rows_before, osc_band=osc_band, psd_range=psd_range, min_rows_per_dataset=min_rows_goal)
        return X, row_indices, summary
    if rows_before <= N_PLOT:
        summary = _stationary_qc_summary(True, False, rows_before, rows_before, osc_band=osc_band, psd_range=psd_range, skip_reason='too_few_rows', min_rows_per_dataset=min_rows_goal)
        return X, row_indices, summary
    if osc_band[0] >= psd_range[1]:
        summary = _stationary_qc_summary(True, False, rows_before, rows_before, osc_band=osc_band, psd_range=psd_range, skip_reason='osc_band_outside_psd_range', min_rows_per_dataset=min_rows_goal)
        return X, row_indices, summary

    rows = pd.DataFrame({
        'pre_signal_qc_row': np.arange(rows_before, dtype=int),
        'source_row_index': row_indices,
    })
    matrix = SignalMatrix(X, rows, float(fs), condition='all_rows')
    min_rows = min(rows_before, max(N_PLOT, int(min_rows_goal)))
    filtered, metrics, summary_table = filter_signal_rows_for_stationary_oscillation(
        {'all_rows': matrix},
        fs=float(fs),
        osc_band=osc_band,
        psd_range=psd_range,
        min_osc_prominence=SIGNAL_QC_MIN_OSC_PROMINENCE,
        min_osc_fraction=SIGNAL_QC_MIN_OSC_FRACTION,
        min_half_psd_corr=SIGNAL_QC_MIN_HALF_PSD_CORR,
        min_psd_reference_corr=SIGNAL_QC_MIN_PSD_REFERENCE_CORR,
        max_mean_robust_z=SIGNAL_QC_MAX_MEAN_ROBUST_Z,
        max_log_std_robust_z=SIGNAL_QC_MAX_LOG_STD_ROBUST_Z,
        min_rows_per_condition=min_rows,
    )
    kept = filtered['all_rows']
    kept_rows = kept.rows.reset_index(drop=True)
    summary = _stationary_qc_summary(True, True, rows_before, kept.X.shape[0], osc_band=osc_band, psd_range=psd_range, min_rows_per_dataset=min_rows_goal)
    if not summary_table.empty:
        row = summary_table.iloc[0]
        summary.update({
            'signal_qc_median_osc_prominence': float(row.get('median_osc_prominence', np.nan)),
            'signal_qc_median_osc_fraction': float(row.get('median_osc_fraction', np.nan)),
            'signal_qc_median_half_psd_corr': float(row.get('median_half_psd_corr', np.nan)),
            'signal_qc_median_psd_reference_corr': float(row.get('median_psd_reference_corr', np.nan)),
            'signal_qc_fallback_rows': int(row.get('fallback_rows', 0)),
        })
    if metrics is not None and not metrics.empty:
        summary.update({
            'signal_qc_metric_rows': int(len(metrics)),
            'signal_qc_metric_keep_rows': int(metrics['qc_keep'].sum()),
        })
    return kept.X.astype(np.float32, copy=False), kept_rows['source_row_index'].to_numpy(), summary


def prepare_rows(X_raw, fs, normalize='rows', row_indices=None, qc=True, osc_band=None, signal_qc_min_rows=None):
    X_raw = np.asarray(X_raw, dtype=np.float32)
    row_indices = np.arange(X_raw.shape[0]) if row_indices is None else np.asarray(row_indices)
    if row_indices.shape[0] != X_raw.shape[0]:
        raise ValueError('row_indices must have one entry per row of X_raw.')

    finite_keep = np.isfinite(X_raw).all(axis=1)
    X_finite = X_raw[finite_keep]
    row_indices_finite = row_indices[finite_keep]
    if X_finite.shape[0] < N_PLOT:
        raise ValueError('Too few finite rows after row QC.')

    raw_means = X_finite.mean(axis=1)
    raw_stds = X_finite.std(axis=1)
    mean_z = robust_zscore(raw_means)
    std_log_z = robust_zscore(np.log(raw_stds + 1e-12))
    stat_keep = (
        (raw_stds > 1e-12)
        & (np.abs(mean_z) <= ROW_QC_MEAN_Z)
        & (np.abs(std_log_z) <= ROW_QC_STD_LOG_Z)
    )
    X_stat = X_finite[stat_keep]
    row_indices_stat = row_indices_finite[stat_keep]
    if X_stat.shape[0] < N_PLOT:
        raise ValueError('Too few rows after mean/variance row QC.')

    if qc:
        X_signal, row_indices_signal, signal_summary = stationary_oscillation_row_qc(
            X_stat,
            fs,
            row_indices_stat,
            osc_band=osc_band,
            signal_qc_min_rows=signal_qc_min_rows,
        )
    else:
        X_signal = X_stat
        row_indices_signal = row_indices_stat
        signal_summary = _stationary_qc_summary(False, False, X_stat.shape[0], X_stat.shape[0])
    if X_signal.shape[0] < N_PLOT:
        raise ValueError('Too few rows after sustained/stationary oscillation row QC.')

    normalize_mode = normalize
    if not STANDARDIZE_FOR_MODEL and normalize_mode == 'rows':
        normalize_mode = 'none'
    if normalize_mode == 'global':
        X_norm = zscore_global(X_signal, drop_nonfinite=False)
    elif normalize_mode in (None, 'none'):
        X_norm = X_signal.astype(np.float32, copy=False)
    else:
        X_norm = standardize_rows(X_signal).astype(np.float32, copy=False)

    if qc and X_norm.shape[0] > N_PLOT:
        spectral_keep, spectral_summary = iterative_psd_deviance_qc(X_norm, fs)
    else:
        spectral_keep = np.ones(X_norm.shape[0], dtype=bool)
        spectral_summary = {}

    X_keep = X_norm[spectral_keep]
    row_indices_keep = row_indices_signal[spectral_keep]
    if X_keep.shape[0] < N_PLOT:
        raise ValueError('Too few rows after spectral row QC.')

    summary = {
        'row_qc_applied': bool(qc),
        'standardize_for_model': bool(STANDARDIZE_FOR_MODEL),
        'normalization_mode': normalize_mode or 'none',
        'row_qc_rows_before': int(X_raw.shape[0]),
        'row_qc_rows_after_finite': int(finite_keep.sum()),
        'row_qc_rows_after_mean_std': int(stat_keep.sum()),
        'row_qc_rows_after_stationary_oscillation': int(X_signal.shape[0]),
        'row_qc_rows_after_spectral': int(X_keep.shape[0]),
        'row_qc_rows_removed': int(X_raw.shape[0] - X_keep.shape[0]),
        'row_qc_mean_z_threshold': float(ROW_QC_MEAN_Z),
        'row_qc_std_log_z_threshold': float(ROW_QC_STD_LOG_Z),
    }
    summary.update(signal_summary)
    summary.update(spectral_summary)
    return X_keep.astype(np.float32, copy=False), row_indices_keep, summary


def line_frequency_ratio(X, fs, target, bandwidth=2.0, eps=1e-12):
    X = zscore_rows(X, drop_nonfinite=True)
    freqs = np.fft.rfftfreq(X.shape[1], d=1 / fs)
    powers = np.sum(np.abs(np.fft.rfft(X, axis=1)) ** 2, axis=0)
    target_idx = int(np.argmin(np.abs(freqs - target)))
    neighbor_mask = (
        (freqs >= target - bandwidth)
        & (freqs <= target + bandwidth)
        & (np.abs(freqs - target) >= 1.0)
    )
    if not np.any(neighbor_mask):
        return np.nan
    return float(powers[target_idx] / (np.mean(powers[neighbor_mask]) + eps))


def make_dataset(name, X, fs, source, metadata, row_indices=None):
    metadata = dict(metadata)
    metadata.update({
        'dataset': name,
        'source': source,
        'fs_hz': float(fs),
        'rows': int(X.shape[0]),
        'time_samples': int(X.shape[1]),
        'duration_s_per_row': float(X.shape[1] / fs),
    })
    out = {
        'name': name,
        'X': X,
        'fs': float(fs),
        'source': source,
        'metadata': metadata,
    }
    if row_indices is not None:
        row_indices = np.asarray(row_indices, dtype=int)
        if len(row_indices) != X.shape[0]:
            raise ValueError('row_indices must match the number of retained rows.')
        out['row_indices'] = row_indices
    return out

def load_mouse_neuropixels_natural_movie(
    cache_dir=ALLEN_CACHE,
    stimulus='natural_movie_one',
    probe_description='probeA',
    region='VISam',
    fs=500,
    max_presentations=None,
    max_channels=None,
):
    cache = EcephysProjectCache.from_warehouse(
        manifest=str(cache_dir / 'manifest.json')
    )
    sessions = cache.get_session_table()
    session_id = sessions.index.values[0]
    session = cache.get_session_data(
        session_id,
        isi_violations_maximum=np.inf,
        amplitude_cutoff_maximum=np.inf,
        presence_ratio_minimum=-np.inf,
    )

    probe_id = session.probes[
        session.probes.description == probe_description
    ].index.values[0]
    lfp = session.get_lfp(probe_id)

    presentations_all = session.stimulus_presentations[
        session.stimulus_presentations.stimulus_name == stimulus
    ]
    if max_presentations is None:
        presentations = presentations_all
    else:
        presentations = presentations_all.head(max_presentations)

    trial_window = np.arange(0, WINDOW_S, 1 / fs)
    time_selection = np.concatenate([
        trial_window + t for t in presentations.start_time.values
    ])
    inds = pd.MultiIndex.from_product(
        (presentations.index.values, trial_window),
        names=('presentation_id', 'time_from_presentation_onset'),
    )

    ds = lfp.sel(time=time_selection, method='nearest').to_dataset(name='lfp')
    ds = ds.assign(time=inds).unstack('time')
    x = ds['lfp']

    channel_regions = np.array([
        session.channels.loc[channel_id, 'ecephys_structure_acronym']
        for channel_id in x.channel.to_numpy()
    ])
    region_inds = np.where(channel_regions == region)[0]
    if len(region_inds) == 0:
        region_inds = np.arange(len(channel_regions))
    if max_channels is not None:
        region_inds = region_inds[:max_channels]

    data = x.isel(channel=region_inds).transpose(
        'presentation_id', 'channel', 'time_from_presentation_onset'
    ).to_numpy()
    X_raw = flatten_epochs_channels(data, normalize=None)
    X, row_indices, row_qc = prepare_rows(X_raw, fs, normalize='rows', osc_band=(6, 12))

    metadata = {
        'species': 'mouse',
        'subject': str(sessions.loc[session_id].get('specimen_id', 'one mouse')),
        'session_id': int(session_id),
        'modality': 'Neuropixels LFP',
        'animal_or_subject_count': 1,
        'probe': probe_description,
        'brain_region': region,
        'electrodes_or_channels': int(len(region_inds)),
        'trials_or_windows': int(len(presentations)),
        'available_trials_or_windows': int(len(presentations_all)),
        'stimulus_or_state': stimulus,
        'normalization': 'row z-score after row QC',
        'row_definition': 'presentation x channel; row = presentation_index * n_channels + channel_index',
        'software_filter': 'none; Allen LFP as loaded from cache',
        'periodic_score_band_hz': (6, 12),
    }
    metadata.update(row_qc)

    return make_dataset(
        'Mouse Neuropixels LFP: natural movie',
        X,
        fs,
        'Allen Visual Coding Neuropixels',
        metadata,
        row_indices=row_indices,
    )


def load_human_eegbci(
    subject=1,
    run=2,
    channel='Oz',
    max_windows=None,
):
    paths = eegbci.load_data(
        subjects=[subject],
        runs=[run],
        path=str(DATA_ROOT / 'mne'),
        update_path=False,
    )
    raw = mne.io.read_raw_edf(paths[0], preload=True, verbose=False)
    eegbci.standardize(raw)
    raw.set_montage('standard_1005', on_missing='ignore')
    raw.pick(picks='eeg')
    if channel not in raw.ch_names:
        raise ValueError(f'EEGBCI channel {channel!r} not found.')
    raw.pick([channel])

    sfreq = raw.info['sfreq']
    X_raw, n_windows = continuous_to_windows(
        raw.get_data(),
        sfreq,
        window_s=WINDOW_S,
        max_windows=max_windows,
        normalize=None,
    )
    X, row_indices, row_qc = prepare_rows(X_raw, sfreq, normalize='rows', osc_band=(8, 13))
    state = 'eyes-closed baseline' if run == 2 else 'eyes-open baseline'

    metadata = {
        'species': 'human',
        'subject': f'S{subject:03d}',
        'animal_or_subject_count': 1,
        'modality': 'scalp EEG',
        'electrodes_or_channels': 1,
        'selected_channel': channel,
        'trials_or_windows': int(n_windows),
        'runs': str(run),
        'stimulus_or_state': state,
        'software_filter': 'none; raw PhysioNet EDF as loaded by MNE',
        'normalization': 'row z-score after row QC',
        'row_definition': f'{WINDOW_S:g} s windows from EEG channel {channel}; row = window_index',
        'periodic_score_band_hz': (8, 13),
    }
    metadata.update(row_qc)

    return make_dataset(
        'Human EEG: EEGBCI eyes-closed alpha',
        X,
        sfreq,
        'MNE EEGBCI / PhysioNet',
        metadata,
        row_indices=row_indices,
    )


LEMON_EEG_S3_PREFIX = 'https://fcp-indi.s3.amazonaws.com/data/Projects/INDI/MPI-LEMON/Compressed_tar/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID'


def download_lemon_subject(subject):
    subject = subject if str(subject).startswith('sub-') else f'sub-{int(subject):06d}'
    archive_path = DATA_ROOT / 'lemon' / f'{subject}.tar.gz'
    return download_to_cache(
        f'{LEMON_EEG_S3_PREFIX}/{subject}.tar.gz',
        archive_path,
    )


def extract_tar_to_cache(archive_path, out_dir):
    out_dir = Path(out_dir)
    if not out_dir.exists() or not any(out_dir.rglob('*')):
        out_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, 'r:gz') as tf:
            tf.extractall(out_dir)
    return out_dir


def fixed_brainvision_header(vhdr_path):
    vhdr_path = Path(vhdr_path)
    subject = vhdr_path.stem
    fixed_vhdr = vhdr_path.with_name(f'{subject}_fixed.vhdr')
    fixed_vmrk = vhdr_path.with_name(f'{subject}_fixed.vmrk')
    eeg_name = f'{subject}.eeg'
    vmrk_name = f'{subject}.vmrk'

    if not fixed_vmrk.exists():
        text = vhdr_path.with_name(vmrk_name).read_text(encoding='utf-8', errors='replace')
        text = re.sub(r'^(DataFile=).*$', rf'\1{eeg_name}', text, flags=re.MULTILINE)
        fixed_vmrk.write_text(text, encoding='utf-8')

    if not fixed_vhdr.exists():
        text = vhdr_path.read_text(encoding='utf-8', errors='replace')
        text = re.sub(r'^(DataFile=).*$', rf'\1{eeg_name}', text, flags=re.MULTILINE)
        text = re.sub(r'^(MarkerFile=).*$', rf'\1{fixed_vmrk.name}', text, flags=re.MULTILINE)
        fixed_vhdr.write_text(text, encoding='utf-8')

    return fixed_vhdr


def marker_segments(raw, marker='S210', max_gap_s=3.0, extend_s=2.0):
    descriptions = np.asarray(raw.annotations.description)
    onsets = np.asarray(raw.annotations.onset, dtype=float)
    keep = np.array([str(desc).endswith(marker) for desc in descriptions], dtype=bool)
    onsets = np.sort(onsets[keep])
    if len(onsets) == 0:
        raise RuntimeError(f'No LEMON marker {marker!r} found in raw annotations.')

    segments = []
    start = onsets[0]
    prev = onsets[0]
    raw_stop = raw.n_times / raw.info['sfreq']
    for onset in onsets[1:]:
        if onset - prev <= max_gap_s:
            prev = onset
        else:
            segments.append((start, min(prev + extend_s, raw_stop)))
            start = onset
            prev = onset
    segments.append((start, min(prev + extend_s, raw_stop)))
    return [(start, stop) for start, stop in segments if stop > start]


def windows_from_segments(raw, segments, window_s, max_windows=None):
    fs = raw.info['sfreq']
    win_samples = int(round(window_s * fs))
    rows = []
    row_indices = []
    window_index = 0
    n_channels = len(raw.ch_names)
    for start_s, stop_s in segments:
        start = max(0, int(round(start_s * fs)))
        stop = min(raw.n_times, int(round(stop_s * fs)))
        n_windows = (stop - start) // win_samples
        if n_windows <= 0:
            continue
        if max_windows is not None:
            n_windows = min(n_windows, max_windows - window_index)
        if n_windows <= 0:
            break
        data = raw.get_data(start=start, stop=start + n_windows * win_samples)
        data = data.reshape(n_channels, n_windows, win_samples).transpose(1, 0, 2)
        rows.append(data.reshape(-1, win_samples))
        base = np.arange(window_index, window_index + n_windows)[:, None] * n_channels
        row_indices.append((base + np.arange(n_channels)[None, :]).reshape(-1))
        window_index += n_windows
        if max_windows is not None and window_index >= max_windows:
            break
    if not rows:
        raise RuntimeError('No complete LEMON eyes-closed windows could be extracted.')
    return np.vstack(rows), np.concatenate(row_indices), window_index


def load_human_lemon_alpha(
    subject='sub-032302',
    channels=('O1', 'Oz', 'O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8'),
    marker='S210',
    window_s=1.0,
    max_windows=None,
):
    archive_path = download_lemon_subject(subject)
    extract_dir = extract_tar_to_cache(archive_path, archive_path.with_suffix('').with_suffix(''))
    vhdrs = sorted(extract_dir.rglob(f'{subject}.vhdr'))
    if not vhdrs:
        vhdrs = sorted(extract_dir.rglob('*.vhdr'))
    if not vhdrs:
        raise FileNotFoundError(f'No LEMON BrainVision header found under {extract_dir}.')

    fixed_vhdr = fixed_brainvision_header(vhdrs[0])
    raw = mne.io.read_raw_brainvision(fixed_vhdr, preload=True, verbose=False)
    available_channels = list(raw.ch_names)
    missing = [channel for channel in channels if channel not in raw.ch_names]
    if missing:
        raise ValueError(f'LEMON channels not found: {missing}. Available: {available_channels}')
    raw.pick(list(channels))

    segments = marker_segments(raw, marker=marker)
    X_raw, row_indices_raw, n_windows = windows_from_segments(
        raw,
        segments,
        window_s=window_s,
        max_windows=max_windows,
    )
    X, row_indices, row_qc = prepare_rows(
        X_raw,
        raw.info['sfreq'],
        normalize='rows',
        row_indices=row_indices_raw,
        osc_band=(8, 13),
    )
    state = 'eyes-closed rest' if marker == 'S210' else f'marker {marker} rest'

    metadata = {
        'species': 'human',
        'subject': subject,
        'animal_or_subject_count': 1,
        'modality': 'scalp EEG',
        'electrodes_or_channels': int(len(channels)),
        'available_electrodes_or_channels': int(len(available_channels)),
        'selected_channels': ', '.join(channels),
        'trials_or_windows': int(n_windows),
        'available_trials_or_windows': int(sum((stop - start) // window_s for start, stop in segments)),
        'runs': 'resting state EEG',
        'stimulus_or_state': f'LEMON {state}; marker {marker}',
        'software_filter': 'none applied in notebook; raw BrainVision signal as distributed',
        'normalization': 'row z-score after row QC',
        'row_definition': f'{window_s:g} s eyes-closed windows x posterior EEG channels; row = window_index * n_channels + channel_index',
        'dataset_note': 'LEMON raw resting EEG has 62 channels sampled at 2500 Hz; 1 s windows give n=2500 samples without resampling.',
        'periodic_score_band_hz': (8, 13),
        'cache_file': archive_path.name,
        'source_url': f'{LEMON_EEG_S3_PREFIX}/{subject}.tar.gz',
    }
    metadata.update(row_qc)

    return make_dataset(
        'Human EEG: LEMON eyes-closed alpha',
        X,
        raw.info['sfreq'],
        'MPI-LEMON raw resting EEG',
        metadata,
        row_indices=row_indices,
    )


def load_human_ssvep(
    subject='01',
    trial_type='stim/12hz',
    channel='Oz',
    trial_duration_s=30.0,
    trial_offset_s=2.0,
    window_stride_s=None,
    max_trials=None,
    max_windows=None,
):
    cache_root = DATA_ROOT / 'mne'
    root = cache_root / 'ssvep-example-data'
    if not root.exists():
        root = Path(ssvep.data_path(
            path=str(cache_root),
            update_path=False,
            download=True,
        ))
    vhdrs = sorted(root.rglob(f'sub-{subject}_ses-01_task-ssvep_eeg.vhdr'))
    if not vhdrs:
        vhdrs = sorted(cache_root.rglob(f'sub-{subject}_ses-01_task-ssvep_eeg.vhdr'))
    if not vhdrs:
        raise FileNotFoundError(f'No SSVEP BrainVision file found for sub-{subject}.')

    vhdr = vhdrs[0]
    events_path = vhdr.with_name(vhdr.name.replace('_eeg.vhdr', '_events.tsv'))
    events = pd.read_csv(events_path, sep='\t')
    events = events[events['trial_type'] == trial_type].copy()
    if max_trials is not None:
        events = events.head(max_trials)
    if events.empty:
        raise ValueError(f'No SSVEP events found for trial_type={trial_type!r}.')

    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
    raw.pick(picks='eeg')
    available_channels = list(raw.ch_names)
    if channel not in raw.ch_names:
        raise ValueError(f'SSVEP channel {channel!r} not found. Available: {available_channels}')
    raw.pick([channel])

    fs = raw.info['sfreq']
    if window_stride_s is None:
        window_stride_s = WINDOW_S
    if window_stride_s < WINDOW_S:
        raise ValueError(
            f'SSVEP windows must be non-overlapping: '
            f'window_stride_s={window_stride_s:g} is shorter than WINDOW_S={WINDOW_S:g}.'
        )
    win_samples = int(round(WINDOW_S * fs))
    stride_samples = max(1, int(round(window_stride_s * fs)))
    first_offset_samples = int(round(trial_offset_s * fs))
    usable_trial_s = max(trial_duration_s - trial_offset_s - WINDOW_S, 0)
    n_windows_per_trial = int(np.floor(usable_trial_s / window_stride_s)) + 1
    rows = []
    row_indices = []
    for trial_i, event in enumerate(events.itertuples(index=False)):
        trial_start = int(event.sample)
        for win_i in range(n_windows_per_trial):
            if max_windows is not None and len(rows) >= max_windows:
                break
            start = trial_start + first_offset_samples + win_i * stride_samples
            stop = start + win_samples
            if stop <= raw.n_times:
                rows.append(raw.get_data(start=start, stop=stop)[0])
                row_indices.append(trial_i * n_windows_per_trial + win_i)
        if max_windows is not None and len(rows) >= max_windows:
            break
    if not rows:
        raise RuntimeError('No complete SSVEP windows could be extracted.')

    X_raw = np.vstack(rows)
    X, row_indices_keep, row_qc = prepare_rows(
        X_raw,
        fs,
        normalize='rows',
        row_indices=np.asarray(row_indices),
        osc_band=(11, 13),
    )

    metadata = {
        'species': 'human',
        'subject': f'sub-{subject}',
        'animal_or_subject_count': 1,
        'modality': 'scalp EEG',
        'electrodes_or_channels': 1,
        'available_electrodes_or_channels': int(len(available_channels)),
        'selected_channel': channel,
        'trials_or_windows': int(X_raw.shape[0]),
        'available_trials_or_windows': int(len(events) * n_windows_per_trial),
        'stimulus_or_state': f'{trial_type} checkerboard SSVEP, homogeneous trials',
        'software_filter': 'none; raw BrainVision signal as loaded by MNE',
        'normalization': 'row z-score after row QC',
        'row_definition': f'{WINDOW_S:g} s windows from EEG channel {channel}, stride={window_stride_s:g} s after a {trial_offset_s:g} s onset skip',
        'dataset_note': 'Stimulus is a periodic checkerboard inversion, so narrow peaks at the stimulus frequency and harmonics are expected.',
        'periodic_score_band_hz': (11, 13),
    }
    metadata.update(row_qc)

    return make_dataset(
        f'Human EEG: SSVEP {trial_type.replace("stim/", "")}',
        X,
        fs,
        'MNE SSVEP / OSF',
        metadata,
        row_indices=row_indices_keep,
    )


COLE_2017_REPO_URL = 'https://github.com/voytekresearch/Cole_2017.git'
COLE_2017_DATA_URL = 'https://github.com/voytekresearch/Cole_2017/raw/master/data.mat'

BCICIII_ECOG_TRAIN_URL = 'https://www.bbci.de/competition/download/competition_iii/tuebingen/Competition_train.mat.gz'
BCICIII_ECOG_DESC_URL = 'https://www.bbci.de/competition/iii/desc_I.html'
BCICIII_ECOG_SELECTED_CHANNELS = (52, 30, 51, 29, 44)  # zero-indexed channels with strongest local 16-30 Hz residual peaks
BCICIII_ECOG_SCORE_BAND = (16, 30)
BCICIII_ECOG_SIGNAL_QC_MIN_ROWS = 2000
BCICIII_ECOG_MIN_ANALYSIS_ROWS = 1000


def ensure_bciciii_ecog_train():
    return download_to_cache(
        BCICIII_ECOG_TRAIN_URL,
        DATA_ROOT / 'bci_competition_iii_i' / 'Competition_train.mat.gz',
    )


def load_human_bciciii_ecog_motor_beta(
    selected_channels=BCICIII_ECOG_SELECTED_CHANNELS,
    max_trials=None,
    max_windows_per_trial=None,
    signal_qc_min_rows=BCICIII_ECOG_SIGNAL_QC_MIN_ROWS,
    min_analysis_rows=BCICIII_ECOG_MIN_ANALYSIS_ROWS,
):
    train_path = ensure_bciciii_ecog_train()
    with gzip.open(train_path, 'rb') as f:
        mat = loadmat(f)

    trials = np.asarray(mat['X'], dtype=np.float32)
    labels = np.asarray(mat.get('Y', np.full(trials.shape[0], np.nan))).squeeze()
    if trials.ndim != 3:
        raise ValueError(f'Expected BCI III X as trials x channels x samples, got {trials.shape}.')
    fs = 1000.0
    n_trials_total, n_channels_total, n_samples = trials.shape
    if max_trials is not None:
        n_trials = min(int(max_trials), n_trials_total)
        trials = trials[:n_trials]
        labels = labels[:n_trials]
    else:
        n_trials = n_trials_total

    selected_channels = tuple(int(ch) for ch in selected_channels)
    if not selected_channels:
        raise ValueError('selected_channels cannot be empty.')
    if min(selected_channels) < 0 or max(selected_channels) >= n_channels_total:
        raise ValueError(f'selected_channels must be zero-indexed channel ids in [0, {n_channels_total - 1}].')

    win_samples = int(round(WINDOW_S * fs))
    n_windows_per_trial = n_samples // win_samples
    if max_windows_per_trial is not None:
        n_windows_per_trial = min(n_windows_per_trial, int(max_windows_per_trial))
    if n_windows_per_trial < 1:
        raise RuntimeError('BCI III trials are too short for the requested WINDOW_S.')

    n_selected_channels = len(selected_channels)
    data = trials[:, selected_channels, : n_windows_per_trial * win_samples]
    # Rows are trial x within-trial window x local channel; this keeps all rows from
    # the same motor-cortex grid and preserves a large matrix for eigenspectrum fitting.
    data = data.reshape(n_trials, n_selected_channels, n_windows_per_trial, win_samples)
    data = data.transpose(0, 2, 1, 3)
    X_raw = data.reshape(-1, win_samples)
    row_indices_raw = np.arange(X_raw.shape[0], dtype=int)

    X, row_indices, row_qc = prepare_rows(
        X_raw,
        fs,
        normalize='rows',
        row_indices=row_indices_raw,
        osc_band=BCICIII_ECOG_SCORE_BAND,
        signal_qc_min_rows=signal_qc_min_rows,
    )
    if X.shape[0] < int(min_analysis_rows):
        raise RuntimeError(
            f'BCI III ECoG matrix has only {X.shape[0]} rows after QC; expected at least {min_analysis_rows}. '
            'Use more channels/windows or relax signal_qc_min_rows.'
        )

    label_values, label_counts = np.unique(labels[~pd.isna(labels)], return_counts=True)
    label_counts_available = {str(int(label)): int(count) for label, count in zip(label_values, label_counts)}
    available_rows_all_channels = int(n_trials * n_windows_per_trial * n_channels_total)

    metadata = {
        'species': 'human',
        'subject': 'BCI Competition III Dataset I participant',
        'animal_or_subject_count': 1,
        'modality': 'ECoG',
        'brain_region': 'contralateral right motor cortex 8x8 grid',
        'electrodes_or_channels': int(n_selected_channels),
        'available_electrodes_or_channels': int(n_channels_total),
        'selected_channel_indices': selected_channels,
        'selected_channel_indices_are_zero_based': True,
        'trials_or_windows': int(n_trials * n_windows_per_trial),
        'available_trials_or_windows': int(n_trials_total * (n_samples // win_samples)),
        'available_matrix_rows_all_channels': available_rows_all_channels,
        'matrix_rows_before_qc': int(X_raw.shape[0]),
        'minimum_analysis_rows_required': int(min_analysis_rows),
        'stimulus_or_state': 'motor imagery: left small finger vs tongue cues',
        'label_counts_available': label_counts_available,
        'software_filter': 'none applied in notebook; competition file is distributed as 1000 Hz ECoG trial epochs',
        'normalization': 'row z-score after row QC',
        'row_definition': f'trial x non-overlapping {WINDOW_S:g} s within-trial window x selected local ECoG channel',
        'dataset_note': 'Common BCI Competition III Dataset I motor-imagery benchmark. Channel subset is selected by condition-blind 16-30 Hz spectral prominence to keep a local, oscillatory, >1000-row matrix.',
        'periodic_score_band_hz': BCICIII_ECOG_SCORE_BAND,
        'citation': 'Lal et al., Methods Towards Invasive Human Brain Computer Interfaces, NIPS 2004',
        'source_url': BCICIII_ECOG_DESC_URL,
        'download_url': BCICIII_ECOG_TRAIN_URL,
        'cache_file': Path(train_path).name,
    }
    metadata.update(row_qc)

    return make_dataset(
        'Human ECoG: BCI III motor mu/beta',
        X,
        fs,
        'BCI Competition III Dataset I / University of Tuebingen',
        metadata,
        row_indices=row_indices,
    )


PARDO_VALENCIA_RECORD_URL = 'https://zenodo.org/records/10078352'
PARDO_VALENCIA_DATASET_DOI = '10.5281/zenodo.10078352'
PARDO_VALENCIA_PAPER_DOI = '10.1113/JP284768'
PARDO_VALENCIA_DATASET_URLS = {
    1: 'https://zenodo.org/api/records/10078352/files/dataset_1.zip/content',
    2: 'https://zenodo.org/api/records/10078352/files/dataset_2.zip/content',
}
PARDO_VALENCIA_FS_BY_PATIENT = {
    1: 5000, 2: 5000, 3: 5000, 4: 5000, 5: 10417, 6: 10417, 7: 10204,
    8: 10417, 9: 10417, 10: 10417, 11: 10417, 12: 10417, 13: 10417,
    14: 10417, 15: 10417, 16: 10417, 17: 10417, 18: 8333, 19: 10417,
    20: 5000, 21: 10417,
}
PARDO_VALENCIA_BETA_BAND = (13, 35)
PARDO_VALENCIA_SURROUND_BANDS = ((6, 12), (36, 55))
PARDO_VALENCIA_TARGET_FS = 1000.0
PARDO_VALENCIA_NARROW_TRACE = (1, 'P1_1_OFF_MASTN.mat')
PARDO_VALENCIA_WINDOW_SECONDS = 2.0
PARDO_VALENCIA_STEP_SECONDS = 0.075


def ensure_pardo_valencia_zip(dataset_index):
    dataset_index = int(dataset_index)
    return download_to_cache(
        PARDO_VALENCIA_DATASET_URLS[dataset_index],
        DATA_ROOT / 'pardo_valencia_stn_beta' / f'dataset_{dataset_index}.zip',
    )


def pardo_valencia_row_beta_ratios(X, fs=PARDO_VALENCIA_TARGET_FS):
    X = np.asarray(X, dtype=np.float32)
    X = X - X.mean(axis=1, keepdims=True)
    taper = np.hanning(X.shape[1]).astype(np.float32)
    freqs = np.fft.rfftfreq(X.shape[1], d=1 / float(fs))
    powers = np.abs(np.fft.rfft(X * taper, axis=1)) ** 2
    beta_mask = (freqs >= PARDO_VALENCIA_BETA_BAND[0]) & (freqs <= PARDO_VALENCIA_BETA_BAND[1])
    surround_mask = np.zeros_like(freqs, dtype=bool)
    for low, high in PARDO_VALENCIA_SURROUND_BANDS:
        surround_mask |= (freqs >= low) & (freqs <= high)
    return powers[:, beta_mask].mean(axis=1) / (powers[:, surround_mask].mean(axis=1) + 1e-12)


def pardo_valencia_beta_metrics(X, fs=PARDO_VALENCIA_TARGET_FS):
    X = np.asarray(X, dtype=np.float32)
    X = X - X.mean(axis=1, keepdims=True)
    taper = np.hanning(X.shape[1]).astype(np.float32)
    freqs = np.fft.rfftfreq(X.shape[1], d=1 / float(fs))
    powers = np.abs(np.fft.rfft(X * taper, axis=1)) ** 2
    mean_power = powers.mean(axis=0)
    beta_mask = (freqs >= PARDO_VALENCIA_BETA_BAND[0]) & (freqs <= PARDO_VALENCIA_BETA_BAND[1])
    surround_mask = np.zeros_like(freqs, dtype=bool)
    for low, high in PARDO_VALENCIA_SURROUND_BANDS:
        surround_mask |= (freqs >= low) & (freqs <= high)
    peak_idx = int(np.argmax(np.where(beta_mask, mean_power, -np.inf)))
    return {
        'beta_broad_mean_ratio': float(mean_power[beta_mask].mean() / (mean_power[surround_mask].mean() + 1e-12)),
        'beta_peak_freq_hz': float(freqs[peak_idx]),
        'beta_peak_surround_ratio': float(mean_power[peak_idx] / (mean_power[surround_mask].mean() + 1e-12)),
    }


def load_human_pardo_valencia_stn_beta(
    dataset_index=PARDO_VALENCIA_NARROW_TRACE[0],
    filename=PARDO_VALENCIA_NARROW_TRACE[1],
    target_fs=PARDO_VALENCIA_TARGET_FS,
    window_seconds=PARDO_VALENCIA_WINDOW_SECONDS,
    step_seconds=PARDO_VALENCIA_STEP_SECONDS,
    min_analysis_rows=1000,
):
    """Load a narrow, high-power STN beta example from Pardo-Valencia.

    The previous pooled version mixed patients/traces whose beta peaks occurred
    at different frequencies. That is useful for a population summary, but it
    smears the eigenspectrum. For this figure, use one high-beta trace and many
    overlapping windows so the example visibly preserves a narrow beta peak.
    """

    target_fs = float(target_fs)
    zip_path = ensure_pardo_valencia_zip(dataset_index)
    filename = str(filename)
    match = re.match(r'P(\d+)_(\d+)_(ON|OFF)_([A-Z]+STN)\.mat$', filename)
    if not match:
        raise ValueError(f'Unexpected Pardo-Valencia filename: {filename!r}.')
    patient = int(match.group(1))
    medication = match.group(3)
    stn = match.group(4)
    source_fs = float(PARDO_VALENCIA_FS_BY_PATIENT[patient])

    with zipfile.ZipFile(zip_path) as zf:
        members = {Path(member).name: member for member in zf.namelist() if member.lower().endswith('.mat')}
        if filename not in members:
            raise FileNotFoundError(f'{filename!r} not found in {zip_path}.')
        variable = Path(filename).stem
        mat = loadmat(BytesIO(zf.read(members[filename])), squeeze_me=False, struct_as_record=False)
        if variable not in mat:
            candidates = [key for key, value in mat.items() if not key.startswith('__') and np.asarray(value).size > 1000]
            if not candidates:
                raise RuntimeError(f'No usable numeric vector found in {filename}.')
            variable = candidates[0]
        trace = np.squeeze(np.asarray(mat[variable], dtype=np.float32))

    trace = trace[np.isfinite(trace)]
    if source_fs != target_fs:
        source_int = int(round(source_fs))
        target_int = int(round(target_fs))
        factor = gcd(source_int, target_int)
        trace = resample_poly(trace, target_int // factor, source_int // factor).astype(np.float32)

    win_samples = int(round(float(window_seconds) * target_fs))
    step_samples = int(round(float(step_seconds) * target_fs))
    if win_samples <= 0 or step_samples <= 0:
        raise ValueError('window_seconds and step_seconds must be positive.')
    starts = np.arange(0, trace.size - win_samples + 1, step_samples, dtype=int)
    if len(starts) < int(min_analysis_rows):
        raise RuntimeError(
            f'Only {len(starts)} Pardo-Valencia windows are available; expected at least {min_analysis_rows}. '
            'Decrease step_seconds or choose a longer trace.'
        )

    X_raw = np.vstack([trace[start:start + win_samples] for start in starts]).astype(np.float32, copy=False)
    row_beta_ratios = pardo_valencia_row_beta_ratios(X_raw, fs=target_fs)
    metrics = pardo_valencia_beta_metrics(X_raw, fs=target_fs)
    row_indices_raw = np.arange(X_raw.shape[0], dtype=int)

    X, row_indices, row_qc = prepare_rows(
        X_raw,
        target_fs,
        normalize='rows',
        row_indices=row_indices_raw,
        qc=False,
        osc_band=PARDO_VALENCIA_BETA_BAND,
    )
    if X.shape[0] < int(min_analysis_rows):
        raise RuntimeError(f'Pardo-Valencia narrow beta matrix has only {X.shape[0]} rows; expected at least {min_analysis_rows}.')

    kept_beta_ratios = row_beta_ratios[row_indices]
    line_metrics = pardo_valencia_beta_metrics(X_raw, fs=target_fs)
    metadata = {
        'species': 'human',
        'subject': f'P{patient}',
        'animal_or_subject_count': 1,
        'modality': 'STN LFP',
        'brain_region': 'subthalamic nucleus',
        'electrodes_or_channels': 1,
        'selected_trace_count': 1,
        'available_trace_count': 55,
        'selected_trace_file': filename,
        'selected_patient': int(patient),
        'selected_medication_state': medication,
        'selected_stn_side': stn,
        'trials_or_windows': int(X.shape[0]),
        'available_trials_or_windows': int(len(starts)),
        'window_seconds': float(window_seconds),
        'step_seconds': float(step_seconds),
        'overlapping_windows': bool(step_samples < win_samples),
        'source_duration_s': float(trace.size / target_fs),
        'row_beta_ratio_min_selected': float(np.min(kept_beta_ratios)),
        'row_beta_ratio_median_selected': float(np.median(kept_beta_ratios)),
        'row_beta_ratio_max_selected': float(np.max(kept_beta_ratios)),
        'beta_broad_mean_ratio': metrics['beta_broad_mean_ratio'],
        'beta_peak_freq_hz': metrics['beta_peak_freq_hz'],
        'beta_peak_surround_ratio': metrics['beta_peak_surround_ratio'],
        'target_fs_hz': target_fs,
        'source_fs_hz': source_fs,
        'stimulus_or_state': f'Parkinson STN LFP, {medication} medication, {stn}, narrow 17 Hz beta trace',
        'software_filter': 'resampled to 1000 Hz with scipy.signal.resample_poly; no bandpass/notch filtering applied in this notebook',
        'normalization': 'row z-score after basic finite/mean/std QC',
        'row_definition': f'overlapping {window_seconds:g} s windows from one continuous STN LFP trace, step={step_seconds:g} s',
        'dataset_note': 'Single-trace mode is used to preserve the narrow beta peak. Pooling across patients/traces smears peaks with different center frequencies.',
        'periodic_score_band_hz': PARDO_VALENCIA_BETA_BAND,
        'doi': PARDO_VALENCIA_PAPER_DOI,
        'dataset_doi': PARDO_VALENCIA_DATASET_DOI,
        'source_url': PARDO_VALENCIA_RECORD_URL,
        'downloaded_archives': (f'dataset_{int(dataset_index)}.zip',),
    }
    metadata.update(row_qc)

    return make_dataset(
        'Human STN LFP: Pardo-Valencia Parkinson beta',
        X,
        target_fs,
        'Pardo-Valencia et al. 2024 / Zenodo',
        metadata,
        row_indices=row_indices,
    )


MILLER_FINGERFLEX_URL = 'https://stacks.stanford.edu/file/druid:zk881ps0522/fingerflex.zip'
MILLER_FINGERFLEX_SUBJECT_CODES = ('bp', 'cc', 'ht', 'jc', 'jp', 'mv', 'wc', 'wm', 'zt')
MILLER_FINGER_LABELS = {1: 'thumb', 2: 'index', 3: 'middle', 4: 'ring', 5: 'little'}
# Selected after scanning all nine full-archive subjects in scripts/search_miller_fingerflex_beta.py.
# The subset below maximizes pooled rows while keeping a clear 12-20 Hz motor-beta
# peak and acceptable pooled circulant structure for this figure.
MILLER_FINGERFLEX_CLEAN_BETA_CONFIGS = (
    {'subject': 'ht', 'electrode_region': 1},   # 151 events x 3 channels; 17-18 Hz local beta
    {'subject': 'jc', 'selected_channels': (46, 45, 44)},  # contiguous high-beta channels; 132 events x 3 channels
    {'subject': 'jp', 'electrode_region': 1},   # 118 events x 7 channels; 15-16 Hz local beta
    {'subject': 'wc', 'electrode_region': 3},   # 154 events x 4 channels; 14-17 Hz local beta
    {'subject': 'zt', 'electrode_region': 1},   # 149 events x 4 channels; 17-18 Hz local beta
)
MILLER_FINGERFLEX_DROPPED_BY_DEFAULT = ('bp', 'cc', 'mv', 'wm')
MILLER_FINGERFLEX_DROPPED_REASONS = {
    'bp': 'strong local beta, but adding it to the maximum-row pooled subset dropped pooled kappa below 0.9',
    'cc': 'weak local beta under the common stim-offset movement-window definition',
    'mv': 'too few usable movement epochs and weak pooled beta under the common definition',
    'wm': 'strong local beta in some channels, but too nonstationary for the pooled matrix',
}


def ensure_cole_2017_repo(repo_dir=DATA_ROOT / 'Cole_2017'):
    repo_dir = Path(repo_dir)
    data_path = repo_dir / 'data.mat'
    if data_path.exists():
        return repo_dir
    download_to_cache(COLE_2017_DATA_URL, data_path)
    if data_path.exists():
        return repo_dir
    raise FileNotFoundError(
        f'Cole_2017 data.mat not found at {data_path}. '
        f'Clone {COLE_2017_REPO_URL} into {repo_dir}.'
    )


def load_human_cole_m1_beta(
    condition='B',
    max_windows=None,
):
    repo_dir = ensure_cole_2017_repo()
    data = loadmat(repo_dir / 'data.mat', squeeze_me=True, struct_as_record=False)
    if condition not in data:
        raise ValueError(f'Cole_2017 condition must be one of B or D, got {condition!r}.')
    traces = [np.asarray(trace, dtype=np.float32).squeeze() for trace in np.ravel(data[condition])]
    data_matrix = np.vstack(traces)
    fs = 1000.0
    X_raw, n_windows = continuous_to_windows(
        data_matrix,
        fs,
        window_s=WINDOW_S,
        max_windows=max_windows,
        normalize=None,
    )
    X, row_indices, row_qc = prepare_rows(X_raw, fs, normalize='rows', osc_band=(13, 30))
    condition_label = 'pre-DBS / untreated Parkinson M1 ECoG' if condition == 'B' else 'on-DBS Parkinson M1 ECoG'

    metadata = {
        'species': 'human',
        'subject': '23 Parkinson disease participants',
        'animal_or_subject_count': int(len(traces)),
        'modality': 'primary motor cortex ECoG',
        'brain_region': 'M1',
        'electrodes_or_channels': int(len(traces)),
        'available_electrodes_or_channels': int(len(traces)),
        'selected_condition': condition,
        'trials_or_windows': int(n_windows),
        'available_trials_or_windows': int(data_matrix.shape[1] // int(round(WINDOW_S * fs))),
        'stimulus_or_state': condition_label,
        'software_filter': 'none applied in notebook; Cole_2017 data.mat is distributed after 200 Hz low-pass and subject-specific high-frequency notch preprocessing',
        'normalization': 'row z-score after row QC',
        'row_definition': 'window x participant M1 ECoG trace; row = window_index * n_subjects + subject_index',
        'dataset_note': 'Small known-positive human M1 ECoG beta benchmark from nonsinusoidal Parkinson beta waveform-shape analysis.',
        'periodic_score_band_hz': (13, 30),
        'doi': '10.1523/JNEUROSCI.2208-16.2017',
        'source_url': COLE_2017_REPO_URL,
        'cache_file': str(repo_dir / 'data.mat'),
    }
    metadata.update(row_qc)

    return make_dataset(
        'Human M1 ECoG: Cole 2017 Parkinson beta',
        X,
        fs,
        'Cole_2017 GitHub / Journal of Neuroscience 2017',
        metadata,
        row_indices=row_indices,
    )


def extract_zip_to_cache(zip_path, out_dir):
    out_dir = Path(out_dir)
    if not out_dir.exists() or not any(out_dir.rglob('*')):
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
    return out_dir


def ensure_miller_fingerflex_zip():
    return download_to_cache(
        MILLER_FINGERFLEX_URL,
        DATA_ROOT / 'miller_fingerflex' / 'fingerflex.zip',
    )


def resolve_miller_subject_code(subject):
    if isinstance(subject, str):
        code = subject.lower().strip()
        if code.startswith('subject '):
            code = code.split()[-1]
        if code.isdigit():
            subject = int(code)
        else:
            if code not in MILLER_FINGERFLEX_SUBJECT_CODES:
                raise ValueError(
                    f'Miller/FingerFlex subject must be one of {MILLER_FINGERFLEX_SUBJECT_CODES}, got {subject!r}.'
                )
            return code
    if isinstance(subject, (int, np.integer)):
        if not 1 <= int(subject) <= len(MILLER_FINGERFLEX_SUBJECT_CODES):
            raise ValueError(
                f'Numeric Miller/FingerFlex subject must be 1-{len(MILLER_FINGERFLEX_SUBJECT_CODES)}, got {subject!r}.'
            )
        return MILLER_FINGERFLEX_SUBJECT_CODES[int(subject) - 1]
    raise TypeError('Miller/FingerFlex subject must be a two-letter code or a 1-indexed integer.')


def miller_zip_member(zip_file, suffix):
    suffix = suffix.replace('\\', '/')
    matches = [
        name for name in zip_file.namelist()
        if name.endswith(suffix) and not name.startswith('__MACOSX/') and '/._' not in name
    ]
    if not matches:
        raise FileNotFoundError(f'No {suffix!r} member found in {zip_file.filename}.')
    if len(matches) > 1:
        raise RuntimeError(f'Ambiguous Miller archive member for {suffix!r}: {matches}')
    return matches[0]


def load_miller_mat_from_zip(zip_path, suffix):
    with zipfile.ZipFile(zip_path) as zf:
        member = miller_zip_member(zf, suffix)
        return loadmat(BytesIO(zf.read(member)), squeeze_me=False, struct_as_record=False), member


def miller_labeled_epochs(labels, fs=1000.0, event_codes=(1, 2, 3, 4, 5), min_event_duration_s=0.25):
    labels = np.asarray(labels).squeeze()
    if labels.ndim != 1:
        raise ValueError('Miller event labels must be a one-dimensional sample-aligned vector.')
    event_codes = tuple(int(code) for code in event_codes)
    min_event_samples = max(1, int(round(min_event_duration_s * fs)))
    change_points = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]

    events = []
    for start, stop in zip(change_points[:-1], change_points[1:]):
        code = int(labels[start])
        if code not in event_codes:
            continue
        if stop - start < min_event_samples:
            continue
        events.append({'start': int(start), 'stop': int(stop), 'code': code, 'duration_samples': int(stop - start)})
    return events


def miller_event_windows(
    data,
    labels,
    fs=1000.0,
    event_codes=(1, 2, 3, 4, 5),
    event_anchor='offset',
    window_start_s=0.0,
    window_s=WINDOW_S,
    max_events=None,
    max_events_per_code=None,
    min_event_duration_s=0.25,
    require_window_within_event=False,
):
    events = miller_labeled_epochs(
        labels,
        fs=fs,
        event_codes=event_codes,
        min_event_duration_s=min_event_duration_s,
    )
    if event_anchor not in ('onset', 'offset'):
        raise ValueError("event_anchor must be 'onset' or 'offset'.")

    n_samples = int(round(window_s * fs))
    start_offset = int(round(window_start_s * fs))
    blocks = []
    row_indices = []
    kept_events = []
    per_code_counts = {int(code): 0 for code in event_codes}
    n_channels = int(data.shape[0])

    for event_i, event in enumerate(events):
        code = int(event['code'])
        if max_events_per_code is not None and per_code_counts[code] >= int(max_events_per_code):
            continue
        anchor_sample = event['start'] if event_anchor == 'onset' else event['stop']
        start = int(anchor_sample + start_offset)
        stop = start + n_samples
        if start < 0 or stop > data.shape[1]:
            continue
        if require_window_within_event and (start < event['start'] or stop > event['stop']):
            continue
        blocks.append(data[:, start:stop])
        kept_events.append({**event, 'event_index': int(event_i), 'window_start': int(start), 'window_stop': int(stop)})
        row_indices.extend(event_i * n_channels + np.arange(n_channels, dtype=int))
        per_code_counts[code] += 1
        if max_events is not None and len(kept_events) >= int(max_events):
            break

    if not blocks:
        raise RuntimeError('No complete Miller cue/stim-aligned windows could be extracted.')
    return np.vstack(blocks), np.asarray(row_indices, dtype=int), kept_events, events


def load_human_miller_fingerflex_beta(
    subject='clean_beta',
    max_channels=None,
    selected_channels=None,
    electrode_region=None,
    max_windows=None,
    event_source='stim',
    event_codes=(1, 2, 3, 4, 5),
    event_anchor='offset',
    event_window_start_s=0.0,
    max_events_per_code=None,
    min_event_duration_s=0.25,
    require_window_within_event=False,
):
    if subject in ('clean_beta', 'pooled_clean_beta'):
        datasets = []
        for config in MILLER_FINGERFLEX_CLEAN_BETA_CONFIGS:
            datasets.append(
                load_human_miller_fingerflex_beta(
                    subject=config['subject'],
                    max_channels=max_channels,
                    selected_channels=config.get('selected_channels'),
                    electrode_region=config.get('electrode_region'),
                    max_windows=max_windows,
                    event_source=event_source,
                    event_codes=event_codes,
                    event_anchor=event_anchor,
                    event_window_start_s=event_window_start_s,
                    max_events_per_code=max_events_per_code,
                    min_event_duration_s=min_event_duration_s,
                    require_window_within_event=require_window_within_event,
                )
            )
        fs_values = {dataset['fs'] for dataset in datasets}
        if len(fs_values) != 1:
            raise RuntimeError(f'Mixed Miller sampling rates: {sorted(fs_values)}')
        X = np.vstack([dataset['X'] for dataset in datasets]).astype(np.float32, copy=False)
        per_subject = [dict(dataset['metadata']) for dataset in datasets]
        metadata = {
            'species': 'human',
            'subject': ','.join(dataset['metadata']['subject'] for dataset in datasets),
            'animal_or_subject_count': int(len(datasets)),
            'modality': 'ECoG',
            'brain_region': 'pooled subject-specific clean motor-beta ECoG channel groups',
            'electrodes_or_channels': int(sum(dataset['metadata']['electrodes_or_channels'] for dataset in datasets)),
            'available_electrodes_or_channels': int(sum(dataset['metadata']['available_electrodes_or_channels'] for dataset in datasets)),
            'trials_or_windows': int(sum(dataset['metadata']['trials_or_windows'] for dataset in datasets)),
            'available_trials_or_windows': int(sum(dataset['metadata']['available_trials_or_windows'] for dataset in datasets)),
            'stimulus_or_state': f'pooled clean 12-20 Hz motor-beta ECoG, {event_source} {event_anchor}-aligned movement epochs',
            'software_filter': 'none applied in notebook; full Miller archive data are distributed at 1000 Hz',
            'normalization': 'row z-score after per-subject row QC',
            'row_definition': 'subject-specific stim movement-offset event x selected ECoG channel; rows pooled across clean-beta subjects',
            'event_source': event_source,
            'event_anchor': event_anchor,
            'event_window_start_s': float(event_window_start_s),
            'event_window_stop_s': float(event_window_start_s + WINDOW_S),
            'event_codes': tuple(int(code) for code in event_codes),
            'selected_subject_configs': tuple(dict(config) for config in MILLER_FINGERFLEX_CLEAN_BETA_CONFIGS),
            'dropped_subjects': MILLER_FINGERFLEX_DROPPED_BY_DEFAULT,
            'dropped_subject_reasons': dict(MILLER_FINGERFLEX_DROPPED_REASONS),
            'all_subjects_considered': MILLER_FINGERFLEX_SUBJECT_CODES,
            'per_subject_rows_after_qc': {dataset['metadata']['subject']: int(dataset['X'].shape[0]) for dataset in datasets},
            'per_subject_event_counts_used': {dataset['metadata']['subject']: dataset['metadata']['event_code_counts_used'] for dataset in datasets},
            'per_subject_selected_channels': {dataset['metadata']['subject']: dataset['metadata']['selected_channel_indices'] for dataset in datasets},
            'dataset_note': 'Full Kai Miller FingerFlex archive was scanned across all nine subjects. The default is pooled across the largest clean multi-subject 12-20 Hz motor-beta subset found so far under a common stim-offset movement-epoch definition. Dropped subjects and reasons are stored in dropped_subject_reasons.',
            'periodic_score_band_hz': (12, 20),
            'doi': '10.1038/s41562-019-0678-3',
            'source_url': MILLER_FINGERFLEX_URL,
            'cache_file': Path(ensure_miller_fingerflex_zip()).name,
            'row_qc_rows_before': int(sum(dataset['metadata'].get('row_qc_rows_before', dataset['X'].shape[0]) for dataset in datasets)),
            'row_qc_rows_after_mean_std': int(sum(dataset['metadata'].get('row_qc_rows_after_mean_std', dataset['X'].shape[0]) for dataset in datasets)),
            'row_qc_rows_after_stationary_oscillation': int(sum(dataset['metadata'].get('row_qc_rows_after_stationary_oscillation', dataset['X'].shape[0]) for dataset in datasets)),
            'row_qc_rows_after_spectral': int(X.shape[0]),
        }
        return make_dataset(
            'Human ECoG: Miller FingerFlex beta pooled',
            X,
            float(next(iter(fs_values))),
            'Kai Miller ECoG library / Stanford Digital Repository',
            metadata,
            row_indices=np.arange(X.shape[0], dtype=int),
        )
    if subject == 'all':
        datasets = [
            load_human_miller_fingerflex_beta(
                subject=code,
                max_channels=max_channels,
                selected_channels=selected_channels,
                electrode_region=electrode_region,
                max_windows=max_windows,
                event_source=event_source,
                event_codes=event_codes,
                event_anchor=event_anchor,
                event_window_start_s=event_window_start_s,
                max_events_per_code=max_events_per_code,
                min_event_duration_s=min_event_duration_s,
                require_window_within_event=require_window_within_event,
            )
            for code in MILLER_FINGERFLEX_SUBJECT_CODES
        ]
        X = np.vstack([dataset['X'] for dataset in datasets]).astype(np.float32, copy=False)
        metadata = dict(datasets[0]['metadata'])
        metadata.update({
            'subject': ','.join(dataset['metadata']['subject'] for dataset in datasets),
            'animal_or_subject_count': int(len(datasets)),
            'electrodes_or_channels': int(sum(dataset['metadata']['electrodes_or_channels'] for dataset in datasets)),
            'trials_or_windows': int(sum(dataset['metadata']['trials_or_windows'] for dataset in datasets)),
            'stimulus_or_state': f'all Miller subjects, {event_source} {event_anchor}-aligned movement epochs',
            'dataset_note': 'All Miller FingerFlex subjects pooled without beta-based participant exclusion.',
            'row_qc_rows_after_spectral': int(X.shape[0]),
        })
        return make_dataset(
            'Human ECoG: Miller FingerFlex beta all subjects',
            X,
            datasets[0]['fs'],
            'Kai Miller ECoG library / Stanford Digital Repository',
            metadata,
            row_indices=np.arange(X.shape[0], dtype=int),
        )

    subject_code = resolve_miller_subject_code(subject)
    if event_source not in ('stim', 'cue'):
        raise ValueError("event_source must be 'stim' or 'cue'.")

    zip_path = ensure_miller_fingerflex_zip()
    finger_mat, finger_member = load_miller_mat_from_zip(
        zip_path,
        f'fingerflex/data/{subject_code}/{subject_code}_fingerflex.mat',
    )
    stim_mat = None
    stim_member = None
    if event_source == 'stim':
        stim_mat, stim_member = load_miller_mat_from_zip(
            zip_path,
            f'fingerflex/data/{subject_code}/{subject_code}_stim.mat',
        )

    data = np.asarray(finger_mat['data'], dtype=np.float32).T
    n_available_channels = data.shape[0]
    elec_regions = None
    if 'elec_regions' in finger_mat:
        elec_regions = np.asarray(finger_mat['elec_regions']).astype(int).squeeze()
    channel_ids = np.arange(n_available_channels, dtype=int)
    if selected_channels is not None:
        keep_channels = np.asarray(selected_channels, dtype=int)
    elif electrode_region is not None and elec_regions is not None:
        keep_channels = np.where(elec_regions == int(electrode_region))[0]
        if keep_channels.size == 0:
            raise ValueError(f'No Miller channels found with electrode_region={electrode_region!r}.')
    else:
        keep_channels = channel_ids
    if max_channels is not None:
        keep_channels = keep_channels[:int(max_channels)]
    data = data[keep_channels]
    fs = 1000.0

    labels = np.asarray(stim_mat['stim'] if event_source == 'stim' else finger_mat['cue']).squeeze()
    if labels.shape[0] != data.shape[1]:
        raise ValueError(
            f'Miller {event_source!r} labels have {labels.shape[0]} samples but data have {data.shape[1]} samples.'
        )

    X_raw, row_indices_raw, kept_events, all_events = miller_event_windows(
        data,
        labels,
        fs=fs,
        event_codes=event_codes,
        event_anchor=event_anchor,
        window_start_s=event_window_start_s,
        window_s=WINDOW_S,
        max_events=max_windows,
        max_events_per_code=max_events_per_code,
        min_event_duration_s=min_event_duration_s,
        require_window_within_event=require_window_within_event,
    )

    X, row_indices, row_qc = prepare_rows(
        X_raw,
        fs,
        normalize='rows',
        row_indices=row_indices_raw,
        osc_band=(12, 20),
    )

    event_code_counts_available = {
        MILLER_FINGER_LABELS.get(int(code), str(code)): int(sum(event['code'] == int(code) for event in all_events))
        for code in event_codes
    }
    event_code_counts_used = {
        MILLER_FINGER_LABELS.get(int(code), str(code)): int(sum(event['code'] == int(code) for event in kept_events))
        for code in event_codes
    }
    event_samples_used = [int(event['start']) for event in kept_events]
    event_codes_used = [int(event['code']) for event in kept_events]

    metadata = {
        'species': 'human',
        'subject': subject_code,
        'animal_or_subject_count': 1,
        'modality': 'ECoG',
        'brain_region': 'subdural grid, selected electrode-region group from full Miller archive',
        'electrodes_or_channels': int(data.shape[0]),
        'available_electrodes_or_channels': int(n_available_channels),
        'selected_channel_indices': tuple(int(ch) for ch in keep_channels),
        'selected_electrode_region': None if electrode_region is None else int(electrode_region),
        'trials_or_windows': int(len(kept_events)),
        'available_trials_or_windows': int(len(all_events)),
        'stimulus_or_state': f'cued individual finger flexion ECoG, {event_source} {event_anchor}-aligned epochs',
        'software_filter': 'none applied in notebook; full Miller archive data are distributed at 1000 Hz',
        'normalization': 'row z-score after row QC',
        'row_definition': (
            f'{event_source} event x ECoG channel; row = event_index * n_channels + channel_index; '
            f'window is {event_window_start_s:g}-{event_window_start_s + WINDOW_S:g} s from {event_anchor}'
        ),
        'event_source': event_source,
        'event_anchor': event_anchor,
        'event_window_start_s': float(event_window_start_s),
        'event_window_stop_s': float(event_window_start_s + WINDOW_S),
        'event_codes': tuple(int(code) for code in event_codes),
        'event_code_counts_available': event_code_counts_available,
        'event_code_counts_used': event_code_counts_used,
        'event_samples_used_first10': event_samples_used[:10],
        'event_codes_used_first10': event_codes_used[:10],
        'max_events_per_code': None if max_events_per_code is None else int(max_events_per_code),
        'min_event_duration_s': float(min_event_duration_s),
        'require_window_within_event': bool(require_window_within_event),
        'dataset_note': 'Full Kai Miller FingerFlex archive. Unlike the BCI Competition derivative, this release includes sample-level cue/stim streams. Defaults use subject ht, stim movement-offset windows, and electrode region 3 because the beta search found a clear 17 Hz motor-beta peak there.',
        'periodic_score_band_hz': (12, 20),
        'doi': '10.1038/s41562-019-0678-3',
        'source_url': MILLER_FINGERFLEX_URL,
        'cache_file': Path(zip_path).name,
        'fingerflex_member': finger_member,
        'stim_member': stim_member,
    }
    if elec_regions is not None:
        metadata['electrode_region_codes'] = sorted(np.unique(elec_regions).astype(int).tolist())
        metadata['selected_electrode_region_codes'] = sorted(np.unique(elec_regions[keep_channels]).astype(int).tolist())
    metadata.update(row_qc)

    return make_dataset(
        f'Human ECoG: Miller FingerFlex beta {subject_code}',
        X,
        fs,
        'Kai Miller ECoG library / Stanford Digital Repository',
        metadata,
        row_indices=row_indices,
    )


def load_human_sleep_spindles(
    subject=0,
    recording=1,
    channel='EEG Fpz-Cz',
    stage='Sleep stage 2',
    max_windows=None,
):
    paths = sleep_physionet.age.fetch_data(
        subjects=[subject],
        recording=[recording],
        path=str(DATA_ROOT / 'mne'),
        on_missing='raise',
    )
    psg_path, hyp_path = paths[0]
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    raw.set_annotations(mne.read_annotations(hyp_path))
    if channel not in raw.ch_names:
        raise ValueError(f'Sleep EEG channel {channel!r} not found. Available: {raw.ch_names}')
    available_channels = list(raw.ch_names)
    raw.pick([channel])

    fs = raw.info['sfreq']
    win_samples = int(round(WINDOW_S * fs))
    rows = []
    row_indices = []
    for ann_i, ann in enumerate(raw.annotations):
        if ann['description'] != stage:
            continue
        start = max(0, int(round(ann['onset'] * fs)))
        duration = int(round(ann['duration'] * fs))
        n_windows = duration // win_samples
        for win_i in range(n_windows):
            if max_windows is not None and len(rows) >= max_windows:
                break
            row_start = start + win_i * win_samples
            row_stop = row_start + win_samples
            if row_stop <= raw.n_times:
                rows.append(raw.get_data(start=row_start, stop=row_stop)[0])
                row_indices.append(ann_i * max(n_windows, 1) + win_i)
        if max_windows is not None and len(rows) >= max_windows:
            break
    if not rows:
        raise RuntimeError(f'No complete {stage!r} windows found in sleep recording.')

    X_raw = np.vstack(rows)
    X, row_indices_keep, row_qc = prepare_rows(
        X_raw,
        fs,
        normalize='rows',
        row_indices=np.asarray(row_indices),
        osc_band=(11, 16),
    )

    metadata = {
        'species': 'human',
        'subject': f'SC4{subject:02d}',
        'animal_or_subject_count': 1,
        'modality': 'sleep EEG',
        'electrodes_or_channels': 1,
        'available_electrodes_or_channels': int(len(available_channels)),
        'selected_channel': channel,
        'trials_or_windows': int(X_raw.shape[0]),
        'stimulus_or_state': f'{stage} windows; sigma/spindle activity expected',
        'software_filter': 'none applied in notebook; raw PhysioNet EDF as loaded by MNE',
        'normalization': 'row z-score after row QC',
        'row_definition': f'{WINDOW_S:g} s windows from one EEG derivation during {stage}',
        'periodic_score_band_hz': (11, 16),
        'dataset_note': 'Stage-2 sleep is used because spindle/sigma activity is expected and not phase-locked to an external trigger.',
        'cache_file': Path(psg_path).name,
    }
    metadata.update(row_qc)

    return make_dataset(
        'Human EEG: PhysioNet N2 sleep spindles',
        X,
        fs,
        'MNE sleep_physionet / PhysioNet Sleep-EDF',
        metadata,
        row_indices=row_indices_keep,
    )

def load_human_epilepsy_ecog(
    max_windows=None,
    region_prefix='ILT',
):
    root = epilepsy_ecog.data_path(
        path=str(DATA_ROOT / 'mne'),
        update_path=False,
        download=True,
    )
    root = Path(root)
    vhdrs = sorted(root.rglob('*.vhdr'))
    if not vhdrs:
        raise FileNotFoundError(f'No BrainVision .vhdr files found under {root}')

    raw = mne.io.read_raw_brainvision(vhdrs[0], preload=True, verbose=False)
    ecog_picks = mne.pick_types(raw.info, ecog=True)
    if len(ecog_picks):
        raw.pick(ecog_picks)
    if region_prefix is not None:
        region_channels = [
            name for name in raw.ch_names
            if re.match(rf'^{re.escape(region_prefix)}\d+$', name)
        ]
        if not region_channels:
            raise ValueError(f'No ECoG channels found for prefix {region_prefix!r}.')
        raw.pick(region_channels)

    data = raw.get_data()
    X_raw, n_windows = continuous_to_windows(
        data,
        raw.info['sfreq'],
        window_s=WINDOW_S,
        max_windows=max_windows,
        normalize=None,
    )
    X, row_indices, row_qc = prepare_rows(X_raw, raw.info['sfreq'], normalize='rows', osc_band=(4, 30))

    metadata = {
        'species': 'human',
        'subject': 'sub-pt1',
        'animal_or_subject_count': 1,
        'modality': 'intracranial ECoG/iEEG',
        'electrodes_or_channels': int(len(raw.ch_names)),
        'selected_region': region_prefix,
        'selected_channels': ', '.join(raw.ch_names),
        'trials_or_windows': int(n_windows),
        'stimulus_or_state': f'ictal clinical recording, same-region ECoG channels, {WINDOW_S:g} s windows',
        'software_filter': 'none; raw BrainVision signal as loaded by MNE',
        'normalization': 'row z-score after row QC',
        'row_definition': 'window x same-region ECoG channel; row = window_index * n_channels + channel_index',
        'periodic_score_band_hz': (4, 30),
    }
    metadata.update(row_qc)

    return make_dataset(
        'Human ECoG: epilepsy example',
        X,
        raw.info['sfreq'],
        'MNE epilepsy_ecog',
        metadata,
        row_indices=row_indices,
    )


def parse_size_mb(size_text):
    value, unit = size_text.split()[:2]
    value = float(value)
    unit = unit.lower()
    if unit.startswith('gb'):
        return value * 1024
    if unit.startswith('kb'):
        return value / 1024
    return value


def cached_neurotycho_zip():
    out_dir = DATA_ROOT / 'neurotycho'
    zips = sorted(out_dir.glob('*.zip'))
    return zips[0] if zips else None


def neurotycho_eeg_ecog_entry():
    cached = cached_neurotycho_zip()
    if cached is not None:
        return {
            'Name': cached.stem,
            'Download': [{'filename': cached.name, 'size': 'cached'}],
            'cached_path': cached,
        }

    with urlopen('https://neurotycho.org/data/detail.json', timeout=60) as f:
        metadata = json.load(f)['expdata']

    hits = [item for item in metadata if item.get('Task') == 'EEGandECoG']
    if not hits:
        raise RuntimeError('No NeuroTycho EEGandECoG entries found.')

    hits = sorted(hits, key=lambda item: parse_size_mb(item['Download'][0]['size']))
    return hits[0]


def retrieve_neurotycho_zip(entry):
    if 'cached_path' in entry:
        print(f"Using cached {entry['cached_path']}")
        return entry['cached_path']

    url = entry['Download'][0]['filename']
    out_dir = DATA_ROOT / 'neurotycho'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(urlparse(url).path).name
    if out_path.exists():
        print(f'Using cached {out_path}')
    else:
        print(f'Downloading {url}')
        urlretrieve(url, out_path)
    return out_path


def load_neurotycho_hdf5_array(raw_bytes, kind, max_channels=None):
    with h5py.File(BytesIO(raw_bytes), 'r') as f:
        if 'WaveData' in f:
            dset = f['WaveData']
            if dset.shape[0] > dset.shape[1]:
                n_channels = min(max_channels or dset.shape[1], dset.shape[1])
                return np.asarray(dset[:, :n_channels]).T
            n_channels = min(max_channels or dset.shape[0], dset.shape[0])
            return np.asarray(dset[:n_channels, :])

        arrays = []
        for key in f.keys():
            if key.lower().startswith(f'{kind.lower()}data_ch'):
                arrays.append(np.asarray(f[key]).squeeze())
        if arrays:
            return np.vstack(arrays)

    raise FileNotFoundError(f'No {kind} data array found in HDF5 MAT file.')


def load_neurotycho_legacy_channels(zf, channel_files, kind):
    traces = []
    for name in channel_files:
        raw_bytes = zf.read(name)
        try:
            mat = loadmat(BytesIO(raw_bytes))
            arrays = [
                value for key, value in mat.items()
                if key.lower().startswith(f'{kind.lower()}data_ch')
            ]
            if not arrays:
                arrays = [
                    value for key, value in mat.items()
                    if not key.startswith('__') and np.asarray(value).ndim <= 2
                ]
            traces.append(np.asarray(arrays[0]).squeeze())
        except NotImplementedError:
            data = load_neurotycho_hdf5_array(raw_bytes, kind)
            if data.ndim == 1:
                traces.append(data)
            else:
                traces.extend(data)

    return np.vstack(traces)


def load_neurotycho_session_matrix(zf, kind, max_channels=None):
    mat_files = [
        name for name in zf.namelist()
        if name.lower().endswith('.mat') and not name.startswith('__MACOSX')
    ]
    candidates = [name for name in mat_files if kind.lower() in Path(name).name.lower()]
    if not candidates and kind.upper() == 'ECOG':
        candidates = mat_files

    for name in candidates:
        raw_bytes = zf.read(name)
        try:
            return load_neurotycho_hdf5_array(raw_bytes, kind, max_channels=max_channels)
        except OSError:
            mat = loadmat(BytesIO(raw_bytes))
            arrays = [
                value for key, value in mat.items()
                if not key.startswith('__') and np.asarray(value).ndim <= 2
            ]
            if arrays:
                data = np.asarray(arrays[0]).squeeze()
                return data if data.shape[0] < data.shape[-1] else data.T

    raise FileNotFoundError(f'No session-level {kind} matrix found in NeuroTycho zip.')


def load_neurotycho_signal(
    kind='ECoG',
    max_channels=128,
    max_windows=NEUROTYCHO_WINDOWS,
):
    entry = neurotycho_eeg_ecog_entry()
    zip_path = retrieve_neurotycho_zip(entry)
    pattern = re.compile(rf'(^|/){kind}_ch(\d+)\.mat$', flags=re.IGNORECASE)

    with zipfile.ZipFile(zip_path) as zf:
        channel_files = []
        for name in zf.namelist():
            match = pattern.search(name)
            if match:
                channel_files.append((int(match.group(2)), name))
        channel_files = [name for _, name in sorted(channel_files)[:max_channels]]

        if channel_files:
            data = load_neurotycho_legacy_channels(zf, channel_files, kind)
        else:
            if kind.upper() == 'EEG':
                preview = '\n'.join(zf.namelist()[:20])
                raise FileNotFoundError(
                    'This NeuroTycho zip does not expose explicit EEG channel files. '
                    f'Use kind="ECoG" for this cached session. Preview:\n{preview}'
                )
            data = load_neurotycho_session_matrix(zf, kind, max_channels=max_channels)

    data = np.asarray(data[:max_channels], dtype=np.float64)
    n_available_channels = data.shape[0]
    fs = 1000.0
    X_raw, n_windows = continuous_to_windows(
        data,
        fs,
        window_s=WINDOW_S,
        max_windows=max_windows,
        normalize=None,
    )
    raw_50hz_ratio = line_frequency_ratio(X_raw, fs, 50.0)
    raw_60hz_ratio = line_frequency_ratio(X_raw, fs, 60.0)
    X, row_indices, row_qc = prepare_rows(X_raw, fs, normalize='rows', osc_band=(6, 30))

    metadata = {
        'species': 'macaque',
        'subject': 'Su',
        'animal_or_subject_count': 1,
        'modality': f'{kind} (NeuroTycho WaveData)',
        'electrodes_or_channels': int(data.shape[0]),
        'available_electrodes_or_channels': int(n_available_channels),
        'trials_or_windows': int(n_windows),
        'stimulus_or_state': f'anesthesia/resting session, all loaded ECoG channels, {WINDOW_S:g} s windows',
        'software_filter': 'none; raw NeuroTycho WaveData, no notebook filtering or resampling',
        'normalization': 'row z-score after row QC',
        'row_definition': 'window x ECoG channel; row = window_index * n_channels + channel_index',
        'line_noise_mask_note': f'Line-noise harmonics {line_noise_bands(fs)} are excluded only from row scoring, QC summaries, and aperiodic fit masks; the signal and plotted eigenspectrum are not filtered.',
        'periodic_score_band_hz': (6, 30),
        'raw_50hz_power_over_neighbors': raw_50hz_ratio,
        'raw_60hz_power_over_neighbors': raw_60hz_ratio,
        'cache_file': zip_path.name,
    }
    metadata.update(row_qc)

    label = 'Macaque EEG' if kind.upper() == 'EEG' else 'Macaque ECoG'
    return make_dataset(
        f'{label}: NeuroTycho EEG/ECoG',
        X,
        fs,
        f"NeuroTycho {entry['Name']}",
        metadata,
        row_indices=row_indices,
    )


def load_rat_medial_septum_theta(
    channel=0,
    max_windows=None,
):
    asset_id = 'a2e0ec66-b963-4640-a785-ec82eb8ca3a8'
    nwb_path = download_to_cache(
        f'https://api.dandiarchive.org/api/assets/{asset_id}/download/',
        DATA_ROOT / 'dandi' / '001607' / 'sub-Banner_ses-Banner-20220123_ecephys.nwb',
    )
    with h5py.File(nwb_path, 'r') as f:
        data_dset = f['scratch/filtered data/data']
        timestamps = np.asarray(f['scratch/filtered data/timestamps'])
        if channel < 0 or channel >= data_dset.shape[1]:
            raise ValueError(f'Rat theta channel must be in [0, {data_dset.shape[1] - 1}].')
        data = np.asarray(data_dset[:, channel], dtype=np.float32)[None, :]
        fs = float(1 / np.nanmedian(np.diff(timestamps)))
        subject = f['general/subject/subject_id'][()].decode()
        species = f['general/subject/species'][()].decode()
        session_description = f['session_description'][()].decode()
        available_channels = int(data_dset.shape[1])

    X_raw, n_windows = continuous_to_windows(
        data,
        fs,
        window_s=WINDOW_S,
        max_windows=max_windows,
        normalize=None,
    )
    X, row_indices, row_qc = prepare_rows(X_raw, fs, normalize='rows', osc_band=(6, 12))

    metadata = {
        'species': species,
        'subject': subject,
        'animal_or_subject_count': 1,
        'modality': 'rat continuous ecephys',
        'electrodes_or_channels': 1,
        'available_electrodes_or_channels': available_channels,
        'selected_channel': int(channel),
        'trials_or_windows': int(n_windows),
        'stimulus_or_state': session_description,
        'software_filter': 'none applied in notebook; uses DANDI scratch/filtered data as distributed',
        'normalization': 'row z-score after row QC',
        'row_definition': f'{WINDOW_S:g} s windows from one continuous rat ecephys channel',
        'dataset_note': 'DANDI 001607 is a rat medial-septal-stimulation dataset with strong theta-timescale structure in the distributed continuous ecephys trace.',
        'periodic_score_band_hz': (6, 12),
        'cache_file': nwb_path.name,
        'dandiset': 'DANDI:001607',
    }
    metadata.update(row_qc)

    return make_dataset(
        'Rat ecephys: medial septal theta',
        X,
        fs,
        'DANDI 001607: Optogenetic disruption of theta-timescale spiking',
        metadata,
        row_indices=row_indices,
    )


def read_hc3_xml_metadata(xml_path):
    root = ET.parse(xml_path).getroot()
    n_channels = int(root.findtext('./acquisitionSystem/nChannels'))
    fs = float(root.findtext('./fieldPotentials/lfpSamplingRate'))
    channel_groups = []
    for group in root.findall('./anatomicalDescription/channelGroups/group'):
        channels = [
            int(channel.text)
            for channel in group.findall('channel')
            if channel.get('skip', '0') != '1'
        ]
        if channels:
            channel_groups.append(channels)
    return n_channels, fs, channel_groups


def load_hc3_hippocampal_eeg(
    eeg_path=Path('~/oscillatory_eigenvectors/data/hc-3/pin01-11-04/11-05_0-06-51/11-05_0-06-51.eeg'),
    channels='all',
    max_windows=None,
):
    eeg_path = Path(eeg_path).expanduser()
    if not eeg_path.exists():
        raise FileNotFoundError(f'HC-3 .eeg file not found: {eeg_path}')

    xml_path = eeg_path.with_suffix('.xml')
    if not xml_path.exists():
        raise FileNotFoundError(f'HC-3 XML metadata file not found: {xml_path}')

    n_channels, fs, channel_groups = read_hc3_xml_metadata(xml_path)
    if channels in (None, 'all'):
        selected_channels = np.arange(n_channels, dtype=int)
        channel_mode = 'all channels'
    elif channels == 'xml':
        selected_channels = np.unique(np.concatenate(channel_groups)).astype(int)
        channel_mode = 'XML anatomical channel groups'
    elif np.isscalar(channels):
        selected_channels = np.asarray([channels], dtype=int)
        channel_mode = 'explicit channel'
    else:
        selected_channels = np.asarray(channels, dtype=int)
        channel_mode = 'explicit channel list'

    if selected_channels.size == 0:
        raise ValueError('No HC-3 channels selected.')
    if np.any((selected_channels < 0) | (selected_channels >= n_channels)):
        raise ValueError(f'HC-3 channels must be in [0, {n_channels - 1}].')

    sample_count = eeg_path.stat().st_size // np.dtype('<i2').itemsize
    if sample_count % n_channels:
        raise ValueError('HC-3 .eeg file size is not divisible by the XML channel count.')
    n_samples = sample_count // n_channels
    n_win = int(round(WINDOW_S * fs))
    n_windows_available = n_samples // n_win
    n_windows = n_windows_available if max_windows is None else min(max_windows, n_windows_available)
    if n_windows < 1:
        raise ValueError('Not enough HC-3 samples for one analysis window.')

    raw = np.memmap(eeg_path, dtype='<i2', mode='r', shape=(n_samples, n_channels))
    sample_stop = n_windows * n_win
    data = np.asarray(raw[:sample_stop, :][:, selected_channels], dtype=np.float32).T
    X_raw, n_windows = continuous_to_windows(
        data,
        fs,
        window_s=WINDOW_S,
        max_windows=None,
        normalize=None,
    )
    X, row_indices, row_qc = prepare_rows(X_raw, fs, normalize='rows', osc_band=(6, 12))

    metadata = {
        'species': 'rat',
        'subject': 'pin01',
        'session_id': 'pin01-11-04/11-05_0-06-51',
        'animal_or_subject_count': 1,
        'modality': 'HC-3 hippocampal LFP (.eeg)',
        'brain_region': 'hippocampus CA1/CA3',
        'electrodes_or_channels': int(len(selected_channels)),
        'available_electrodes_or_channels': int(n_channels),
        'selected_channel_mode': channel_mode,
        'selected_channels': ', '.join(map(str, selected_channels.tolist())),
        'xml_anatomical_channel_groups': str(channel_groups),
        'trials_or_windows': int(n_windows),
        'available_trials_or_windows': int(n_windows_available),
        'stimulus_or_state': 'continuous hippocampal CA1/CA3 recording',
        'software_filter': 'none applied in notebook; uses HC-3 .eeg LFP binary as distributed',
        'normalization': 'row z-score after row QC',
        'row_definition': 'window x hippocampal channel; row = window_index * n_channels + channel_index',
        'dataset_note': 'Local HC-3 recording supplied by path; all channels are treated as hippocampal CA1/CA3 rhythms.',
        'periodic_score_band_hz': (6, 12),
        'cache_file': eeg_path.name,
        'data_path': str(eeg_path),
    }
    metadata.update(row_qc)

    return make_dataset(
        'Rat HC-3 hippocampus: CA1/CA3 LFP',
        X,
        fs,
        'HC-3 local hippocampal .eeg recording',
        metadata,
        row_indices=row_indices,
    )


def load_macaque_visual_grating_ecog(
    channel=19,
    max_windows=None,
):
    zip_path = download_to_cache(
        'http://neurotycho.brain.riken.jp/download/2016/20100805S1_Visual+Grating_K2_Kazuhito+Takenaka-Toru+Yanagawa_mat_ECoG128-Eye.zip',
        DATA_ROOT / 'neurotycho' / '20100805S1_Visual-Grating_K2_mat_ECoG128-Eye.zip',
    )
    with zipfile.ZipFile(zip_path) as zf:
        ecog_files = [
            name for name in zf.namelist()
            if name.endswith('ECoG-3.mat') and not name.startswith('__MACOSX')
        ]
        if not ecog_files:
            raise FileNotFoundError('No ECoG-3.mat file found in NeuroTycho visual grating archive.')
        mat = loadmat(BytesIO(zf.read(ecog_files[0])))
    data_all = np.asarray(mat['X'], dtype=np.float32)
    n_ecog_channels = min(128, data_all.shape[0])
    if channel < 0 or channel >= n_ecog_channels:
        raise ValueError(f'Macaque ECoG channel must be in [0, {n_ecog_channels - 1}].')
    data = data_all[channel:channel + 1]
    fs = 1000.0

    X_raw, n_windows = continuous_to_windows(
        data,
        fs,
        window_s=WINDOW_S,
        max_windows=max_windows,
        normalize=None,
    )
    X, row_indices, row_qc = prepare_rows(X_raw, fs, normalize='rows', osc_band=(12, 20))

    metadata = {
        'species': 'macaque',
        'subject': 'K2',
        'animal_or_subject_count': 1,
        'modality': 'ECoG',
        'electrodes_or_channels': 1,
        'available_electrodes_or_channels': int(n_ecog_channels),
        'selected_channel': int(channel),
        'trials_or_windows': int(n_windows),
        'stimulus_or_state': f'visual grating task, channel {channel}',
        'software_filter': 'none applied in notebook; raw NeuroTycho ECoG matrix as distributed',
        'normalization': 'row z-score after row QC',
        'row_definition': f'{WINDOW_S:g} s windows from one macaque ECoG channel during visual grating',
        'dataset_note': 'A single high-kappa ECoG channel is used because all 128 channels mix heterogeneous cortical regions and reduce circulant structure.',
        'periodic_score_band_hz': (12, 20),
        'cache_file': zip_path.name,
    }
    metadata.update(row_qc)

    return make_dataset(
        'Macaque ECoG: visual grating',
        X,
        fs,
        'NeuroTycho Visual Grating K2 2010-08-05',
        metadata,
        row_indices=row_indices,
    )

DATASET_REGISTRY = {
    'human_eegbci_alpha': load_human_eegbci,
    'human_lemon_alpha': load_human_lemon_alpha,
    'human_cole_m1_beta': load_human_cole_m1_beta,
    'human_pardo_valencia_stn_beta': load_human_pardo_valencia_stn_beta,
    'human_bciciii_ecog_motor_beta': load_human_bciciii_ecog_motor_beta,
    'human_miller_fingerflex_beta': load_human_miller_fingerflex_beta,
    'human_sleep_spindles': load_human_sleep_spindles,
    'human_ssvep_12hz': load_human_ssvep,
    'human_epilepsy_ecog': load_human_epilepsy_ecog,
    'mouse_neuropixels_natural_movie': load_mouse_neuropixels_natural_movie,
    'rat_hc3_ca1_ca3': load_hc3_hippocampal_eeg,
    'rat_medial_septum_theta': load_rat_medial_septum_theta,
    'macaque_visual_grating_ecog': load_macaque_visual_grating_ecog,
    'macaque_neurotycho_ecog': lambda: load_neurotycho_signal('ECoG'),
}

unknown_keys = [key for key in DATASET_KEYS if key not in DATASET_REGISTRY]
if unknown_keys:
    raise KeyError(f'Unknown dataset keys: {unknown_keys}. Available: {list(DATASET_REGISTRY)}')

dataset_loaders = [DATASET_REGISTRY[key] for key in DATASET_KEYS]

# Implementation defaults kept out of the main settings cell.
FIT_CHUNK_ROWS = 4096
SCORE_CHUNK_ROWS = 4096
PSD_AR_BOUNDS = (-0.95, 0.95)
PSD_FIRST_PHI_GUESS = 0.8
PSD_LOSS = 'huber'
PSD_F_SCALE = 0.15
PSD_MAXFEV = 50000
PSD_FIT_MIN_FREQ = 1.0
PSD_FIT_MAX_FREQ = None
PSD_FIT_RESID_Z = 2.5
PSD_FIT_ITERATIONS = 2
PSD_FALLBACK_ORDER = 5
N_CANDIDATE_ROWS = 1024
MAX_SCORE_ROWS = 50000
MIN_PERIODIC_PEAK_FRACTION = 0.2
MIN_SPECTRAL_PEAK_RATIO = 1.25
MAX_LINE_NOISE_FRACTION = 0.4
MIN_VARIANCE_BALANCE = 0.02
SUSTAINED_SEGMENT_S = 0.2
SUSTAINED_RMS_RATIO = 0.35
MIN_SUSTAINED_FRACTION = 0.6
KAPPA_SELECTION_QUANTILES = (1.0, 0.995, 0.99, 0.98, 0.95, 0.9, 0.85, 0.8, 0.7)
MIN_KAPPA_SELECTION_ROWS = 30


MIN_PERIODIC_PEAK_FRACTION = 0.8
MAX_LINE_NOISE_FRACTION = 0.1
MIN_VARIANCE_BALANCE = 0.02
MIN_SUSTAINED_FRACTION = 0.8
PLOT_ROW_MIN_RAW_PERIODIC_CORR = 0.8

# Dataset-specific ARPSD defaults. Edit CUSTOM_PSD_FIT_SETTINGS_BY_DATASET above this cell
# to tune individual datasets without changing the fitting code below.
# Full publication preset is available, but can be slow at AR(100).
# The suggested examples below use FAST_PUBLICATION_AR_FIT_SETTINGS by default.
PUBLICATION_AR_FIT_SETTINGS = {
    'use_robust_aperiodic_ar': True,
    'order': 100,
    'fallback_order': 50,
    'order_sequence': (100, 50, 25, 10),
    'fit_min_freq': 1e-9,
    'fit_max_freq': 550.0,
    'exclude_bands': (),
    'loss_fn': 'huber',
    'f_scale': 0.1,
    'maxfev': 50_000,
    'asymmetric_overfit_penalty': False,
    'overfit_weight': 0.0,
    'low_freq_residual_weight': 1.0,
    'low_freq_overfit_weight': 0.0,
    'low_freq_max': 5.0,
    'overfit_tolerance_log10': 0.1,
    'ceiling_to_data': False,
    'mask_peaks_for_fit': True,
    'peak_mask_min_freq': 5.0,
    'peak_mask_window': 5,
    'peak_mask_threshold': 0.75,
    'loglog_slope_change_weight': 0.0,
    'positive_loglog_slope_weight': 0.0,
    'positive_loglog_slope_tolerance': 0.0,
    'slope_penalty_grid_size': None,
}
FAST_PUBLICATION_AR_FIT_SETTINGS = {
    **PUBLICATION_AR_FIT_SETTINGS,
    'order': 10,
    'fallback_order': 5,
    'order_sequence': (10, 5, 3),
}

DEFAULT_PSD_FIT_SETTINGS_BY_DATASET = {
    'human_pardo_valencia_stn_beta': {
        **FAST_PUBLICATION_AR_FIT_SETTINGS,
        'exclude_bands': ((13.0, 35.0),),
        'fit_max_freq': 120.0,
        'order': 5,
        'fallback_order': 3,
        'order_sequence': (5, 3),
    },
    'human_bciciii_ecog_motor_beta': {
        **FAST_PUBLICATION_AR_FIT_SETTINGS,
        'exclude_bands': ((8.0, 30.0),),
        'fit_max_freq': 120.0,
        'order': 5,
        'fallback_order': 3,
        'order_sequence': (5, 3),
    },
    'human_miller_fingerflex_beta': {
        'exclude_bands': ((12.0, 20.0),),
        'fit_max_freq': 120.0,
        'order': 5,
        'fallback_order': 3,
    },
}
SUGGESTED_PSD_FIT_SETTINGS_BY_DATASET = {
    'macaque_visual_grating_ecog': {
        **FAST_PUBLICATION_AR_FIT_SETTINGS,
        'fit_max_freq': 120.0,
    },
    'Macaque ECoG: visual grating': {
        **FAST_PUBLICATION_AR_FIT_SETTINGS,
        'fit_max_freq': 120.0,
    },
}
CUSTOM_PSD_FIT_SETTINGS_BY_DATASET = {
    **SUGGESTED_PSD_FIT_SETTINGS_BY_DATASET,
    **_module_settings().get('CUSTOM_PSD_FIT_SETTINGS_BY_DATASET', {}),
}
PSD_FIT_SETTINGS_BY_DATASET = {
    **DEFAULT_PSD_FIT_SETTINGS_BY_DATASET,
    **CUSTOM_PSD_FIT_SETTINGS_BY_DATASET,
}


def default_psd_fit_settings():
    return {
        'order': PSD_ORDER,
        'fallback_order': PSD_FALLBACK_ORDER,
        'order_sequence': None,
        'ar_bounds': PSD_AR_BOUNDS,
        'power_bounds_scale': (1e-8, 1e8),
        'first_phi_guess': PSD_FIRST_PHI_GUESS,
        'guess_phis': None,
        'loss_fn': PSD_LOSS,
        'f_scale': PSD_F_SCALE,
        'maxfev': PSD_MAXFEV,
        # Optional scipy.optimize.curve_fit weighting. Lower sigma gives a
        # frequency bin more influence. With ARPSD's default log scaling,
        # sigma is interpreted in log10-power units.
        'sigma': None,
        'sigma_kwargs': {},
        'absolute_sigma': False,
        'curve_fit_kwargs': {},
        'fit_min_freq': PSD_FIT_MIN_FREQ,
        'fit_max_freq': PSD_FIT_MAX_FREQ,
        # If true, the fitted PSD used for inversion is replaced with the raw
        # eigenspectrum outside [fit_min_freq, fit_max_freq]. This avoids
        # inventing low/high-frequency structure around high-pass/low-pass edges.
        'fill_outside_fit_range_with_raw': False,
        # If None, follows fill_outside_fit_range_with_raw. If true, plots the
        # fit only inside [fit_min_freq, fit_max_freq].
        'plot_fit_range_only': None,
        'fit_resid_z': PSD_FIT_RESID_Z,
        'fit_iterations': PSD_FIT_ITERATIONS,
        'exclude_bands': (),
        'include_line_noise': False,
        'use_robust_aperiodic_ar': False,
        'asymmetric_overfit_penalty': False,
        'overfit_weight': 0.0,
        'low_freq_residual_weight': 1.0,
        'low_freq_overfit_weight': 0.0,
        'low_freq_max': 5.0,
        'overfit_tolerance_log10': 0.1,
        'ceiling_to_data': False,
        'mask_peaks_for_fit': False,
        'peak_mask_min_freq': None,
        'peak_mask_window': 5,
        'peak_mask_threshold': 0.75,
        'loglog_slope_change_weight': 0.0,
        'positive_loglog_slope_weight': 0.0,
        'positive_loglog_slope_tolerance': 0.0,
        'slope_penalty_grid_size': None,
    }


def resolve_psd_fit_settings(overrides=None):
    settings = default_psd_fit_settings()
    if overrides:
        settings.update(overrides)
    aliases = {
        'ar_order': 'order',
        'AR_ORDER': 'order',
        'AR_FIT_MIN_FREQ': 'fit_min_freq',
        'AR_FIT_MAX_FREQ': 'fit_max_freq',
        'AR_EXCLUDE_BANDS': 'exclude_bands',
        'AR_LOSS_FN': 'loss_fn',
        'AR_F_SCALE': 'f_scale',
        'AR_MAXFEV': 'maxfev',
        'AR_ASYMMETRIC_OVERFIT_PENALTY': 'asymmetric_overfit_penalty',
        'AR_OVERFIT_WEIGHT': 'overfit_weight',
        'AR_LOW_FREQ_RESIDUAL_WEIGHT': 'low_freq_residual_weight',
        'AR_LOW_FREQ_OVERFIT_WEIGHT': 'low_freq_overfit_weight',
        'AR_LOW_FREQ_MAX': 'low_freq_max',
        'AR_OVERFIT_TOLERANCE_LOG10': 'overfit_tolerance_log10',
        'AR_CEILING_TO_DATA': 'ceiling_to_data',
        'AR_MASK_PEAKS': 'mask_peaks_for_fit',
        'AR_PEAK_MASK_MIN_FREQ': 'peak_mask_min_freq',
        'AR_PEAK_MASK_WINDOW': 'peak_mask_window',
        'AR_PEAK_MASK_THRESHOLD': 'peak_mask_threshold',
        'AR_LOGLOG_SLOPE_CHANGE_WEIGHT': 'loglog_slope_change_weight',
        'AR_POSITIVE_LOGLOG_SLOPE_WEIGHT': 'positive_loglog_slope_weight',
        'AR_POSITIVE_LOGLOG_SLOPE_TOLERANCE': 'positive_loglog_slope_tolerance',
        'AR_SLOPE_PENALTY_GRID_SIZE': 'slope_penalty_grid_size',
    }
    for old_key, new_key in aliases.items():
        if old_key in settings:
            settings[new_key] = settings.pop(old_key)
    if 'loss' in settings:
        settings['loss_fn'] = settings['loss']
    if 'psd_loss' in settings:
        settings['loss_fn'] = settings['psd_loss']
    if (
        settings.get('asymmetric_overfit_penalty')
        or settings.get('ceiling_to_data')
        or float(settings.get('loglog_slope_change_weight', 0.0)) > 0
        or float(settings.get('positive_loglog_slope_weight', 0.0)) > 0
    ):
        settings['use_robust_aperiodic_ar'] = True
    return settings


def psd_settings_for_dataset(dataset, settings_by_dataset=None):
    settings_by_dataset = PSD_FIT_SETTINGS_BY_DATASET if settings_by_dataset is None else settings_by_dataset
    keys = [
        'default',
        dataset.get('metadata', {}).get('dataset'),
        dataset.get('name'),
        dataset.get('metadata', {}).get('dataset_key'),
        dataset.get('key'),
    ]
    overrides = {}
    for key in keys:
        if key is not None and key in settings_by_dataset:
            overrides.update(settings_by_dataset[key])
    return resolve_psd_fit_settings(overrides)


def compact_psd_fit_settings(settings):
    keep = [
        'order', 'fallback_order', 'order_sequence', 'ar_bounds',
        'first_phi_guess', 'guess_phis', 'loss_fn', 'f_scale', 'maxfev',
        'sigma', 'sigma_kwargs', 'absolute_sigma', 'fit_min_freq',
        'fit_max_freq', 'fill_outside_fit_range_with_raw',
        'plot_fit_range_only', 'fit_resid_z', 'fit_iterations',
        'exclude_bands', 'include_line_noise', 'use_robust_aperiodic_ar',
        'asymmetric_overfit_penalty', 'overfit_weight',
        'low_freq_residual_weight', 'low_freq_overfit_weight',
        'low_freq_max', 'overfit_tolerance_log10', 'ceiling_to_data',
        'mask_peaks_for_fit', 'peak_mask_min_freq', 'peak_mask_window',
        'peak_mask_threshold', 'loglog_slope_change_weight',
        'positive_loglog_slope_weight', 'positive_loglog_slope_tolerance',
        'slope_penalty_grid_size',
    ]
    compact = {}
    for key in keep:
        value = settings.get(key)
        if key == 'sigma' and callable(value):
            value = getattr(value, '__name__', repr(value))
        elif key == 'sigma' and value is not None and np.ndim(value) > 0:
            value = f'array shape {np.shape(value)}'
        compact[key] = value
    return compact


def compute_spectrum_and_kappa(X, fs, chunk_rows=FIT_CHUNK_ROWS, eps=1e-12):
    X = np.asarray(X)
    m, n = X.shape
    scatter = np.zeros(n, dtype=float)
    A_sum = np.zeros((n, n), dtype=np.complex128)
    for start in range(0, m, chunk_rows):
        chunk = X[start:start + chunk_rows]
        Z = np.fft.fft(chunk, axis=1) / np.sqrt(n)
        scatter += np.sum(np.abs(Z) ** 2, axis=0).real
        A_sum += Z.conj().T @ Z
    A = A_sum / m
    d = np.maximum(np.real(np.diag(A)), eps)
    total_sq = np.sum(np.abs(A) ** 2)
    diag_sq = np.sum(d ** 2)
    off_sq = total_sq - diag_sq
    expected_off_sq = ((d.sum() ** 2) - np.sum(d ** 2)) / m
    excess_off_sq = max(off_sq - expected_off_sq, 0.0)
    kappa = float(diag_sq / (diag_sq + excess_off_sq + eps))
    freqs_pos = np.fft.rfftfreq(n, d=1 / fs)
    powers_pos = scatter[:len(freqs_pos)]
    return freqs_pos, powers_pos, scatter, kappa


def analysis_frequency_mask(freqs, fs, include_line=False, band=None):
    low, high = PERIODIC_SCORE_BAND if band is None else band
    high = min(high, 0.45 * fs)
    mask = (freqs >= low) & (freqs <= high)
    if not include_line:
        mask &= ~line_noise_frequency_mask(freqs, fs)
    return mask


def fit_range_mask(freqs, fs, settings=None):
    settings = resolve_psd_fit_settings(settings)
    mask = freqs >= settings['fit_min_freq']
    if settings['fit_max_freq'] is not None:
        mask &= freqs <= settings['fit_max_freq']
    else:
        mask &= freqs <= 0.45 * fs
    return mask


def peak_stable_frequency_mask(freqs, powers, settings):
    stable = np.ones_like(freqs, dtype=bool)
    if not settings.get('mask_peaks_for_fit', False):
        return stable
    stable[:] = False
    stable[1:] = mask_peaks(
        powers,
        window=int(settings.get('peak_mask_window', 5)),
        threshold=float(settings.get('peak_mask_threshold', 0.75)),
    )
    peak_mask_min_freq = settings.get('peak_mask_min_freq')
    if peak_mask_min_freq is not None:
        stable |= freqs < float(peak_mask_min_freq)
    return stable


def fit_frequency_mask(freqs, powers, fs, settings=None):
    settings = resolve_psd_fit_settings(settings)
    mask = finite_positive_mask(freqs, powers)
    mask &= fit_range_mask(freqs, fs, settings)
    if not settings.get('include_line_noise', False):
        mask &= ~line_noise_frequency_mask(freqs, fs)
    mask &= mask_frequency_bands(freqs, settings.get('exclude_bands', ()))
    mask &= peak_stable_frequency_mask(freqs, powers, settings)
    return mask


def fit_plot_mask(freqs, powers_fit, fs, settings=None):
    settings = resolve_psd_fit_settings(settings)
    mask = finite_positive_mask(freqs, powers_fit)
    plot_fit_range_only = settings.get('plot_fit_range_only')
    if plot_fit_range_only is None:
        plot_fit_range_only = settings.get('fill_outside_fit_range_with_raw', False)
    if plot_fit_range_only:
        mask &= fit_range_mask(freqs, fs, settings)
    return mask


def edge_weight_sigma(freqs, powers=None, fs=None, settings=None, edge_weight=20.0, exponent=4):
    freqs = np.asarray(freqs, dtype=float)
    if len(freqs) <= 1:
        return np.ones_like(freqs)
    x = np.linspace(-1.0, 1.0, len(freqs))
    weights = 1.0 + (edge_weight - 1.0) * np.abs(x) ** exponent
    return 1.0 / np.sqrt(weights)


def resolve_curve_fit_sigma(sigma, freqs_full, powers_full, active_mask, fs, settings):
    if sigma is None:
        return None

    active_freqs = freqs_full[active_mask]
    active_powers = powers_full[active_mask]
    sigma_kwargs = dict(settings.get('sigma_kwargs') or {})

    if callable(sigma):
        try:
            sigma = sigma(
                freqs=active_freqs,
                powers=active_powers,
                fs=fs,
                settings=settings,
                **sigma_kwargs,
            )
        except TypeError:
            sigma = sigma(active_freqs, active_powers, fs, **sigma_kwargs)

    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 0:
        sigma_value = float(sigma)
        if not np.isfinite(sigma_value) or sigma_value <= 0:
            raise ValueError('curve_fit sigma must be positive and finite.')
        return sigma_value

    if sigma.shape != active_freqs.shape:
        if sigma.shape == freqs_full.shape:
            sigma = sigma[active_mask]
        else:
            raise ValueError(
                'curve_fit sigma must be scalar, callable, match active fit frequencies, '
                'or match the full positive-frequency vector.'
            )

    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError('curve_fit sigma values must be positive and finite.')
    return sigma


def make_psd_model(freqs, powers, fs, order, settings=None, curve_fit_kwargs=None):
    settings = resolve_psd_fit_settings(settings)
    fit_mask = finite_positive_mask(freqs, powers)
    power_scale = float(np.nanmedian(powers[fit_mask])) if np.any(fit_mask) else 1.0
    power_scale = max(power_scale, 1e-12)
    guess_phis = settings.get('guess_phis')
    if guess_phis is None:
        phi_guess = [settings['first_phi_guess']] + [0.0] * (order - 1)
    else:
        phi_guess = list(guess_phis)[:order] + [0.0] * max(order - len(guess_phis), 0)
    guess = [*phi_guess, power_scale]
    ar_low, ar_high = settings['ar_bounds']
    power_low, power_high = settings.get('power_bounds_scale', (1e-8, 1e8))
    bounds = [
        [ar_low] * order + [power_scale * power_low],
        [ar_high] * order + [power_scale * power_high],
    ]
    return ARPSD(
        order=order,
        fs=fs,
        bounds=bounds,
        guess=guess,
        maxfev=settings['maxfev'],
        loss_fn=settings['loss_fn'],
        f_scale=settings['f_scale'],
        curve_fit_kwargs=curve_fit_kwargs,
    )


def predict_arpsd(psd_model, freqs):
    k = np.arange(1, psd_model.order + 1)
    exp = np.exp(-2j * np.pi * np.outer(freqs, k) / psd_model.fs).T
    return _ar_spectrum(exp, *psd_model.params)


def robust_ar_exclude_bands(fs, settings):
    bands = list(settings.get('exclude_bands', ()) or ())
    if not settings.get('include_line_noise', False):
        bands.extend(line_noise_bands(fs))
    return tuple(bands)


def use_robust_ar_fit(settings):
    return bool(
        settings.get('use_robust_aperiodic_ar')
        or settings.get('mask_peaks_for_fit')
        or settings.get('asymmetric_overfit_penalty')
        or settings.get('ceiling_to_data')
        or float(settings.get('loglog_slope_change_weight', 0.0)) > 0
        or float(settings.get('positive_loglog_slope_weight', 0.0)) > 0
    )


def fit_powers_with_robust_ar(freqs_pos, powers_pos, fs, order, settings=None):
    settings = resolve_psd_fit_settings(settings)
    model = RobustAperiodicAR(
        fs=float(fs),
        order=int(order),
        fit_min_freq=float(settings['fit_min_freq']),
        fit_max_freq=settings['fit_max_freq'],
        exclude_bands=robust_ar_exclude_bands(fs, settings),
        loss_fn=settings['loss_fn'],
        f_scale=settings['f_scale'],
        maxfev=int(settings['maxfev']),
        asymmetric_overfit_penalty=bool(settings.get('asymmetric_overfit_penalty', False)),
        overfit_weight=float(settings.get('overfit_weight', 0.0)),
        low_freq_residual_weight=float(settings.get('low_freq_residual_weight', 1.0)),
        low_freq_overfit_weight=float(settings.get('low_freq_overfit_weight', 0.0)),
        low_freq_max=float(settings.get('low_freq_max', 5.0)),
        overfit_tolerance_log10=float(settings.get('overfit_tolerance_log10', 0.1)),
        ceiling_to_data=bool(settings.get('ceiling_to_data', False)),
        mask_peaks_for_fit=bool(settings.get('mask_peaks_for_fit', False)),
        peak_mask_min_freq=settings.get('peak_mask_min_freq'),
        peak_mask_window=int(settings.get('peak_mask_window', 5)),
        peak_mask_threshold=float(settings.get('peak_mask_threshold', 0.75)),
        loglog_slope_change_weight=float(settings.get('loglog_slope_change_weight', 0.0)),
        positive_loglog_slope_weight=float(settings.get('positive_loglog_slope_weight', 0.0)),
        positive_loglog_slope_tolerance=float(settings.get('positive_loglog_slope_tolerance', 0.0)),
        slope_penalty_grid_size=settings.get('slope_penalty_grid_size'),
    )
    powers_fit_pos = np.maximum(model.fit(freqs_pos, powers_pos), 1e-12)
    powers_fit_pos[0] = powers_pos[0]
    powers_fit_full = mirror_pos_to_full(powers_fit_pos, (len(freqs_pos) - 1) * 2)
    range_mask = model.range_mask if model.range_mask is not None else fit_range_mask(freqs_pos, fs, settings)
    fit_mask_before_peak = model.fit_mask_before_peak_mask if model.fit_mask_before_peak_mask is not None else fit_frequency_mask(freqs_pos, powers_pos, fs, settings)
    fit_mask = model.fit_mask if model.fit_mask is not None else fit_mask_before_peak
    peak_mask = model.peak_mask if model.peak_mask is not None else np.ones_like(fit_mask, dtype=bool)
    filled_raw_bins = int((~range_mask).sum())
    fit_info = {
        'psd_fit_method': 'RobustAperiodicAR',
        'psd_order_used': int(order),
        'psd_fit_bins_initial': int(fit_mask_before_peak.sum()),
        'psd_fit_bins_final': int(fit_mask.sum()),
        'psd_fit_excluded_bins': int(fit_mask_before_peak.sum() - fit_mask.sum()),
        'psd_fit_range_bins': int(range_mask.sum()),
        'psd_fit_filled_raw_bins': filled_raw_bins,
        'psd_fit_fill_outside_range_with_raw': True,
        'psd_fit_plot_range_only': bool(settings.get('plot_fit_range_only', False)),
        'psd_fit_uses_sigma': False,
        'psd_fit_uses_peak_mask': bool(settings.get('mask_peaks_for_fit', False)),
        'psd_fit_peak_mask_kept_bins': int(peak_mask.sum()),
        'psd_fit_peak_mask_removed_bins': int(len(peak_mask) - peak_mask.sum()),
        'psd_fit_asymmetric_overfit_penalty': bool(settings.get('asymmetric_overfit_penalty', False)),
        'psd_fit_loglog_slope_change_weight': float(settings.get('loglog_slope_change_weight', 0.0)),
        'psd_fit_positive_loglog_slope_weight': float(settings.get('positive_loglog_slope_weight', 0.0)),
        'psd_fit_positive_loglog_slope_tolerance': float(settings.get('positive_loglog_slope_tolerance', 0.0)),
        'psd_fit_ceiling_to_data': bool(settings.get('ceiling_to_data', False)),
    }
    return model, powers_fit_pos, powers_fit_full, fit_info


def fit_powers_for_order(freqs_pos, powers_pos, fs, order, settings=None):
    settings = resolve_psd_fit_settings(settings)
    if use_robust_ar_fit(settings):
        return fit_powers_with_robust_ar(freqs_pos, powers_pos, fs, order, settings)
    fit_mask = fit_frequency_mask(freqs_pos, powers_pos, fs, settings)
    if fit_mask.sum() <= order + 1:
        raise RuntimeError('Too few frequency bins remain for ARPSD fitting.')
    active_mask = fit_mask.copy()
    psd_model = None
    sigma_used = None
    for _ in range(max(1, settings['fit_iterations'])):
        curve_fit_kwargs = dict(settings.get('curve_fit_kwargs') or {})
        sigma = resolve_curve_fit_sigma(
            settings.get('sigma'),
            freqs_pos,
            powers_pos,
            active_mask,
            fs,
            settings,
        )
        if sigma is not None:
            curve_fit_kwargs['sigma'] = sigma
            curve_fit_kwargs['absolute_sigma'] = settings.get('absolute_sigma', False)
            sigma_used = sigma
        psd_model = make_psd_model(
            freqs_pos[active_mask],
            powers_pos[active_mask],
            fs,
            order,
            settings,
            curve_fit_kwargs=curve_fit_kwargs,
        )
        psd_model.fit(freqs_pos[active_mask], powers_pos[active_mask])
        powers_fit_pos = np.maximum(predict_arpsd(psd_model, freqs_pos), 1e-12)
        residual = np.full_like(freqs_pos, np.nan, dtype=float)
        valid = fit_mask & (powers_fit_pos > 0) & (powers_pos > 0)
        residual[valid] = np.log10(powers_pos[valid]) - np.log10(powers_fit_pos[valid])
        resid_z = robust_zscore(residual[valid])
        next_mask = fit_mask.copy()
        valid_indices = np.flatnonzero(valid)
        next_mask[valid_indices] &= np.abs(resid_z) <= settings['fit_resid_z']
        if next_mask.sum() <= order + 1 or np.array_equal(next_mask, active_mask):
            break
        active_mask = next_mask
    powers_fit_pos = np.maximum(predict_arpsd(psd_model, freqs_pos), 1e-12)
    range_mask = fit_range_mask(freqs_pos, fs, settings)
    filled_raw_bins = 0
    if settings.get('fill_outside_fit_range_with_raw', False):
        outside_range = ~range_mask
        powers_fit_pos[outside_range] = powers_pos[outside_range]
        filled_raw_bins = int(outside_range.sum())
    powers_fit_pos[0] = powers_pos[0]
    powers_fit_full = mirror_pos_to_full(powers_fit_pos, (len(freqs_pos) - 1) * 2)
    fit_info = {
        'psd_order_used': int(order),
        'psd_fit_bins_initial': int(fit_mask.sum()),
        'psd_fit_bins_final': int(active_mask.sum()),
        'psd_fit_excluded_bins': int(fit_mask.sum() - active_mask.sum()),
        'psd_fit_range_bins': int(range_mask.sum()),
        'psd_fit_filled_raw_bins': int(filled_raw_bins),
        'psd_fit_fill_outside_range_with_raw': bool(settings.get('fill_outside_fit_range_with_raw', False)),
        'psd_fit_plot_range_only': bool(settings.get('plot_fit_range_only') if settings.get('plot_fit_range_only') is not None else settings.get('fill_outside_fit_range_with_raw', False)),
        'psd_fit_uses_sigma': bool(sigma_used is not None),
        'psd_fit_uses_peak_mask': bool(settings.get('mask_peaks_for_fit', False)),
    }
    if sigma_used is not None and np.ndim(sigma_used) > 0:
        fit_info.update({
            'psd_fit_sigma_min': float(np.min(sigma_used)),
            'psd_fit_sigma_max': float(np.max(sigma_used)),
        })
    return psd_model, powers_fit_pos, powers_fit_full, fit_info


def fit_powers(freqs_pos, powers_pos, fs, settings=None):
    settings = resolve_psd_fit_settings(settings)
    if settings.get('order_sequence') is not None:
        orders = list(settings['order_sequence'])
    else:
        orders = [settings['order']]
        fallback_order = settings.get('fallback_order')
        if fallback_order is not None and fallback_order != settings['order']:
            orders.append(fallback_order)
    last_error = None
    for order in orders:
        try:
            return fit_powers_for_order(freqs_pos, powers_pos, fs, int(order), settings)
        except Exception as exc:
            last_error = exc
            print(f'  warning: AR({order}) fit failed: {exc}')
    raise last_error


def periodic_frequency_mask(n, fs, band=None):
    freqs = np.abs(np.fft.fftfreq(n, d=1 / fs))
    return analysis_frequency_mask(freqs, fs, include_line=False, band=band)


def positive_periodic_frequency_mask(n, fs, band=None):
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    return freqs, analysis_frequency_mask(freqs, fs, include_line=False, band=band)


def positive_line_frequency_mask(n, fs, band=None):
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    band_mask = analysis_frequency_mask(freqs, fs, include_line=True, band=band)
    return freqs, band_mask & line_noise_frequency_mask(freqs, fs)


def periodic_row_scores(X, powers_fit_full, n_fit_rows, fs, score_band=None, chunk_rows=SCORE_CHUNK_ROWS):
    X = np.asarray(X)
    n = X.shape[1]
    per_row_fit = np.maximum(powers_fit_full / n_fit_rows, 1e-12)
    score_mask = periodic_frequency_mask(n, fs, band=score_band)
    scores = np.zeros(X.shape[0], dtype=float)
    if not np.any(score_mask):
        raise ValueError('No Fourier bins remain for periodic row scoring.')
    for start in range(0, X.shape[0], chunk_rows):
        chunk = X[start:start + chunk_rows]
        Z = np.fft.fft(chunk, axis=1) / np.sqrt(n)
        row_power = np.abs(Z[:, score_mask]) ** 2
        excess = np.maximum(row_power - per_row_fit[score_mask][None, :], 0.0)
        scores[start:start + len(chunk)] = excess.max(axis=1)
    return scores


def decompose_selected_rows(
    X_rows,
    powers_raw_full,
    powers_fit_full,
    eps=1e-12,
    periodic_mode=None,
):
    periodic_mode = (
        PERIODIC_DECOMPOSITION_MODE if periodic_mode is None else periodic_mode
    )
    return decompose_fourier(
        X_rows,
        powers_raw_full,
        powers_fit_full,
        periodic_mode=periodic_mode,
        eps=eps,
    )


def sustained_row_quality(X_rows, fs, eps=1e-12):
    X_rows = np.asarray(X_rows)
    seg_len = max(4, int(round(SUSTAINED_SEGMENT_S * fs)))
    n_segments = X_rows.shape[1] // seg_len
    if n_segments < 2:
        return {
            'sustained_fraction': np.ones(X_rows.shape[0]),
            'sustained_cv': np.zeros(X_rows.shape[0]),
            'sustained_score': np.ones(X_rows.shape[0]),
        }
    trimmed = X_rows[:, :n_segments * seg_len]
    segments = trimmed.reshape(X_rows.shape[0], n_segments, seg_len)
    segment_rms = np.sqrt(np.mean(segments ** 2, axis=2))
    max_rms = np.max(segment_rms, axis=1, keepdims=True)
    sustained_fraction = np.mean(segment_rms >= SUSTAINED_RMS_RATIO * (max_rms + eps), axis=1)
    sustained_cv = np.std(segment_rms, axis=1) / (np.mean(segment_rms, axis=1) + eps)
    sustained_score = sustained_fraction / (1 + sustained_cv)
    return {
        'sustained_fraction': sustained_fraction,
        'sustained_cv': sustained_cv,
        'sustained_score': sustained_score,
    }


def raw_periodic_correlations(X_raw_rows, X_periodic_rows, eps=1e-12):
    X_raw_rows = np.asarray(X_raw_rows, dtype=float)
    X_periodic_rows = np.asarray(X_periodic_rows, dtype=float)
    raw = X_raw_rows - X_raw_rows.mean(axis=1, keepdims=True)
    periodic = X_periodic_rows - X_periodic_rows.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(raw, axis=1) * np.linalg.norm(periodic, axis=1)
    return np.sum(raw * periodic, axis=1) / (denom + eps)


def periodic_quality(X_rows, fs, score_band=None, eps=1e-12):
    X_rows = np.asarray(X_rows)
    freqs, periodic_mask = positive_periodic_frequency_mask(X_rows.shape[1], fs, band=score_band)
    _, line_mask = positive_line_frequency_mask(X_rows.shape[1], fs, band=score_band)
    total_mask = analysis_frequency_mask(freqs, fs, include_line=True, band=score_band)
    if not np.any(periodic_mask):
        raise ValueError('No positive-frequency bins remain for periodic quality.')
    powers = np.abs(np.fft.rfft(X_rows, axis=1)) ** 2
    band_powers = powers[:, periodic_mask]
    band_freqs = freqs[periodic_mask]
    peak_inds = np.argmax(band_powers, axis=1)
    peak_powers = band_powers[np.arange(len(X_rows)), peak_inds]
    total_periodic_powers = band_powers.sum(axis=1)
    total_powers = powers[:, total_mask].sum(axis=1) if np.any(total_mask) else total_periodic_powers
    line_powers = powers[:, line_mask].sum(axis=1) if np.any(line_mask) else np.zeros(len(X_rows))
    peak_fraction = peak_powers / (total_periodic_powers + eps)
    line_fraction = line_powers / (total_powers + eps)
    periodic_variance = X_rows.var(axis=1)
    sustained = sustained_row_quality(X_rows, fs, eps=eps)
    return pd.DataFrame({
        'peak_freq': band_freqs[peak_inds],
        'peak_fraction': peak_fraction,
        'line_fraction': line_fraction,
        'periodic_variance': periodic_variance,
        **sustained,
    })


def spectrum_bump_quality(freqs, powers, powers_fit, fs, score_band=None, eps=1e-12):
    mask = analysis_frequency_mask(freqs, fs, include_line=False, band=score_band)
    mask &= (powers > 0) & (powers_fit > 0)
    if not np.any(mask):
        return {'spectrum_peak_freq': np.nan, 'spectrum_peak_ratio': np.nan,
                'spectrum_excess_peak_freq': np.nan, 'spectrum_excess_power': np.nan,
                'spectrum_ratio_peak_freq': np.nan, 'spectrum_ratio_peak_ratio': np.nan}
    ratio = powers / np.maximum(powers_fit, eps)
    excess = np.maximum(powers - powers_fit, 0.0)
    ratio_idx = np.nanargmax(np.where(mask, ratio, np.nan))
    excess_idx = np.nanargmax(np.where(mask, excess, np.nan))
    return {
        'spectrum_peak_freq': float(freqs[excess_idx]),
        'spectrum_peak_ratio': float(ratio[excess_idx]),
        'spectrum_excess_peak_freq': float(freqs[excess_idx]),
        'spectrum_excess_power': float(excess[excess_idx]),
        'spectrum_ratio_peak_freq': float(freqs[ratio_idx]),
        'spectrum_ratio_peak_ratio': float(ratio[ratio_idx]),
    }


def aperiodic_fit_quality(freqs, powers, powers_fit, fs, settings=None, eps=1e-12):
    mask = fit_frequency_mask(freqs, powers, fs, settings)
    mask &= powers_fit > 0
    if mask.sum() == 0:
        return {'psd_fit_log_mae': np.nan, 'psd_fit_log_bias': np.nan}
    log_err = np.log10(powers[mask] + eps) - np.log10(powers_fit[mask] + eps)
    return {
        'psd_fit_log_mae': float(np.nanmean(np.abs(log_err))),
        'psd_fit_log_bias': float(np.nanmean(log_err)),
    }


def selected_row_metadata(row_indices, n_channels):
    row_indices = np.asarray(row_indices, dtype=int)
    return pd.DataFrame({
        'source_row': row_indices,
        'trial_or_window_index': row_indices // n_channels,
        'channel_index': row_indices % n_channels,
    })


def select_rows_for_kappa(X, fs):
    X = np.asarray(X)
    min_rows = min(X.shape[0], max(N_PLOT, MIN_KAPPA_SELECTION_ROWS, PSD_ORDER + 2))
    if X.shape[0] <= min_rows:
        freqs, powers, powers_full, kappa = compute_spectrum_and_kappa(X, fs)
        return {
            'indices': np.arange(X.shape[0]),
            'freqs': freqs,
            'powers': powers,
            'powers_full': powers_full,
            'kappa': kappa,
            'selection_quantile': 1.0,
            'selection_threshold': np.nan,
            'rows_before': int(X.shape[0]),
            'rows_after': int(X.shape[0]),
            'reason': 'all_rows_too_few_to_refine',
        }

    _, freq_mask = row_qc_frequency_mask(X.shape[1], fs)
    if not np.any(freq_mask):
        freqs, powers, powers_full, kappa = compute_spectrum_and_kappa(X, fs)
        return {
            'indices': np.arange(X.shape[0]),
            'freqs': freqs,
            'powers': powers,
            'powers_full': powers_full,
            'kappa': kappa,
            'selection_quantile': 1.0,
            'selection_threshold': np.nan,
            'rows_before': int(X.shape[0]),
            'rows_after': int(X.shape[0]),
            'reason': 'all_rows_no_qc_frequency_bins',
        }

    keep_all = np.ones(X.shape[0], dtype=bool)
    center = row_log_psd_center(X, fs, keep_all, freq_mask)
    distances = row_log_psd_distances(X, center, freq_mask)
    finite = np.isfinite(distances)
    if finite.sum() < min_rows:
        finite[:] = True
        distances = np.zeros(X.shape[0], dtype=float)

    best_below_target = None
    for quantile in KAPPA_SELECTION_QUANTILES:
        threshold = float(np.nanquantile(distances[finite], quantile))
        keep = finite & (distances <= threshold)
        if keep.sum() < min_rows:
            continue
        freqs, powers, powers_full, kappa = compute_spectrum_and_kappa(X[keep], fs)
        candidate = {
            'indices': np.flatnonzero(keep),
            'freqs': freqs,
            'powers': powers,
            'powers_full': powers_full,
            'kappa': float(kappa),
            'selection_quantile': float(quantile),
            'selection_threshold': threshold,
            'rows_before': int(X.shape[0]),
            'rows_after': int(keep.sum()),
            'distance_median': float(np.nanmedian(distances[keep])),
            'distance_q95': float(np.nanquantile(distances[keep], 0.95)),
            'reason': 'largest_subset_reaching_kappa_target',
        }
        if candidate['kappa'] >= KAPPA_TARGET:
            return candidate
        if best_below_target is None or (candidate['kappa'], candidate['rows_after']) > (best_below_target['kappa'], best_below_target['rows_after']):
            best_below_target = candidate

    if best_below_target is not None:
        best_below_target['reason'] = 'highest_kappa_subset_below_target'
        return best_below_target

    freqs, powers, powers_full, kappa = compute_spectrum_and_kappa(X, fs)
    return {
        'indices': np.arange(X.shape[0]),
        'freqs': freqs,
        'powers': powers,
        'powers_full': powers_full,
        'kappa': kappa,
        'selection_quantile': 1.0,
        'selection_threshold': np.nan,
        'rows_before': int(X.shape[0]),
        'rows_after': int(X.shape[0]),
        'reason': 'all_rows_no_valid_kappa_subset',
    }


def fit_dataset(dataset):
    psd_settings = psd_settings_for_dataset(dataset)
    score_band = tuple(dataset.get('metadata', {}).get('periodic_score_band_hz', PERIODIC_SCORE_BAND))
    dataset['metadata']['periodic_score_band_hz'] = score_band
    kappa_selection = select_rows_for_kappa(dataset['X'], dataset['fs'])
    fit_inds = kappa_selection['indices']
    X = dataset['X'][fit_inds]
    freqs = kappa_selection['freqs']
    powers = kappa_selection['powers']
    powers_full = kappa_selection['powers_full']
    kappa = kappa_selection['kappa']
    if kappa < MIN_KAPPA:
        print(f"  warning: {dataset['name']} kappa={kappa:.3f} below MIN_KAPPA={MIN_KAPPA}")

    psd_model, powers_fit, powers_fit_full, fit_info = fit_powers(freqs, powers, dataset['fs'], psd_settings)
    spectrum_quality = spectrum_bump_quality(freqs, powers, powers_fit, dataset['fs'], score_band)
    fit_quality = aperiodic_fit_quality(freqs, powers, powers_fit, dataset['fs'], psd_settings)
    if np.isfinite(spectrum_quality['spectrum_peak_ratio']) and spectrum_quality['spectrum_peak_ratio'] < MIN_SPECTRAL_PEAK_RATIO:
        print(
            f"  warning: weak non-line eigenspectrum bump "
            f"(ratio={spectrum_quality['spectrum_peak_ratio']:.2f})"
        )

    dataset['metadata'].update(fit_info)
    dataset['metadata'].update(fit_quality)
    dataset['metadata'].update({
        'rows_before_kappa_selection': int(kappa_selection['rows_before']),
        'rows_used_for_fit': int(kappa_selection['rows_after']),
        'kappa_selection_quantile': float(kappa_selection['selection_quantile']),
        'kappa_selection_threshold': float(kappa_selection['selection_threshold']) if np.isfinite(kappa_selection['selection_threshold']) else np.nan,
        'kappa_selection_reason': kappa_selection['reason'],
        'kappa_selection_distance_median': float(kappa_selection.get('distance_median', np.nan)),
        'kappa_selection_distance_q95': float(kappa_selection.get('distance_q95', np.nan)),
        'kappa': float(kappa),
        'kappa_target': float(KAPPA_TARGET),
        'kappa_target_gap': float(max(KAPPA_TARGET - kappa, 0.0)),
        'periodic_score_band_hz': tuple(score_band),
        'psd_fit_settings': str(compact_psd_fit_settings(psd_settings)),
        'periodic_decomposition_mode': PERIODIC_DECOMPOSITION_MODE,
    })

    if X.shape[0] > MAX_SCORE_ROWS:
        score_pool_inds = np.unique(np.linspace(0, X.shape[0] - 1, MAX_SCORE_ROWS, dtype=int))
    else:
        score_pool_inds = np.arange(X.shape[0])
    row_scores_pool = periodic_row_scores(X[score_pool_inds], powers_fit_full, X.shape[0], dataset['fs'], score_band=score_band)
    n_candidates = min(max(N_CANDIDATE_ROWS, N_PLOT), len(score_pool_inds))
    candidate_order = np.argsort(row_scores_pool)[-n_candidates:][::-1]
    candidate_inds = score_pool_inds[candidate_order]
    candidate_scores = row_scores_pool[candidate_order]
    X_candidates = X[candidate_inds]
    X_ap_candidates, X_pe_candidates = decompose_selected_rows(
        X_candidates,
        powers_full,
        powers_fit_full,
        periodic_mode=PERIODIC_DECOMPOSITION_MODE,
    )
    quality = periodic_quality(X_pe_candidates, dataset['fs'], score_band=score_band)
    quality['row_score'] = candidate_scores
    quality['raw_periodic_corr'] = raw_periodic_correlations(X_candidates, X_pe_candidates)
    quality['aperiodic_variance'] = X_ap_candidates.var(axis=1)
    quality['periodic_variance'] = X_pe_candidates.var(axis=1)
    quality['variance_balance'] = np.minimum(
        quality['aperiodic_variance'], quality['periodic_variance']
    ) / (np.maximum(quality['aperiodic_variance'], quality['periodic_variance']) + 1e-12)
    quality['periodicity_score'] = (
        quality['periodic_variance']
        * quality['peak_fraction']
        * (1 - quality['line_fraction'])
        * np.sqrt(np.maximum(quality['variance_balance'], 0))
        * quality['sustained_score']
    )

    # Same display-row rule as the hippocampal publication figure: traverse high
    # oscillatory-power candidates, keep rows whose isolated periodic component
    # visibly matches the raw trace, then fall back to best correlated high-power rows.
    high_power_correlated = quality[quality['raw_periodic_corr'] >= PLOT_ROW_MIN_RAW_PERIODIC_CORR]
    if len(high_power_correlated) >= N_PLOT:
        selected_quality = high_power_correlated.head(N_PLOT)
        plot_selection_reason = 'high_power_rows_passing_raw_periodic_corr'
    else:
        print(
            f"  warning: only {len(high_power_correlated)} high-power candidates reached "
            f"raw-periodic corr >= {PLOT_ROW_MIN_RAW_PERIODIC_CORR:.2f}; using best correlated high-power rows"
        )
        selected_quality = quality.sort_values(
            ['raw_periodic_corr', 'row_score'],
            ascending=[False, False],
        ).head(N_PLOT)
        plot_selection_reason = 'fallback_best_raw_periodic_corr_then_power'
    order = selected_quality.index.to_numpy(dtype=int)
    plot_inds = candidate_inds[order]
    analysis_source_rows = dataset.get('row_indices', np.arange(dataset['X'].shape[0]))[fit_inds]
    source_row_indices = analysis_source_rows[plot_inds]
    X_plot = X_candidates[order]
    X_ap = X_ap_candidates[order]
    X_pe = X_pe_candidates[order]
    selected_rows = selected_row_metadata(source_row_indices, dataset['metadata']['electrodes_or_channels'])
    selected_rows.insert(0, 'fit_matrix_row', plot_inds)
    selected_rows = pd.concat([selected_rows.reset_index(drop=True), quality.loc[order].reset_index(drop=True)], axis=1)
    dataset['metadata'].update({
        'plot_row_selection': 'high oscillatory-power rows with raw-periodic correlation threshold, matching notebook 14',
        'plot_row_min_raw_periodic_corr': float(PLOT_ROW_MIN_RAW_PERIODIC_CORR),
        'plot_row_selection_reason': plot_selection_reason,
        'plot_row_candidates': int(n_candidates),
        'plot_row_candidates_passing_raw_periodic_corr': int(len(high_power_correlated)),
        'plot_row_raw_periodic_corr_median': float(selected_rows['raw_periodic_corr'].median()),
        'plot_row_raw_periodic_corr_min': float(selected_rows['raw_periodic_corr'].min()),
    })
    out = dict(dataset)
    out.update({
        'psd_model': psd_model,
        'X_plot': X_plot,
        'X_ap': X_ap,
        'X_pe': X_pe,
        'plot_inds': plot_inds,
        'fit_row_indices': fit_inds,
        'source_row_indices': source_row_indices,
        'selected_rows': selected_rows,
        'periodic_scores': quality.loc[order, 'row_score'].to_numpy(),
        'spectrum_quality': spectrum_quality,
        'fit_quality': fit_quality,
        'kappa': kappa,
        'freqs': freqs,
        'powers': powers,
        'powers_fit': powers_fit,
        'psd_settings': psd_settings,
        'n_fit_rows': int(X.shape[0]),
    })
    del out['X']
    return out

import hashlib
import pickle

RESULT_CACHE_DIR = DATA_ROOT / 'result_cache'
RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Bump this after changing row selection, fitting, loader logic, or decomposition code.
RESULT_CACHE_VERSION = 'multimodal-results-v14-balanced-quadrature'

# Set True once when you want to ignore cached fits and rebuild them.
REFIT_RESULTS = False
CACHE_RESULTS = True

# Dataset-specific cache tags invalidate only the named dataset.
DATASET_CACHE_VERSION_BY_KEY = {
    'human_lemon_alpha': 'lemon-window-1.0',
    'human_pardo_valencia_stn_beta': 'pardo-valencia-p1-mastn-2s-step075-v1',
    'human_bciciii_ecog_motor_beta': 'bciciii-i-top5-1s-qc2000-v1',
    'human_miller_fingerflex_beta': 'full-stanford-clean-beta-pooled-v2-maxrows-kappa09',
}

# Optional display-name aliases used only for cache invalidation. Prefer dataset
# keys in CUSTOM_PSD_FIT_SETTINGS_BY_DATASET; aliases keep old name-keyed
# overrides from invalidating unrelated datasets.
PSD_CACHE_SETTING_ALIASES_BY_KEY = {
    'human_cole_m1_beta': ('Human M1 ECoG: Cole 2017 Parkinson beta',),
    'human_pardo_valencia_stn_beta': ('Human STN LFP: Pardo-Valencia Parkinson beta',),
    'human_bciciii_ecog_motor_beta': ('Human ECoG: BCI III motor mu/beta',),
    'human_miller_fingerflex_beta': tuple(
        f'Human ECoG: Miller FingerFlex beta {code}'
        for code in MILLER_FINGERFLEX_SUBJECT_CODES
    ),
    'macaque_visual_grating_ecog': ('Macaque ECoG: visual grating',),
}

RESULT_CACHE_SETTING_NAMES = [
    'WINDOW_S',
    'N_PLOT',
    'MIN_KAPPA',
    'KAPPA_TARGET',
    'PSD_ORDER',
    'PERIODIC_DECOMPOSITION_MODE',
    'PERIODIC_SCORE_BAND',
    'LINE_NOISE_BASES',
    'ROW_QC_PSD_DEVIANCE_Z',
    'ROW_QC_MEAN_Z',
    'ROW_QC_STD_LOG_Z',
    'ROW_QC_MAX_ITER',
    'ROW_QC_FREQ_RANGE',
    'APPLY_SIGNAL_QC',
    'STANDARDIZE_FOR_MODEL',
    'SIGNAL_QC_PSD_RANGE',
    'SIGNAL_QC_MIN_OSC_PROMINENCE',
    'SIGNAL_QC_MIN_OSC_FRACTION',
    'SIGNAL_QC_MIN_HALF_PSD_CORR',
    'SIGNAL_QC_MIN_PSD_REFERENCE_CORR',
    'SIGNAL_QC_MAX_MEAN_ROBUST_Z',
    'SIGNAL_QC_MAX_LOG_STD_ROBUST_Z',
    'SIGNAL_QC_MIN_ROWS_PER_DATASET',
    'FIT_CHUNK_ROWS',
    'SCORE_CHUNK_ROWS',
    'PSD_AR_BOUNDS',
    'PSD_FIRST_PHI_GUESS',
    'PSD_LOSS',
    'PSD_F_SCALE',
    'PSD_MAXFEV',
    'PSD_FIT_MIN_FREQ',
    'PSD_FIT_MAX_FREQ',
    'PSD_FIT_RESID_Z',
    'PSD_FIT_ITERATIONS',
    'PSD_FALLBACK_ORDER',
    'N_CANDIDATE_ROWS',
    'MAX_SCORE_ROWS',
    'MIN_PERIODIC_PEAK_FRACTION',
    'MIN_SPECTRAL_PEAK_RATIO',
    'MAX_LINE_NOISE_FRACTION',
    'MIN_VARIANCE_BALANCE',
    'SUSTAINED_SEGMENT_S',
    'SUSTAINED_RMS_RATIO',
    'MIN_SUSTAINED_FRACTION',
    'PLOT_ROW_MIN_RAW_PERIODIC_CORR',
    'KAPPA_SELECTION_QUANTILES',
    'MIN_KAPPA_SELECTION_ROWS',
]


def cache_repr(value):
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        return {
            'type': 'ndarray',
            'shape': arr.shape,
            'dtype': str(arr.dtype),
            'sha256': hashlib.sha256(arr.view(np.uint8)).hexdigest(),
        }
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): cache_repr(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [cache_repr(val) for val in value]
    if callable(value):
        return getattr(value, '__name__', repr(value))
    return value


def psd_cache_keys_for_dataset(dataset_key):
    keys = ['default']
    keys.extend(PSD_CACHE_SETTING_ALIASES_BY_KEY.get(dataset_key, ()))
    keys.append(dataset_key)
    return [key for key in keys if key in PSD_FIT_SETTINGS_BY_DATASET]


def psd_cache_settings_for_dataset(dataset_key, settings_by_dataset=None):
    settings_by_dataset = PSD_FIT_SETTINGS_BY_DATASET if settings_by_dataset is None else settings_by_dataset
    overrides = {}
    for key in psd_cache_keys_for_dataset(dataset_key):
        if key in settings_by_dataset:
            overrides.update(settings_by_dataset[key])
    return resolve_psd_fit_settings(overrides)


def result_cache_key(dataset_key):
    module_settings = _module_settings()
    settings = {
        name: cache_repr(module_settings[name])
        for name in RESULT_CACHE_SETTING_NAMES
        if name in module_settings
    }
    payload = {
        'version': RESULT_CACHE_VERSION,
        'dataset_key': dataset_key,
        'dataset_cache_version': DATASET_CACHE_VERSION_BY_KEY.get(dataset_key, ''),
        'settings': settings,
        'psd_fit_setting_keys_for_dataset': psd_cache_keys_for_dataset(dataset_key),
        'psd_fit_settings_for_dataset': cache_repr(psd_cache_settings_for_dataset(dataset_key)),
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def result_cache_path(dataset_key):
    return RESULT_CACHE_DIR / f'{dataset_key}-{result_cache_key(dataset_key)}.pkl'


def cacheable_result(result):
    out = dict(result)
    out.pop('psd_model', None)  # ARPSD objects are not needed for plotting/tables and may not pickle cleanly.
    out['metadata'] = dict(out.get('metadata', {}))
    out['metadata']['loaded_from_result_cache'] = False
    return out


def load_cached_result(dataset_key):
    path = result_cache_path(dataset_key)
    if not CACHE_RESULTS or REFIT_RESULTS or not path.exists():
        return None
    with path.open('rb') as f:
        result = pickle.load(f)
    result['metadata'] = dict(result.get('metadata', {}))
    result['metadata']['loaded_from_result_cache'] = True
    result['metadata']['result_cache_file'] = path.name
    print(f'Using cached fit for {dataset_key}: {path.name}')
    return result


def save_cached_result(dataset_key, result):
    if not CACHE_RESULTS:
        return
    path = result_cache_path(dataset_key)
    with path.open('wb') as f:
        pickle.dump(cacheable_result(result), f, protocol=pickle.HIGHEST_PROTOCOL)
    result['metadata']['loaded_from_result_cache'] = False
    result['metadata']['result_cache_file'] = path.name


def print_result_summary(result):
    print(
        f"  kappa={result['kappa']:.3f} (gap to target={result['metadata']['kappa_target_gap']:.3f}); "
        f"selected source rows={result['source_row_indices'].tolist()}; "
        f"channels/electrodes={result['metadata']['electrodes_or_channels']}; "
        f"trials/windows={result['metadata']['trials_or_windows']}"
    )
    stationary_rows = result['metadata'].get(
        'row_qc_rows_after_stationary_oscillation',
        result['metadata'].get('row_qc_rows_after_mean_std', result['metadata']['row_qc_rows_after_spectral']),
    )
    print(
        f"  row QC kept {result['metadata']['row_qc_rows_after_spectral']} / "
        f"{result['metadata']['row_qc_rows_before']} rows; "
        f"stationary-oscillation gate kept {stationary_rows} / "
        f"{result['metadata'].get('row_qc_rows_after_mean_std', result['metadata']['row_qc_rows_before'])}; "
        f"fit-row selection kept {result['metadata']['rows_used_for_fit']} / "
        f"{result['metadata']['rows_before_kappa_selection']} rows "
        f"(q={result['metadata']['kappa_selection_quantile']:.3g}); "
        f"AR({result['metadata']['psd_order_used']}) fit bins "
        f"{result['metadata']['psd_fit_bins_final']} / {result['metadata']['psd_fit_bins_initial']}; "
        f"fit MAE={result['metadata']['psd_fit_log_mae']:.3f} log10"
    )
    print(
        f"  eigenspectrum excess peak={result['spectrum_quality']['spectrum_peak_freq']:.1f} Hz "
        f"(ratio={result['spectrum_quality']['spectrum_peak_ratio']:.2f}); "
        f"ratio peak={result['spectrum_quality']['spectrum_ratio_peak_freq']:.1f} Hz"
    )
    print(
        '  periodic peaks=' + ', '.join(
            f"{freq:.1f} Hz (frac={frac:.2f}, line={line:.2f}, balance={bal:.2f}, sustain={sus:.2f})"
            for freq, frac, line, bal, sus in zip(
                result['selected_rows']['peak_freq'],
                result['selected_rows']['peak_fraction'],
                result['selected_rows']['line_fraction'],
                result['selected_rows']['variance_balance'],
                result['selected_rows']['sustained_fraction'],
            )
        )
    )

def configure_multimodal_settings(settings=None, **overrides):
    """Apply an explicit notebook settings mapping to this module.

    The multimodal notebook keeps user-tunable values as all-caps variables.
    Pass a dictionary of those values so the moved helper functions see the
    same values without redefining functions in the notebook.
    """

    incoming = {}
    if settings is not None:
        if hasattr(settings, 'items'):
            incoming.update(settings)
        else:
            incoming.update(vars(settings))
    incoming.update(overrides)

    module_settings = _module_settings()
    applied = {}
    for key, value in incoming.items():
        if key.isupper() or key == 'WORKDIR':
            module_settings[key] = value
            applied[key] = value

    if module_settings.get('APPLY_CUSTOM_PSD_SETTINGS_TO_MAIN_LOOP', False):
        module_settings['PSD_FIT_SETTINGS_BY_DATASET'] = {
            **module_settings.get('DEFAULT_PSD_FIT_SETTINGS_BY_DATASET', {}),
            **module_settings.get('CUSTOM_PSD_FIT_SETTINGS_BY_DATASET', {}),
        }
        applied['PSD_FIT_SETTINGS_BY_DATASET'] = module_settings['PSD_FIT_SETTINGS_BY_DATASET']
    return applied


def dataset_loaders_for_keys(dataset_keys=None, registry=None):
    dataset_keys = list(DATASET_KEYS if dataset_keys is None else dataset_keys)
    registry = DATASET_REGISTRY if registry is None else registry
    unknown = [key for key in dataset_keys if key not in registry]
    if unknown:
        raise KeyError(f'Unknown dataset keys: {unknown}. Available: {list(registry)}')
    return [registry[key] for key in dataset_keys]


def run_multimodal_results(dataset_keys=None, *, settings=None):
    """Load/cache/fetch all selected multimodal examples and fit decompositions."""

    if settings is not None:
        configure_multimodal_settings(settings)
    dataset_keys = list(DATASET_KEYS if dataset_keys is None else dataset_keys)
    loaders = dataset_loaders_for_keys(dataset_keys)
    results = []
    for dataset_key, loader in zip(dataset_keys, loaders):
        result = load_cached_result(dataset_key)
        if result is None:
            dataset = loader()
            dataset['key'] = dataset_key
            dataset['metadata']['dataset_key'] = dataset_key
            print(f"{dataset['name']}: X={dataset['X'].shape}, fs={dataset['fs']:.1f} Hz")
            result = fit_dataset(dataset)
            save_cached_result(dataset_key, result)
            del dataset
        else:
            print(f"{result['name']}: X=(cached), fs={result['fs']:.1f} Hz")
        results.append(result)
        print_result_summary(result)

    metadata_table = pd.DataFrame([result['metadata'] for result in results])
    selected_rows_table = pd.concat(
        [result['selected_rows'].assign(dataset=result['name']) for result in results],
        ignore_index=True,
    )
    return results, metadata_table, selected_rows_table


def panel_scale(X):
    return max(np.nanpercentile(np.abs(X), 99), 1e-12)


def plot_signal_rows(ax, time, X, color='black', alpha=1.0, label=None, lw=0.85, scale=None):
    scale = panel_scale(X) if scale is None else scale
    offsets = 2.7 * scale * np.arange(X.shape[0])
    for row_idx, offset in enumerate(offsets):
        row_label = label if row_idx == 0 else None
        ax.plot(time, X[row_idx] + offset, color=color, lw=lw, label=row_label, alpha=alpha)
    ax.set_ylim(-1.25 * scale, offsets[-1] + 1.25 * scale)
    ax.set_yticks(offsets)
    ax.set_yticklabels(np.arange(X.shape[0]))
    ax.margins(x=0)
    return scale


def add_zoom_inset(ax, time, X, idx, color='C1'):
    inset = ax.inset_axes([0.48, 0.08, 0.48, 0.44])
    row = X[idx] - np.nanmean(X[idx])
    zoom_s = min(0.3, time[-1] - time[0])
    start = time[0]
    mask = (time >= start) & (time <= start + zoom_s)
    inset.plot(time[mask], row[mask], color=color, lw=0.95)
    inset.axhline(0, color='0.75', lw=0.55)
    inset.set_xticks([])
    inset.set_xlabel('')
    inset.set_yticks([])
    inset.text(
        0.98,
        0.06,
        '300 ms',
        transform=inset.transAxes,
        ha='right',
        va='bottom',
        fontsize=7.5,
        color='0.15',
    )
    inset.set_facecolor('white')
    for spine in inset.spines.values():
        spine.set_color('0.2')
        spine.set_linewidth(0.65)


def plot_multimodal_decomposition_figure(
    results,
    *,
    output_path='multimodal_decomposition.svg',
    idxs_inset=None,
):
    """Plot raw signals, eigenspectra, and time-domain components."""

    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'svg.fonttype': 'none',
    })

    n_rows = len(results)
    if idxs_inset is None:
        idxs_inset = [min(4, result['X_pe'].shape[0] - 1) for result in results]

    fig, axes = plt.subplots(
        n_rows,
        4,
        figsize=(16, max(4.0, 3.5 * n_rows)),
        gridspec_kw={'width_ratios': [3.0, 2.6, 3.0, 3.0]},
    )
    if n_rows == 1:
        axes = np.asarray([axes])
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.06, wspace=0.10, hspace=0.10)

    column_titles = ['Signal', 'Eigenspectrum', 'Aperiodic Fit', 'Periodic Fit']
    for i, result in enumerate(results):
        time = np.arange(result['X_plot'].shape[1]) / result['fs']

        signal_scale = panel_scale(result['X_plot'])
        plot_signal_rows(axes[i, 0], time, result['X_plot'], color='black', scale=signal_scale)

        powers_plot = result['powers'] / result['n_fit_rows']
        powers_fit_plot = result['powers_fit'] / result['n_fit_rows']
        spec_mask = fit_plot_mask(result['freqs'], powers_plot, result['fs'], result.get('psd_settings'))
        axes[i, 1].loglog(
            result['freqs'][spec_mask],
            powers_plot[spec_mask],
            color='C0',
            lw=1.25,
            label='Eigenvalues',
        )
        fit_mask = fit_plot_mask(result['freqs'], powers_fit_plot, result['fs'], result.get('psd_settings'))
        axes[i, 1].loglog(
            result['freqs'][fit_mask],
            powers_fit_plot[fit_mask],
            color='C1',
            lw=1.35,
            linestyle='--',
            label='Aperiodic fit',
        )
        axes[i, 1].set_box_aspect(1)
        axes[i, 1].text(
            0.06,
            0.06,
            f'$\\kappa$ = {result["kappa"]:.3f}',
            transform=axes[i, 1].transAxes,
            ha='left',
            va='bottom',
            fontsize=9.5,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2.2),
        )
        print(result['name'])
        axes[i, 0].set_ylabel(result['name'])
        if i == 0:
            axes[i, 1].legend(frameon=True, facecolor='white', framealpha=1, edgecolor='0.7')

        plot_signal_rows(axes[i, 2], time, result['X_ap'], color='C1', scale=signal_scale)
        plot_signal_rows(axes[i, 3], time, result['X_pe'], color='C2', scale=signal_scale)
        plot_signal_rows(axes[i, 2], time, result['X_plot'], color='k', alpha=0.14, lw=0.7, scale=signal_scale)
        plot_signal_rows(axes[i, 3], time, result['X_plot'], color='k', alpha=0.14, lw=0.7, scale=signal_scale)
        add_zoom_inset(axes[i, 3], time, result['X_pe'], idxs_inset[i], color='C2')

        if i == 0:
            for j, title in enumerate(column_titles):
                axes[i, j].set_title(title, pad=8, fontweight='bold')
        if i < n_rows - 1:
            for ax in axes[i, [0, 2, 3]].ravel():
                ax.tick_params(labelbottom=False)
            axes[i, 1].tick_params(labelbottom=False)

    for ax in axes[-1, [0, 2, 3]].ravel():
        ax.set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Frequency (Hz)')

    for ax in axes[:, 0].ravel():
        ax.set_ylabel('Row')
    for ax in axes[:, 1].ravel():
        ax.set_ylabel('Power')
    for ax in axes[:, [2, 3]].ravel():
        ax.set_ylabel('Row')

    for ax in axes.ravel():
        ax.tick_params(axis='both', which='major', length=3.5)
        ax.tick_params(axis='both', which='minor', length=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig
