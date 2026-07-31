"""Row selection methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

from .data import SignalMatrix


def select_rows_for_circulant_kappa(
    X,
    target_kappa=0.95,
    drop_fraction=0.10,
    max_drop_fraction=0.50,
    min_rows=None,
    n_pairs=4096,
    chunk_rows=2048,
    max_iter=20,
    random_state=0,
):
    """Select rows that make the covariance more Fourier-circulant.

    Circulant covariance is diagonal in the Fourier basis. This function scores
    rows by whether their Fourier outer-products align with the current
    off-diagonal Fourier covariance, then removes the worst rows iteratively.
    """

    X = np.asarray(X, dtype=np.float32)
    m0, n = X.shape
    rng = np.random.default_rng(random_state)

    # Standardize internally so row badness is shape/covariance-driven rather
    # than dominated by absolute voltage scale.
    X = X - X.mean(axis=1, keepdims=True)
    X = X / (X.std(axis=1, keepdims=True) + 1e-12)

    keep = np.arange(m0)
    if min_rows is None:
        min_rows = max(32, int(np.ceil((1 - max_drop_fraction) * m0)))
    else:
        min_rows = max(int(min_rows), 1)

    def _kappa(row_idx):
        """Compute diagonal vs excess off-diagonal Fourier covariance energy."""

        A = np.zeros((n, n), dtype=np.complex128)
        for start in range(0, len(row_idx), chunk_rows):
            rows = row_idx[start:start + chunk_rows]
            Z = np.fft.fft(X[rows], axis=1) / np.sqrt(n)
            A += Z.conj().T @ Z
        A /= len(row_idx)

        d = np.maximum(np.real(np.diag(A)), 1e-12)
        total_sq = np.sum(np.abs(A) ** 2)
        diag_sq = np.sum(d ** 2)
        off_sq = total_sq - diag_sq
        expected_off_sq = ((d.sum() ** 2) - np.sum(d ** 2)) / len(row_idx)
        excess_off_sq = max(off_sq - expected_off_sq, 0.0)
        return float(diag_sq / (diag_sq + excess_off_sq + 1e-12))

    def _badness(row_idx):
        """Approximate each row's contribution to off-diagonal covariance.

        Sampling frequency pairs avoids materializing every off-diagonal pair
        when the number of columns is large.
        """

        p = rng.integers(0, n, size=n_pairs)
        q = rng.integers(0, n - 1, size=n_pairs)
        q = q + (q >= p)

        A_pair = np.zeros(n_pairs, dtype=np.complex128)
        for start in range(0, len(row_idx), chunk_rows):
            rows = row_idx[start:start + chunk_rows]
            Z = np.fft.fft(X[rows], axis=1) / np.sqrt(n)
            A_pair += np.sum(Z[:, p].conj() * Z[:, q], axis=0)
        A_pair /= len(row_idx)

        scores = np.empty(len(row_idx), dtype=float)
        for start in range(0, len(row_idx), chunk_rows):
            stop = min(start + chunk_rows, len(row_idx))
            rows = row_idx[start:stop]
            Z = np.fft.fft(X[rows], axis=1) / np.sqrt(n)
            scores[start:stop] = np.real(
                np.sum(Z[:, p] * Z[:, q].conj() * A_pair, axis=1)
            ) / n_pairs
        return scores

    history = []
    best_keep = keep.copy()
    best_kappa = -np.inf
    for iteration in range(max_iter):
        kappa = _kappa(keep)
        history.append({"iteration": iteration, "rows": len(keep), "kappa": kappa, "dropped": 0})
        if kappa > best_kappa:
            best_kappa = kappa
            best_keep = keep.copy()
        if kappa >= target_kappa or len(keep) <= min_rows:
            break

        scores = _badness(keep)
        n_drop = int(np.ceil(drop_fraction * len(keep)))
        n_drop = min(n_drop, len(keep) - min_rows)
        if n_drop <= 0:
            break

        # Drop the rows that reinforce current off-diagonal Fourier covariance.
        keep = np.delete(keep, np.argsort(scores)[-n_drop:])
        history[-1]["dropped"] = n_drop

    history = pd.DataFrame(history)
    if not history.empty:
        history["selected"] = history["kappa"].eq(best_kappa)

    return best_keep, history


def select_rows_by_psd_correlation(
    X,
    *,
    fs,
    min_corr=0.90,
    min_rows=None,
    max_drop_fraction=0.95,
    f_range=None,
    nperseg=None,
    chunk_rows=512,
    drop_fraction=0.05,
    max_iter=200,
):
    """Keep rows whose PSDs form a mutually similar spectral subset.

    This computes Welch PSDs directly from rows of ``X``, normalizes each PSD to unit
    length, and greedily removes the rows with the lowest PSD cosine similarity
    to another retained row until every retained pair is at least ``min_corr``
    or ``min_rows`` is reached.
    """
    X = _as_2d_float(X)
    m, _ = X.shape
    if min_rows is None:
        min_rows = max(2, int(np.ceil((1.0 - max_drop_fraction) * m)))
    min_rows = min(max(int(min_rows), 2), m)

    psd = _normalized_welch_psd(
        X,
        fs=float(fs),
        f_range=f_range,
        nperseg=nperseg,
    )

    keep = np.arange(m)
    history = []
    for iteration in range(int(max_iter) + 1):
        row_min = _row_min_similarity(psd, keep, chunk_rows=chunk_rows)
        min_pair = float(row_min.min()) if len(row_min) else np.nan
        history.append(
            {
                "iteration": iteration,
                "rows": len(keep),
                "min_psd_corr": min_pair,
                "median_min_psd_corr": float(np.median(row_min)) if len(row_min) else np.nan,
                "dropped": 0,
            }
        )
        if min_pair >= min_corr or len(keep) <= min_rows:
            break

        violators = np.flatnonzero(row_min < min_corr)
        n_drop = max(1, int(np.ceil(drop_fraction * len(keep))))
        n_drop = min(n_drop, len(violators), len(keep) - min_rows)
        if n_drop <= 0:
            break

        # The only criterion is PSD agreement: drop rows with the worst
        # minimum similarity to any other retained row.
        drop_local = violators[np.argsort(row_min[violators])[:n_drop]]
        keep = np.delete(keep, drop_local)
        history[-1]["dropped"] = int(n_drop)

    history = pd.DataFrame(history)
    history["selected"] = False
    if not history.empty:
        history.loc[history.index[-1], "selected"] = True
    return np.sort(keep), history


def filter_signal_rows_for_stationary_oscillation(
    matrices: dict[str, SignalMatrix],
    *,
    fs: float | None = None,
    osc_band: tuple[float, float] = (4.0, 12.0),
    psd_range: tuple[float, float] = (1.0, 120.0),
    min_osc_prominence: float = 2.0,
    min_osc_fraction: float = 0.08,
    min_half_psd_corr: float = 0.35,
    min_psd_reference_corr: float = 0.40,
    max_mean_robust_z: float = 8.0,
    max_log_std_robust_z: float = 6.0,
    min_rows_per_condition: int = 250,
    nperseg: int | None = None,
) -> tuple[dict[str, SignalMatrix], pd.DataFrame, pd.DataFrame]:
    """Condition-blind QC for stationary oscillatory rows.

    Rows are retained when they have a visible oscillatory peak, stable spectra
    across the first/second half of the window, and a PSD similar to the pooled
    condition-blind reference spectrum. No condition labels or classifier
    outcomes are used to set row scores.
    """

    if not matrices:
        return {}, pd.DataFrame(), pd.DataFrame()
    fs_values = {float(matrix.fs) for matrix in matrices.values()}
    if fs is None:
        if len(fs_values) != 1:
            raise ValueError(f"Mixed sampling rates require fs; got {sorted(fs_values)}.")
        fs = fs_values.pop()
    fs = float(fs)

    blocks = []
    provenance = []
    for condition, matrix in matrices.items():
        X = _as_2d_float(matrix.X)
        blocks.append(X)
        rows = matrix.rows.copy().reset_index(drop=True)
        rows["condition"] = condition
        rows["matrix_row"] = np.arange(X.shape[0], dtype=int)
        provenance.append(rows)
    X_all = np.vstack(blocks)
    rows_all = pd.concat(provenance, ignore_index=True)

    metrics = _stationary_oscillation_metrics(
        X_all,
        fs=fs,
        osc_band=osc_band,
        psd_range=psd_range,
        nperseg=nperseg,
    )
    metrics = pd.concat([rows_all.reset_index(drop=True), metrics], axis=1)
    metric_keep = (
        metrics["finite_row"]
        & (metrics["osc_prominence"] >= min_osc_prominence)
        & (metrics["osc_fraction"] >= min_osc_fraction)
        & (metrics["half_psd_corr"] >= min_half_psd_corr)
        & (metrics["psd_reference_corr"] >= min_psd_reference_corr)
        & (metrics["abs_mean_robust_z"] <= max_mean_robust_z)
        & (metrics["abs_log_std_robust_z"] <= max_log_std_robust_z)
    )
    metrics["qc_keep"] = metric_keep.to_numpy(dtype=bool)

    filtered = {}
    for condition, matrix in matrices.items():
        condition_metrics = metrics[metrics["condition"].eq(condition)].copy()
        keep = condition_metrics["qc_keep"].to_numpy(dtype=bool)
        if keep.sum() < min_rows_per_condition and len(condition_metrics) >= min_rows_per_condition:
            # Keep enough rows for covariance estimation by ranking only the
            # same condition-blind QC scores, not by labels or classifier fit.
            score = _stationary_oscillation_score(condition_metrics)
            fallback_local = np.argsort(score)[-int(min_rows_per_condition):]
            keep[fallback_local] = True
            condition_metrics.loc[condition_metrics.index[fallback_local], "qc_keep"] = True
            condition_metrics.loc[condition_metrics.index[fallback_local], "qc_fallback_keep"] = True
        else:
            condition_metrics["qc_fallback_keep"] = False
        metrics.loc[condition_metrics.index, ["qc_keep", "qc_fallback_keep"]] = condition_metrics[
            ["qc_keep", "qc_fallback_keep"]
        ]
        keep_indices = condition_metrics.loc[condition_metrics["qc_keep"], "matrix_row"].to_numpy(dtype=int)
        filtered[condition] = matrix.take(keep_indices)

    summary = (
        metrics
        .groupby("condition", as_index=False)
        .agg(
            rows_before=("qc_keep", "size"),
            rows_after=("qc_keep", "sum"),
            median_osc_prominence=("osc_prominence", "median"),
            median_osc_fraction=("osc_fraction", "median"),
            median_half_psd_corr=("half_psd_corr", "median"),
            median_psd_reference_corr=("psd_reference_corr", "median"),
            fallback_rows=("qc_fallback_keep", "sum"),
        )
    )
    summary["rows_dropped"] = summary["rows_before"] - summary["rows_after"]
    summary["fraction_kept"] = summary["rows_after"] / summary["rows_before"]
    return filtered, metrics, summary


def _as_2d_float(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be 2D: rows=observations, columns=time.")
    return X


def _normalized_welch_psd(X, *, fs, f_range=None, nperseg=None, eps=1e-12):
    X = _as_2d_float(X)
    if nperseg is None:
        nperseg = min(X.shape[1], int(round(fs)))
    else:
        nperseg = min(X.shape[1], int(nperseg))
    freqs, powers = welch(
        X,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 8,
        detrend=False,
        return_onesided=True,
        scaling="density",
        axis=1,
    )
    if f_range is not None:
        low, high = f_range
        keep = (freqs >= low) & (freqs <= high)
        powers = powers[:, keep]
    powers = np.maximum(powers.astype(np.float32, copy=False), eps)
    return powers / (np.linalg.norm(powers, axis=1, keepdims=True) + eps)


def _stationary_oscillation_metrics(
    X,
    *,
    fs,
    osc_band,
    psd_range,
    nperseg,
    eps=1e-12,
):
    X = _as_2d_float(X)
    row_mean = np.nanmean(X, axis=1)
    row_std = np.nanstd(X, axis=1)
    finite_row = np.isfinite(X).all(axis=1) & np.isfinite(row_mean) & np.isfinite(row_std) & (row_std > eps)

    psd_freqs, psd = _welch_psd(X, fs=fs, f_range=psd_range, nperseg=nperseg)
    psd_norm = psd / (np.linalg.norm(psd, axis=1, keepdims=True) + eps)
    reference = np.nanmedian(psd_norm, axis=0)
    reference = reference / (np.linalg.norm(reference) + eps)
    psd_reference_corr = psd_norm @ reference

    osc_mask = (psd_freqs >= osc_band[0]) & (psd_freqs <= osc_band[1])
    background_mask = ~osc_mask
    osc_power = np.sum(psd[:, osc_mask], axis=1)
    total_power = np.sum(psd, axis=1) + eps
    osc_fraction = osc_power / total_power
    osc_peak = np.max(psd[:, osc_mask], axis=1) if np.any(osc_mask) else np.full(X.shape[0], np.nan)
    background = np.nanmedian(psd[:, background_mask], axis=1) if np.any(background_mask) else total_power
    osc_prominence = osc_peak / (background + eps)

    midpoint = X.shape[1] // 2
    first = X[:, :midpoint]
    second = X[:, midpoint:]
    half_nperseg = min(first.shape[1], second.shape[1], nperseg or int(round(fs / 2)))
    _, psd_first = _welch_psd(first, fs=fs, f_range=psd_range, nperseg=half_nperseg)
    _, psd_second = _welch_psd(second, fs=fs, f_range=psd_range, nperseg=half_nperseg)
    first_norm = psd_first / (np.linalg.norm(psd_first, axis=1, keepdims=True) + eps)
    second_norm = psd_second / (np.linalg.norm(psd_second, axis=1, keepdims=True) + eps)
    half_psd_corr = np.sum(first_norm * second_norm, axis=1)

    mean_center = np.nanmedian(row_mean)
    mean_scale = _mad(row_mean) + eps
    log_std = np.log(row_std + eps)
    log_std_center = np.nanmedian(log_std)
    log_std_scale = _mad(log_std) + eps

    return pd.DataFrame(
        {
            "finite_row": finite_row,
            "row_mean": row_mean,
            "row_std": row_std,
            "abs_mean_robust_z": np.abs((row_mean - mean_center) / mean_scale),
            "abs_log_std_robust_z": np.abs((log_std - log_std_center) / log_std_scale),
            "osc_prominence": osc_prominence,
            "osc_fraction": osc_fraction,
            "half_psd_corr": half_psd_corr,
            "psd_reference_corr": psd_reference_corr,
        }
    )


def _welch_psd(X, *, fs, f_range, nperseg):
    X = _as_2d_float(X)
    if nperseg is None:
        nperseg = min(X.shape[1], int(round(fs)))
    else:
        nperseg = min(X.shape[1], int(nperseg))
    freqs, powers = welch(
        X,
        fs=fs,
        window="hann",
        nperseg=max(8, nperseg),
        noverlap=max(0, nperseg // 8),
        detrend=False,
        return_onesided=True,
        scaling="density",
        axis=1,
    )
    if f_range is not None:
        low, high = f_range
        keep = (freqs >= low) & (freqs <= high)
        freqs = freqs[keep]
        powers = powers[:, keep]
    return freqs, np.maximum(powers.astype(np.float32, copy=False), 1e-12)


def _mad(values):
    values = np.asarray(values, dtype=float)
    center = np.nanmedian(values)
    return 1.4826 * np.nanmedian(np.abs(values - center))


def _stationary_oscillation_score(metrics: pd.DataFrame) -> np.ndarray:
    columns = ["osc_prominence", "osc_fraction", "half_psd_corr", "psd_reference_corr"]
    score = np.zeros(len(metrics), dtype=float)
    for column in columns:
        values = metrics[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.sum() < 2:
            continue
        lo, hi = np.nanquantile(values[finite], [0.05, 0.95])
        score += np.clip((values - lo) / (hi - lo + 1e-12), 0, 1)
    score -= np.clip(metrics["abs_mean_robust_z"].to_numpy(dtype=float) / 8.0, 0, 1)
    score -= np.clip(metrics["abs_log_std_robust_z"].to_numpy(dtype=float) / 6.0, 0, 1)
    return score


def _row_min_similarity(vectors, indices, *, chunk_rows):
    selected = vectors[np.asarray(indices, dtype=int)]
    n_rows = selected.shape[0]
    if n_rows <= 1:
        return np.ones(n_rows, dtype=np.float32)

    row_min = np.empty(n_rows, dtype=np.float32)
    for start in range(0, n_rows, int(chunk_rows)):
        stop = min(start + int(chunk_rows), n_rows)
        sim = selected[start:stop] @ selected.T
        local_rows = np.arange(stop - start)
        global_rows = np.arange(start, stop)
        sim[local_rows, global_rows] = np.inf
        row_min[start:stop] = sim.min(axis=1)
    return row_min
