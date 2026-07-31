"""Downstream feature tables, statistics, and classifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import welch
from scipy.stats import ttest_ind
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import SignalMatrix


BYCYCLE_FEATURES = [
    "period_ms",
    "time_rdsym",
    "time_ptsym",
    "volt_rise",
    "volt_decay",
    "band_amp",
    "amp_fraction",
    "amp_consistency",
    "period_consistency",
    "monotonicity",
]
APERIODIC_FEATURES = [
    "aperiodic_ar1",
    "aperiodic_innovation_std",
    "aperiodic_hjorth_mobility",
    "aperiodic_hjorth_complexity",
]
PAC_FEATURES = ["theta_gamma_pac"]
DEFAULT_ROW_INDEX_COLUMNS = [
    "component",
    "condition",
    "matrix_row",
    "animal",
    "session",
    "session_id",
    "probe_id",
    "probe_name",
    "region",
    "channel_id",
    "raw_channel",
    "local_index",
    "trial_id",
    "trial_direction",
    "cooling_state",
    "error",
    "window_index",
    "window_chunk",
    "window_start_s",
    "window_stop_s",
    "window_start_time",
    "window_stop_time",
    "source_matrix_row",
    "time_block",
    "dandi_epoch_tag",
    "stimulus_name",
    "stimulus_block_index",
]


def aggregate_bycycle_row_features(
    features: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...] = BYCYCLE_FEATURES,
    *,
    index_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Collapse bycycle cycles to one median feature row per signal row."""

    if features.empty:
        return pd.DataFrame()
    index_columns = DEFAULT_ROW_INDEX_COLUMNS if index_columns is None else list(index_columns)
    available_index = [column for column in index_columns if column in features.columns]
    available_features = [column for column in feature_columns if column in features.columns]
    if not available_index:
        raise ValueError("No grouping columns are present in the bycycle feature table.")
    if not available_features:
        raise ValueError("No requested bycycle feature columns are present.")

    return (
        features
        .groupby(available_index, dropna=False)[available_features]
        .median()
        .reset_index()
    )


def compute_aperiodic_row_features(
    matrices: dict[str, SignalMatrix],
    *,
    feature_prefix: str = "aperiodic",
    provenance_columns: list[str] | tuple[str, ...] | None = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Estimate simple aperiodic time-series features for each matrix row.

    The AR(1) coefficient is fit directly in time after row demeaning:
    x[t + 1] ~= phi * x[t]. This is deliberately separate from the ARPSD
    spectral fit; it describes the decomposed aperiodic time series itself.
    """

    provenance_columns = [] if provenance_columns is None else list(provenance_columns)
    tables = []
    for condition, matrix in matrices.items():
        X = np.asarray(matrix.X, dtype=float)
        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError("Each matrix must be 2D with at least two time samples.")

        X_centered = X - np.nanmean(X, axis=1, keepdims=True)
        lag = X_centered[:, :-1]
        lead = X_centered[:, 1:]
        denom = np.sum(lag * lag, axis=1)
        ar1 = np.divide(
            np.sum(lag * lead, axis=1),
            denom + eps,
            out=np.full(X.shape[0], np.nan, dtype=float),
            where=denom > eps,
        )
        innovation = lead - ar1[:, None] * lag
        diff1 = np.diff(X_centered, axis=1)
        diff2 = np.diff(diff1, axis=1) if X_centered.shape[1] > 2 else np.empty((X.shape[0], 0))
        row_var = np.nanvar(X_centered, axis=1, ddof=1)
        diff1_var = np.nanvar(diff1, axis=1, ddof=1)
        diff2_var = (
            np.nanvar(diff2, axis=1, ddof=1)
            if diff2.shape[1] > 1
            else np.full(X.shape[0], np.nan, dtype=float)
        )
        hjorth_mobility = np.sqrt(np.divide(diff1_var, row_var + eps))
        diff_mobility = np.sqrt(np.divide(diff2_var, diff1_var + eps))
        hjorth_complexity = np.divide(
            diff_mobility,
            hjorth_mobility + eps,
            out=np.full(X.shape[0], np.nan, dtype=float),
            where=np.isfinite(hjorth_mobility),
        )
        signs = np.signbit(X_centered)
        zero_crossing_rate = np.nanmean(signs[:, 1:] != signs[:, :-1], axis=1)

        table = pd.DataFrame(
            {
                "condition": condition,
                "matrix_row": np.arange(X.shape[0], dtype=int),
                f"{feature_prefix}_ar1": ar1,
                f"{feature_prefix}_std": np.nanstd(X_centered, axis=1, ddof=1),
                f"{feature_prefix}_rms": np.sqrt(np.nanmean(X_centered * X_centered, axis=1)),
                f"{feature_prefix}_innovation_std": np.nanstd(innovation, axis=1, ddof=1),
                f"{feature_prefix}_hjorth_mobility": hjorth_mobility,
                f"{feature_prefix}_hjorth_complexity": hjorth_complexity,
                f"{feature_prefix}_line_length": np.nanmean(np.abs(diff1), axis=1),
                f"{feature_prefix}_zero_crossing_rate": zero_crossing_rate,
            }
        )
        for column in provenance_columns:
            if column in matrix.rows.columns and column not in table.columns:
                table[column] = matrix.rows[column].to_numpy()
        tables.append(table)

    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def compute_bandpower_row_features(
    matrices: dict[str, SignalMatrix],
    bands: dict[str, tuple[float, float]],
    *,
    reference_band: tuple[float, float] = (1.0, 120.0),
    provenance_columns: list[str] | tuple[str, ...] | None = None,
    nperseg: int | None = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Compute row-wise target-band power relative to nearby broadband power."""

    provenance_columns = [] if provenance_columns is None else list(provenance_columns)
    tables = []
    for condition, matrix in matrices.items():
        X = np.asarray(matrix.X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Each matrix must be 2D: rows=observations, columns=time.")
        fs = float(matrix.fs)
        seg = min(X.shape[1], int(round(fs)) if nperseg is None else int(nperseg))
        freqs, powers = welch(
            X - np.nanmean(X, axis=1, keepdims=True),
            fs=fs,
            window="hann",
            nperseg=seg,
            noverlap=seg // 8,
            detrend=False,
            return_onesided=True,
            scaling="density",
            axis=1,
        )
        reference_mask = (freqs >= reference_band[0]) & (freqs <= reference_band[1])
        reference_power = np.trapz(powers[:, reference_mask], freqs[reference_mask], axis=1) + eps
        table = pd.DataFrame(
            {
                "condition": condition,
                "matrix_row": np.arange(X.shape[0], dtype=int),
            }
        )
        for name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            outside_mask = reference_mask & ~band_mask
            band_power = np.trapz(powers[:, band_mask], freqs[band_mask], axis=1) if np.any(band_mask) else np.zeros(X.shape[0])
            outside_power = (
                np.trapz(powers[:, outside_mask], freqs[outside_mask], axis=1)
                if np.any(outside_mask)
                else np.zeros(X.shape[0])
            )
            band_peak = np.nanmax(powers[:, band_mask], axis=1) if np.any(band_mask) else np.full(X.shape[0], np.nan)
            outside_median = (
                np.nanmedian(powers[:, outside_mask], axis=1)
                if np.any(outside_mask)
                else np.full(X.shape[0], np.nan)
            )
            table[f"{name}_power_fraction"] = band_power / reference_power
            table[f"{name}_power_log10_ratio"] = np.log10((band_power + eps) / (outside_power + eps))
            table[f"{name}_peak_prominence"] = band_peak / (outside_median + eps)
        for column in provenance_columns:
            if column in matrix.rows.columns and column not in table.columns:
                table[column] = matrix.rows[column].to_numpy()
        tables.append(table)

    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def compute_pac_row_features(
    matrices: dict[str, SignalMatrix],
    *,
    component: str,
    phase_band: tuple[float, float] = (4.0, 12.0),
    amplitude_band: tuple[float, float] = (30.0, 80.0),
    feature_name: str = "theta_gamma_pac",
    filter_order: int = 4,
    provenance_columns: list[str] | tuple[str, ...] | None = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Compute row-wise phase-amplitude coupling by modulation vector length.

    This is feature extraction, not model preprocessing: the input matrices are
    not changed. Theta phase and gamma amplitude are estimated with temporary
    bandpass filters, then summarized as ``abs(mean(A_gamma * exp(1j*phi_theta)))
    / mean(A_gamma)`` per row.
    """

    from scipy.signal import butter, hilbert, sosfiltfilt

    provenance_columns = [] if provenance_columns is None else list(provenance_columns)
    tables = []
    for condition, matrix in matrices.items():
        fs = float(matrix.fs)
        nyquist = fs / 2
        if not (0 < phase_band[0] < phase_band[1] < nyquist):
            raise ValueError(f"Invalid phase_band={phase_band} for fs={fs}.")
        if not (0 < amplitude_band[0] < amplitude_band[1] < nyquist):
            raise ValueError(f"Invalid amplitude_band={amplitude_band} for fs={fs}.")

        X = np.asarray(matrix.X, dtype=float)
        X = X - np.nanmean(X, axis=1, keepdims=True)
        phase_sos = butter(filter_order, phase_band, btype="bandpass", fs=fs, output="sos")
        amp_sos = butter(filter_order, amplitude_band, btype="bandpass", fs=fs, output="sos")
        theta = sosfiltfilt(phase_sos, X, axis=1)
        gamma = sosfiltfilt(amp_sos, X, axis=1)
        theta_phase = np.angle(hilbert(theta, axis=1))
        gamma_amp = np.abs(hilbert(gamma, axis=1))
        pac = np.abs(np.nanmean(gamma_amp * np.exp(1j * theta_phase), axis=1))
        pac /= np.nanmean(gamma_amp, axis=1) + eps

        table = pd.DataFrame(
            {
                "component": component,
                "condition": condition,
                "matrix_row": np.arange(X.shape[0], dtype=int),
                feature_name: pac,
            }
        )
        for column in provenance_columns:
            if column in matrix.rows.columns and column not in table.columns:
                table[column] = matrix.rows[column].to_numpy()
        tables.append(table)

    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def build_downstream_feature_table(
    bycycle_features: pd.DataFrame,
    *,
    bycycle_feature_columns: list[str] | tuple[str, ...] = BYCYCLE_FEATURES,
    aperiodic_matrices: dict[str, SignalMatrix] | None = None,
    aperiodic_feature_columns: list[str] | tuple[str, ...] = APERIODIC_FEATURES,
    index_columns: list[str] | tuple[str, ...] | None = None,
    aperiodic_provenance_columns: list[str] | tuple[str, ...] | None = None,
    merge_columns: list[str] | tuple[str, ...] = ("condition", "matrix_row"),
) -> pd.DataFrame:
    """Build one row per signal/window with bycycle and optional aperiodic features."""

    row_features = aggregate_bycycle_row_features(
        bycycle_features,
        bycycle_feature_columns,
        index_columns=index_columns,
    )
    if aperiodic_matrices is None or row_features.empty:
        return row_features

    aperiodic_features = compute_aperiodic_row_features(
        aperiodic_matrices,
        provenance_columns=aperiodic_provenance_columns,
    )
    available_merge = [
        column
        for column in merge_columns
        if column in row_features.columns and column in aperiodic_features.columns
    ]
    if not available_merge:
        raise ValueError("No shared merge columns between bycycle rows and aperiodic rows.")

    available_aperiodic = [column for column in aperiodic_feature_columns if column in aperiodic_features.columns]
    if not available_aperiodic:
        raise ValueError("No requested aperiodic feature columns were computed.")

    return row_features.merge(
        aperiodic_features[available_merge + available_aperiodic],
        on=available_merge,
        how="left",
        validate="many_to_one",
    )


def cohens_d(a, b) -> float:
    """Return pooled-standard-deviation Cohen's d for two vectors."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = np.sqrt((np.nanvar(a, ddof=1) + np.nanvar(b, ddof=1)) / 2)
    return float((np.nanmean(a) - np.nanmean(b)) / (pooled + 1e-12))


def feature_t_tests(
    row_features: pd.DataFrame,
    *,
    groups: tuple[str, str] | list[str],
    feature_columns: list[str] | tuple[str, ...],
    components: tuple[str, ...] | list[str] = ("raw", "periodic"),
    condition_column: str = "condition",
) -> pd.DataFrame:
    """Run Welch t-tests and effect sizes for each component/feature."""

    group_a, group_b = tuple(groups)
    rows = []
    for component in components:
        component_rows = row_features[row_features["component"].eq(component)]
        rows_a = component_rows[component_rows[condition_column].eq(group_a)]
        rows_b = component_rows[component_rows[condition_column].eq(group_b)]
        for feature in feature_columns:
            values_a = rows_a[feature].replace([np.inf, -np.inf], np.nan).dropna()
            values_b = rows_b[feature].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values_a) >= 2 and len(values_b) >= 2:
                _, p_value = ttest_ind(values_a, values_b, equal_var=False)
                effect = cohens_d(values_a, values_b)
            else:
                p_value = np.nan
                effect = np.nan
            rows.append(
                {
                    "component": component,
                    "feature": feature,
                    f"{group_a}_mean": float(values_a.mean()) if len(values_a) else np.nan,
                    f"{group_b}_mean": float(values_b.mean()) if len(values_b) else np.nan,
                    "cohens_d": effect,
                    "abs_cohens_d": abs(effect) if np.isfinite(effect) else np.nan,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                    "n_a": int(len(values_a)),
                    "n_b": int(len(values_b)),
                }
            )
    return pd.DataFrame(rows)


def _fold_feature_diagnostics(
    *,
    component: str,
    fold: int,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    feature_columns: list[str],
    condition_column: str,
    group_column: str,
    positive_group: str,
    negative_group: str,
    fold_model,
) -> list[dict]:
    """Summarize feature instability for one classifier fold."""

    scaler = fold_model.named_steps["standardscaler"]
    classifier = fold_model.named_steps["logisticregression"]
    X_test_z = scaler.transform(test_features[feature_columns].to_numpy())
    coefs = classifier.coef_[0]

    rows = []
    for feature_index, feature in enumerate(feature_columns):
        train_negative = train_features.loc[
            train_features[condition_column].eq(negative_group),
            feature,
        ].replace([np.inf, -np.inf], np.nan).dropna()
        train_positive = train_features.loc[
            train_features[condition_column].eq(positive_group),
            feature,
        ].replace([np.inf, -np.inf], np.nan).dropna()
        test_negative = test_features.loc[
            test_features[condition_column].eq(negative_group),
            feature,
        ].replace([np.inf, -np.inf], np.nan).dropna()
        test_positive = test_features.loc[
            test_features[condition_column].eq(positive_group),
            feature,
        ].replace([np.inf, -np.inf], np.nan).dropna()
        train_delta = float(train_positive.mean() - train_negative.mean())
        test_delta = float(test_positive.mean() - test_negative.mean())
        z = X_test_z[:, feature_index]
        rows.append(
            {
                "component": component,
                "fold": int(fold),
                "feature": feature,
                "positive_group": positive_group,
                "negative_group": negative_group,
                "train_positive_minus_negative": train_delta,
                "test_positive_minus_negative": test_delta,
                "effect_sign_flip": bool(np.sign(train_delta) != np.sign(test_delta)),
                "standardized_logistic_coef": float(coefs[feature_index]),
                "abs_coef_if_flip": float(abs(coefs[feature_index]))
                if np.sign(train_delta) != np.sign(test_delta)
                else 0.0,
                "test_abs_z_p95": float(np.nanpercentile(np.abs(z), 95)),
                "test_abs_z_max": float(np.nanmax(np.abs(z))),
                "test_frac_abs_z_gt_3": float(np.mean(np.abs(z) > 3)),
                "test_frac_abs_z_gt_5": float(np.mean(np.abs(z) > 5)),
                "held_out_groups": ", ".join(
                    sorted(map(str, pd.unique(test_features[group_column])))
                )
                if group_column in test_features
                else "",
            }
        )
    return rows


def _group_balanced_sample_weight(
    frame: pd.DataFrame,
    *,
    condition_column: str,
    group_column: str,
    mode: str | None,
) -> np.ndarray | None:
    """Return row weights so grouped CV training does not over-count large sessions."""

    if mode is None or mode == "none":
        return None
    if mode not in {"group", "group_condition"}:
        raise ValueError("sample_weight_mode must be None, 'none', 'group', or 'group_condition'.")
    columns = [group_column] if mode == "group" else [group_column, condition_column]
    block_sizes = frame.groupby(columns, dropna=False)[condition_column].transform("size").to_numpy(dtype=float)
    return 1.0 / np.maximum(block_sizes, 1.0)


def _select_group_stable_features(
    train_features: pd.DataFrame,
    feature_columns: list[str],
    *,
    condition_column: str,
    group_column: str,
    positive_group: str,
    negative_group: str,
    min_groups: int,
    min_sign_consistency: float,
    min_abs_delta: float,
) -> tuple[list[str], pd.DataFrame]:
    """Select features whose condition effect has a stable sign in training groups.

    The selector uses only the training fold. For each feature, it computes
    positive-minus-negative means within each training group that has both
    conditions. A feature is kept only when enough groups agree with the
    training pooled direction. This prevents one unstable session-level effect
    from dominating the held-out classifier.
    """

    rows = []
    selected = []
    if group_column not in train_features.columns:
        raise ValueError(f"feature_stability_group_column={group_column!r} is not present in the feature table.")
    for feature in feature_columns:
        pooled_positive = (
            train_features.loc[train_features[condition_column].eq(positive_group), feature]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        pooled_negative = (
            train_features.loc[train_features[condition_column].eq(negative_group), feature]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        pooled_delta = (
            float(pooled_positive.mean() - pooled_negative.mean())
            if len(pooled_positive) and len(pooled_negative)
            else np.nan
        )
        group_deltas = []
        for _, group_rows in train_features.groupby(group_column, dropna=False):
            positive = (
                group_rows.loc[group_rows[condition_column].eq(positive_group), feature]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            negative = (
                group_rows.loc[group_rows[condition_column].eq(negative_group), feature]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(positive) and len(negative):
                delta = float(positive.mean() - negative.mean())
                if np.isfinite(delta) and abs(delta) >= min_abs_delta:
                    group_deltas.append(delta)

        deltas = np.asarray(group_deltas, dtype=float)
        signs = np.sign(deltas)
        signs = signs[signs != 0]
        if np.isfinite(pooled_delta) and abs(pooled_delta) >= min_abs_delta and pooled_delta != 0:
            reference_sign = float(np.sign(pooled_delta))
        elif len(signs):
            reference_sign = 1.0 if np.mean(signs > 0) >= 0.5 else -1.0
        else:
            reference_sign = np.nan
        sign_consistency = (
            float(np.mean(signs == reference_sign))
            if len(signs) and np.isfinite(reference_sign)
            else np.nan
        )
        keep = bool(
            len(signs) >= min_groups
            and np.isfinite(sign_consistency)
            and sign_consistency >= min_sign_consistency
        )
        if keep:
            selected.append(feature)
        rows.append(
            {
                "feature": feature,
                "keep": keep,
                "stability_group_column": group_column,
                "n_stability_groups": int(len(signs)),
                "train_positive_minus_negative": pooled_delta,
                "reference_sign": reference_sign,
                "sign_consistency": sign_consistency,
                "min_groups": int(min_groups),
                "min_sign_consistency": float(min_sign_consistency),
                "min_abs_delta": float(min_abs_delta),
            }
        )

    return selected, pd.DataFrame(rows)


def _positive_class_scores(fold_model, X: np.ndarray) -> np.ndarray:
    """Return predicted probability for encoded class 1 with explicit checks."""

    classifier = fold_model.named_steps["logisticregression"]
    classes = np.asarray(classifier.classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one positive class 1; got classes={classes.tolist()}.")
    return fold_model.predict_proba(X)[:, int(matches[0])]


def grouped_logistic_classifier(
    row_features: pd.DataFrame,
    *,
    groups: tuple[str, str] | list[str],
    feature_columns: list[str] | tuple[str, ...],
    group_column: str,
    positive_group: str | None = None,
    components: tuple[str, ...] | list[str] = ("raw", "periodic"),
    condition_column: str = "condition",
    max_splits: int = 5,
    min_splits: int = 2,
    random_state: int = 0,
    C: float = 1.0,
    sample_weight_mode: str | None = None,
    feature_stability_group_column: str | None = None,
    feature_stability_min_groups: int = 4,
    feature_stability_min_sign_consistency: float = 0.9,
    feature_stability_min_abs_delta: float = 0.0,
    n_jobs: int | None = 1,
    joblib_prefer: str | None = "threads",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train a standardized logistic classifier with grouped cross-validation."""

    groups = tuple(groups)
    feature_columns = list(feature_columns)
    positive_group = groups[0] if positive_group is None else positive_group
    score_rows = []
    coef_rows = []
    prediction_rows = []
    feature_diagnostic_rows = []
    feature_selection_rows = []
    for component in components:
        data = row_features[row_features["component"].eq(component)].copy()
        data = data[data[condition_column].isin(groups)]
        missing_features = [feature for feature in feature_columns if feature not in data.columns]
        if missing_features:
            raise RuntimeError(
                f"Missing requested classifier features for component {component!r}: {missing_features}. "
                f"Available columns: {sorted(data.columns)}"
            )
        data = data.dropna(subset=[*feature_columns, group_column, condition_column])
        if data.empty:
            raise RuntimeError(f"No rows remain for component {component!r}.")

        groups_per_class = data.groupby(condition_column)[group_column].nunique()
        if len(groups_per_class) < 2:
            raise RuntimeError(f"Need both groups for component {component!r}.")
        n_splits = int(min(max_splits, groups_per_class.min()))
        if n_splits < min_splits:
            raise RuntimeError(
                f"Need at least {min_splits} {group_column!r} groups per condition; "
                f"got {groups_per_class.to_dict()}."
            )
        negative_group = next((group for group in groups if group != positive_group), groups[0])

        y = data[condition_column].eq(positive_group).astype(int).to_numpy()
        cv_groups = data[group_column].to_numpy()
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(enumerate(cv.split(np.zeros((len(data), 1)), y, groups=cv_groups)))

        def fit_fold(fold: int, train_index: np.ndarray, test_index: np.ndarray) -> dict[str, list[dict]]:
            fold_score_rows = []
            fold_coef_rows = []
            fold_prediction_rows = []
            fold_feature_diagnostic_rows = []
            fold_feature_selection_rows = []

            train_features = data.iloc[train_index].copy()
            test_features = data.iloc[test_index].copy()
            if feature_stability_group_column is not None:
                selected_features, selection = _select_group_stable_features(
                    train_features,
                    feature_columns,
                    condition_column=condition_column,
                    group_column=feature_stability_group_column,
                    positive_group=positive_group,
                    negative_group=negative_group,
                    min_groups=feature_stability_min_groups,
                    min_sign_consistency=feature_stability_min_sign_consistency,
                    min_abs_delta=feature_stability_min_abs_delta,
                )
                selection.insert(0, "component", component)
                selection.insert(1, "fold", fold)
                fold_feature_selection_rows.extend(selection.to_dict("records"))
                if not selected_features:
                    raise RuntimeError(
                        f"No stable features remain for {component!r} fold {fold}. "
                        "Lower feature_stability_min_sign_consistency or inspect feature_selection attrs."
                    )
            else:
                selected_features = feature_columns

            train_features = train_features.dropna(subset=[*selected_features, condition_column])
            test_features = test_features.dropna(subset=[*selected_features, condition_column])
            if train_features.empty or test_features.empty:
                return {
                    "score_rows": fold_score_rows,
                    "coef_rows": fold_coef_rows,
                    "prediction_rows": fold_prediction_rows,
                    "feature_diagnostic_rows": fold_feature_diagnostic_rows,
                    "feature_selection_rows": fold_feature_selection_rows,
                }
            X_train = train_features[selected_features].to_numpy()
            X_test = test_features[selected_features].to_numpy()
            y_train = train_features[condition_column].eq(positive_group).astype(int).to_numpy()
            y_test = test_features[condition_column].eq(positive_group).astype(int).to_numpy()
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                return {
                    "score_rows": fold_score_rows,
                    "coef_rows": fold_coef_rows,
                    "prediction_rows": fold_prediction_rows,
                    "feature_diagnostic_rows": fold_feature_diagnostic_rows,
                    "feature_selection_rows": fold_feature_selection_rows,
                }

            fold_model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", C=C),
            )
            sample_weight = _group_balanced_sample_weight(
                train_features,
                condition_column=condition_column,
                group_column=group_column,
                mode=sample_weight_mode,
            )
            fit_kwargs = (
                {"logisticregression__sample_weight": sample_weight}
                if sample_weight is not None
                else {}
            )
            fold_model.fit(X_train, y_train, **fit_kwargs)
            y_score = _positive_class_scores(fold_model, X_test)
            y_pred = (y_score >= 0.5).astype(int)
            score = balanced_accuracy_score(y_test, y_pred)
            fold_auc = (
                roc_auc_score(y_test, y_score)
                if len(np.unique(y_test)) == 2
                else np.nan
            )
            fold_feature_diagnostic_rows.extend(
                _fold_feature_diagnostics(
                    component=component,
                    fold=fold,
                    train_features=train_features,
                    test_features=test_features,
                    feature_columns=selected_features,
                    condition_column=condition_column,
                    group_column=group_column,
                    positive_group=positive_group,
                    negative_group=negative_group,
                    fold_model=fold_model,
                )
            )
            fold_score_rows.append(
                {
                    "component": component,
                    "fold": fold,
                    "balanced_accuracy": float(score),
                    "roc_auc": float(fold_auc) if np.isfinite(fold_auc) else np.nan,
                    "n_test": int(len(test_features)),
                    "held_out_group_column": group_column,
                    "positive_group": positive_group,
                    "n_features": int(len(selected_features)),
                }
            )
            for row_index, score_value, true_value, group_value in zip(
                test_features.index.to_numpy(),
                y_score,
                y_test,
                test_features[group_column].to_numpy(),
            ):
                fold_prediction_rows.append(
                    {
                        "component": component,
                        "fold": fold,
                        "row_index": int(row_index),
                        "group": group_value,
                        "y_true": int(true_value),
                        "condition": positive_group if int(true_value) == 1 else negative_group,
                        "y_score": float(score_value),
                        "positive_group": positive_group,
                    }
                )
            coefs = fold_model.named_steps["logisticregression"].coef_[0]
            for feature, coef in zip(selected_features, coefs):
                fold_coef_rows.append(
                    {
                        "component": component,
                        "fold": fold,
                        "feature": feature,
                        "standardized_logistic_coef": float(coef),
                        "positive_group": positive_group,
                    }
                )

            return {
                "score_rows": fold_score_rows,
                "coef_rows": fold_coef_rows,
                "prediction_rows": fold_prediction_rows,
                "feature_diagnostic_rows": fold_feature_diagnostic_rows,
                "feature_selection_rows": fold_feature_selection_rows,
            }

        fold_results = Parallel(n_jobs=n_jobs, prefer=joblib_prefer)(
            delayed(fit_fold)(fold, train_index, test_index)
            for fold, (train_index, test_index) in splits
        )
        for result in fold_results:
            score_rows.extend(result["score_rows"])
            coef_rows.extend(result["coef_rows"])
            prediction_rows.extend(result["prediction_rows"])
            feature_diagnostic_rows.extend(result["feature_diagnostic_rows"])
            feature_selection_rows.extend(result["feature_selection_rows"])

    coefficients = pd.DataFrame(coef_rows)
    if not coefficients.empty:
        coefficients = (
            coefficients
            .groupby(["component", "feature", "positive_group"], as_index=False)
            ["standardized_logistic_coef"]
            .mean()
        )
    scores = pd.DataFrame(score_rows)
    scores.attrs["oof_predictions"] = pd.DataFrame(prediction_rows)
    scores.attrs["fold_feature_diagnostics"] = pd.DataFrame(feature_diagnostic_rows)
    scores.attrs["feature_selection"] = pd.DataFrame(feature_selection_rows)
    return scores, coefficients


def pooled_decomposition_logistic_classifier(
    matrices: dict[str, SignalMatrix],
    split_rows: pd.DataFrame,
    *,
    groups: tuple[str, str] | list[str],
    feature_columns: list[str] | tuple[str, ...],
    group_column: str,
    bycycle_band: tuple[float, float] | dict[str, tuple[float, float]],
    extra_bycycle_bands: dict[str, tuple[float, float]] | None = None,
    bycycle_thresholds=None,
    bycycle_burst_only: bool = False,
    bycycle_min_cycles_per_row: int = 1,
    pac_phase_band: tuple[float, float] | None = None,
    pac_amplitude_band: tuple[float, float] | None = None,
    provenance_columns: list[str] | tuple[str, ...] | None = None,
    decomposition_kwargs: dict | None = None,
    periodic_transform=None,
    extra_bycycle_transforms: dict[str, object] | None = None,
    component: str = "periodic",
    positive_group: str | None = None,
    condition_column: str = "condition",
    matrix_row_column: str = "matrix_row",
    max_splits: int = 5,
    min_splits: int = 2,
    random_state: int = 0,
    C: float = 1.0,
    sample_weight_mode: str | None = None,
    feature_stability_group_column: str | None = None,
    feature_stability_min_groups: int = 4,
    feature_stability_min_sign_consistency: float = 0.9,
    feature_stability_min_abs_delta: float = 0.0,
    n_jobs: int | None = 1,
    joblib_prefer: str | None = "threads",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classifier using train-only pooled decomposition inside each CV fold.

    This avoids two leaks that are easy to miss: held-out rows are not used to
    fit the eigenspectrum/AR decomposition, and the true condition label is not
    used to choose a condition-specific decomposition model. Each fold pools the
    training rows across conditions, fits one unsupervised decomposition model,
    transforms train/test rows with that model, then extracts periodic waveform
    and aperiodic-residual features.
    """

    from .bycycle import measure_bycycle_features
    from .spectral import decompose_with_fitted_model, fit_pooled_decomposition_model

    groups = tuple(groups)
    feature_columns = list(feature_columns)
    positive_group = groups[0] if positive_group is None else positive_group
    provenance_columns = [] if provenance_columns is None else list(provenance_columns)
    decomposition_kwargs = {} if decomposition_kwargs is None else dict(decomposition_kwargs)
    extra_bycycle_bands = {} if extra_bycycle_bands is None else dict(extra_bycycle_bands)
    extra_bycycle_transforms = {} if extra_bycycle_transforms is None else dict(extra_bycycle_transforms)
    pac_features = [feature for feature in feature_columns if feature in PAC_FEATURES]
    waveform_features = [
        feature
        for feature in feature_columns
        if not feature.startswith("aperiodic_") and feature not in pac_features
    ]
    theta_waveform_features = [feature for feature in waveform_features if feature in BYCYCLE_FEATURES]
    aperiodic_features = [feature for feature in feature_columns if feature.startswith("aperiodic_")]
    if pac_features:
        if pac_phase_band is None:
            pac_phase_band = bycycle_band if not isinstance(bycycle_band, dict) else (4.0, 12.0)
        if pac_amplitude_band is None:
            pac_amplitude_band = extra_bycycle_bands.get("gamma", (30.0, 100.0))

    data = split_rows.copy()
    data = data[data[condition_column].isin(groups)]
    data = data.dropna(subset=[condition_column, group_column, matrix_row_column])
    data = data.drop_duplicates(subset=[condition_column, matrix_row_column])
    if data.empty:
        raise RuntimeError("No split rows remain for pooled decomposition classifier.")

    groups_per_class = data.groupby(condition_column)[group_column].nunique()
    if len(groups_per_class) < 2:
        raise RuntimeError("Need both groups for pooled decomposition classifier.")
    n_splits = int(min(max_splits, groups_per_class.min()))
    if n_splits < min_splits:
        raise RuntimeError(
            f"Need at least {min_splits} {group_column!r} groups per condition; "
            f"got {groups_per_class.to_dict()}."
        )
    negative_group = next((group for group in groups if group != positive_group), groups[0])

    y = data[condition_column].eq(positive_group).astype(int).to_numpy()
    cv_groups = data[group_column].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    score_rows = []
    coef_rows = []
    prediction_rows = []
    feature_diagnostic_rows = []
    feature_selection_rows = []

    def subset_matrices(rows: pd.DataFrame) -> dict[str, SignalMatrix]:
        out = {}
        for condition in groups:
            condition_rows = rows[rows[condition_column].eq(condition)]
            indices = condition_rows[matrix_row_column].astype(int).to_numpy()
            if len(indices):
                out[condition] = matrices[condition].take(indices)
        return out

    def fold_feature_table(
        aperiodic_matrices: dict[str, SignalMatrix],
        periodic_matrices: dict[str, SignalMatrix],
    ) -> pd.DataFrame:
        if periodic_transform is not None:
            periodic_matrices = {
                condition: periodic_transform(matrix)
                for condition, matrix in periodic_matrices.items()
            }
        bycycle = measure_bycycle_features(
            periodic_matrices,
            bycycle_band,
            component=component,
            thresholds=bycycle_thresholds,
            provenance_columns=provenance_columns,
            burst_only=bycycle_burst_only,
            min_cycles_per_row=bycycle_min_cycles_per_row,
        )
        if bycycle.empty:
            return pd.DataFrame()
        row_features = build_downstream_feature_table(
            bycycle,
            bycycle_feature_columns=theta_waveform_features,
        )
        for band_name, band_range in extra_bycycle_bands.items():
            prefix = f"{band_name}_"
            requested = [feature for feature in waveform_features if feature.startswith(prefix)]
            if not requested:
                continue
            source_features = [feature[len(prefix):] for feature in requested]
            source_features = [feature for feature in source_features if feature in BYCYCLE_FEATURES]
            if not source_features:
                continue
            band_matrices = periodic_matrices
            band_transform = extra_bycycle_transforms.get(band_name)
            if band_transform is not None:
                band_matrices = {
                    condition: band_transform(matrix)
                    for condition, matrix in periodic_matrices.items()
                }
            band_bycycle = measure_bycycle_features(
                band_matrices,
                band_range,
                component=component,
                thresholds=bycycle_thresholds,
                provenance_columns=provenance_columns,
                burst_only=bycycle_burst_only,
                min_cycles_per_row=bycycle_min_cycles_per_row,
            )
            if band_bycycle.empty:
                continue
            band_rows = build_downstream_feature_table(
                band_bycycle,
                bycycle_feature_columns=source_features,
            )
            rename = {feature: f"{prefix}{feature}" for feature in source_features if feature in band_rows}
            band_rows = band_rows.rename(columns=rename)
            band_columns = list(rename.values())
            row_features = row_features.merge(
                band_rows[["component", condition_column, matrix_row_column, *band_columns]],
                on=["component", condition_column, matrix_row_column],
                how="left",
                validate="one_to_one",
            )
        if aperiodic_features:
            aperiodic = compute_aperiodic_row_features(
                aperiodic_matrices,
                provenance_columns=provenance_columns,
            )
            aperiodic["component"] = component
            row_features = row_features.merge(
                aperiodic[["component", condition_column, matrix_row_column, *aperiodic_features]],
                on=["component", condition_column, matrix_row_column],
                how="left",
                validate="one_to_one",
            )
        if pac_features:
            pac = compute_pac_row_features(
                periodic_matrices,
                component=component,
                phase_band=pac_phase_band,
                amplitude_band=pac_amplitude_band,
                provenance_columns=provenance_columns,
            )
            available_pac = [feature for feature in pac_features if feature in pac.columns]
            row_features = row_features.merge(
                pac[["component", condition_column, matrix_row_column, *available_pac]],
                on=["component", condition_column, matrix_row_column],
                how="left",
                validate="one_to_one",
            )
        return row_features

    splits = list(enumerate(splitter.split(np.zeros((len(data), 1)), y, groups=cv_groups)))

    def fit_fold(fold: int, train_index: np.ndarray, test_index: np.ndarray) -> dict[str, list[dict]]:
        fold_score_rows = []
        fold_coef_rows = []
        fold_prediction_rows = []
        fold_feature_diagnostic_rows = []
        fold_feature_selection_rows = []

        train_meta = data.iloc[train_index]
        test_meta = data.iloc[test_index]
        train_matrices = subset_matrices(train_meta)
        test_matrices = subset_matrices(test_meta)
        decomposition_model = fit_pooled_decomposition_model(train_matrices, **decomposition_kwargs)

        train_aperiodic = {}
        train_periodic = {}
        test_aperiodic = {}
        test_periodic = {}
        for condition, matrix in train_matrices.items():
            train_aperiodic[condition], train_periodic[condition] = decompose_with_fitted_model(
                decomposition_model,
                matrix,
                condition=condition,
            )
        for condition, matrix in test_matrices.items():
            test_aperiodic[condition], test_periodic[condition] = decompose_with_fitted_model(
                decomposition_model,
                matrix,
                condition=condition,
            )

        train_features = fold_feature_table(train_aperiodic, train_periodic)
        test_features = fold_feature_table(test_aperiodic, test_periodic)
        train_features = train_features.dropna(subset=[condition_column])
        test_features = test_features.dropna(subset=[condition_column])
        if train_features.empty or test_features.empty:
            return {
                "score_rows": fold_score_rows,
                "coef_rows": fold_coef_rows,
                "prediction_rows": fold_prediction_rows,
                "feature_diagnostic_rows": fold_feature_diagnostic_rows,
                "feature_selection_rows": fold_feature_selection_rows,
            }
        missing_features = [
            feature
            for feature in feature_columns
            if feature not in train_features.columns or feature not in test_features.columns
        ]
        if missing_features:
            available = sorted(set(train_features.columns) & set(test_features.columns))
            raise RuntimeError(
                f"Missing requested fold-safe classifier features for {component!r} fold {fold}: "
                f"{missing_features}. Available train/test columns: {available}"
            )

        if feature_stability_group_column is not None:
            selected_features, selection = _select_group_stable_features(
                train_features,
                feature_columns,
                condition_column=condition_column,
                group_column=feature_stability_group_column,
                positive_group=positive_group,
                negative_group=negative_group,
                min_groups=feature_stability_min_groups,
                min_sign_consistency=feature_stability_min_sign_consistency,
                min_abs_delta=feature_stability_min_abs_delta,
            )
            selection.insert(0, "component", component)
            selection.insert(1, "fold", fold)
            fold_feature_selection_rows.extend(selection.to_dict("records"))
            if not selected_features:
                raise RuntimeError(
                    f"No stable features remain for {component!r} fold {fold}. "
                    "Lower feature_stability_min_sign_consistency or inspect feature_selection attrs."
                )
        else:
            selected_features = feature_columns

        train_features = train_features.dropna(subset=[*selected_features, condition_column])
        test_features = test_features.dropna(subset=[*selected_features, condition_column])
        if train_features.empty or test_features.empty:
            return {
                "score_rows": fold_score_rows,
                "coef_rows": fold_coef_rows,
                "prediction_rows": fold_prediction_rows,
                "feature_diagnostic_rows": fold_feature_diagnostic_rows,
                "feature_selection_rows": fold_feature_selection_rows,
            }

        X_train = train_features[selected_features].to_numpy()
        y_train = train_features[condition_column].eq(positive_group).astype(int).to_numpy()
        X_test = test_features[selected_features].to_numpy()
        y_test = test_features[condition_column].eq(positive_group).astype(int).to_numpy()
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            return {
                "score_rows": fold_score_rows,
                "coef_rows": fold_coef_rows,
                "prediction_rows": fold_prediction_rows,
                "feature_diagnostic_rows": fold_feature_diagnostic_rows,
                "feature_selection_rows": fold_feature_selection_rows,
            }

        fold_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", C=C),
        )
        sample_weight = _group_balanced_sample_weight(
            train_features,
            condition_column=condition_column,
            group_column=group_column,
            mode=sample_weight_mode,
        )
        fit_kwargs = (
            {"logisticregression__sample_weight": sample_weight}
            if sample_weight is not None
            else {}
        )
        fold_model.fit(X_train, y_train, **fit_kwargs)
        y_score = _positive_class_scores(fold_model, X_test)
        y_pred = (y_score >= 0.5).astype(int)
        fold_feature_diagnostic_rows.extend(
            _fold_feature_diagnostics(
                component=component,
                fold=fold,
                train_features=train_features,
                test_features=test_features,
                feature_columns=selected_features,
                condition_column=condition_column,
                group_column=group_column,
                positive_group=positive_group,
                negative_group=negative_group,
                fold_model=fold_model,
            )
        )
        fold_score_rows.append(
            {
                "component": component,
                "fold": fold,
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "roc_auc": float(roc_auc_score(y_test, y_score)),
                "n_test": int(len(test_features)),
                "held_out_group_column": group_column,
                "positive_group": positive_group,
                "n_features": int(len(selected_features)),
            }
        )
        for row_index, score_value, true_value, group_value in zip(
            test_features.index.to_numpy(),
            y_score,
            y_test,
            test_features[group_column].to_numpy() if group_column in test_features else np.repeat(fold, len(test_features)),
        ):
            fold_prediction_rows.append(
                {
                    "component": component,
                    "fold": fold,
                    "row_index": int(row_index),
                    "group": group_value,
                    "y_true": int(true_value),
                    "condition": positive_group if int(true_value) == 1 else negative_group,
                    "y_score": float(score_value),
                    "positive_group": positive_group,
                }
            )
        coefs = fold_model.named_steps["logisticregression"].coef_[0]
        for feature, coef in zip(selected_features, coefs):
            fold_coef_rows.append(
                {
                    "component": component,
                    "fold": fold,
                    "feature": feature,
                    "standardized_logistic_coef": float(coef),
                    "positive_group": positive_group,
                }
            )

        return {
            "score_rows": fold_score_rows,
            "coef_rows": fold_coef_rows,
            "prediction_rows": fold_prediction_rows,
            "feature_diagnostic_rows": fold_feature_diagnostic_rows,
            "feature_selection_rows": fold_feature_selection_rows,
        }

    fold_results = Parallel(n_jobs=n_jobs, prefer=joblib_prefer)(
        delayed(fit_fold)(fold, train_index, test_index)
        for fold, (train_index, test_index) in splits
    )
    for result in fold_results:
        score_rows.extend(result["score_rows"])
        coef_rows.extend(result["coef_rows"])
        prediction_rows.extend(result["prediction_rows"])
        feature_diagnostic_rows.extend(result["feature_diagnostic_rows"])
        feature_selection_rows.extend(result["feature_selection_rows"])

    scores = pd.DataFrame(score_rows)
    scores.attrs["oof_predictions"] = pd.DataFrame(prediction_rows)
    scores.attrs["fold_feature_diagnostics"] = pd.DataFrame(feature_diagnostic_rows)
    scores.attrs["feature_selection"] = pd.DataFrame(feature_selection_rows)
    coefficients = pd.DataFrame(coef_rows)
    if not coefficients.empty:
        coefficients = (
            coefficients
            .groupby(["component", "feature", "positive_group"], as_index=False)
            ["standardized_logistic_coef"]
            .mean()
        )
    return scores, coefficients


def summarize_classifier_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize cross-validated balanced accuracy by component."""

    aggregations = {
        "mean_balanced_accuracy": ("balanced_accuracy", "mean"),
        "sd_balanced_accuracy": ("balanced_accuracy", "std"),
        "folds": ("balanced_accuracy", "size"),
    }
    if "roc_auc" in scores.columns:
        aggregations["mean_roc_auc"] = ("roc_auc", "mean")
        aggregations["sd_roc_auc"] = ("roc_auc", "std")
    return (
        scores
        .groupby("component", as_index=False)
        .agg(**aggregations)
    )


def classifier_roc_diagnostics(scores: pd.DataFrame) -> pd.DataFrame:
    """Recompute fold ROC AUCs from stored out-of-fold predictions."""

    predictions = scores.attrs.get("oof_predictions", pd.DataFrame())
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        return pd.DataFrame()
    rows = []
    for (component, fold), fold_rows in predictions.dropna(subset=["y_true", "y_score"]).groupby(["component", "fold"]):
        y_true = fold_rows["y_true"].to_numpy(dtype=int)
        y_score = fold_rows["y_score"].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            auc = np.nan
        else:
            auc = float(roc_auc_score(y_true, y_score))
        positive_score_mean = float(np.nanmean(y_score[y_true == 1])) if np.any(y_true == 1) else np.nan
        negative_score_mean = float(np.nanmean(y_score[y_true == 0])) if np.any(y_true == 0) else np.nan
        rows.append(
            {
                "component": component,
                "fold": fold,
                "auc_from_predictions": auc,
                "score_auc_if_flipped": 1.0 - auc if np.isfinite(auc) else np.nan,
                "positive_score_mean": positive_score_mean,
                "negative_score_mean": negative_score_mean,
                "positive_minus_negative_score": positive_score_mean - negative_score_mean,
                "below_chance_auc": bool(auc < 0.5) if np.isfinite(auc) else False,
            }
        )
    diagnostics = pd.DataFrame(rows)
    if diagnostics.empty or "roc_auc" not in scores:
        return diagnostics
    diagnostics = diagnostics.merge(
        scores[["component", "fold", "roc_auc"]],
        on=["component", "fold"],
        how="left",
    )
    diagnostics["auc_table_minus_predictions"] = diagnostics["roc_auc"] - diagnostics["auc_from_predictions"]
    return diagnostics


def clean_feature_name(feature: str) -> str:
    """Human-readable short labels for downstream feature plots."""

    labels = {
        "period_ms": "Period",
        "time_rdsym": "Rise-decay",
        "time_ptsym": "Peak-trough",
        "volt_rise": "Rise voltage",
        "volt_decay": "Decay voltage",
        "band_amp": "Band amp",
        "amp_fraction": "Amplitude fraction",
        "amp_consistency": "Amplitude consistency",
        "period_consistency": "Period consistency",
        "monotonicity": "Monotonicity",
        "aperiodic_ar1": "AR(1)",
        "aperiodic_std": "Aperiodic SD",
        "aperiodic_rms": "Aperiodic RMS",
        "aperiodic_innovation_std": "Innovation SD",
        "aperiodic_hjorth_mobility": "Mobility",
        "aperiodic_hjorth_complexity": "Complexity",
        "aperiodic_line_length": "Line length",
        "aperiodic_zero_crossing_rate": "Zero-crossing",
        "theta_gamma_pac": "PAC",
    }
    for prefix, label_prefix in [
        ("slow_gamma_", "Slow gamma "),
        ("mid_gamma_", "Mid gamma "),
        ("gamma_", "Gamma "),
    ]:
        if feature.startswith(prefix):
            return label_prefix + clean_feature_name(feature[len(prefix):]).lower()
    return labels.get(feature, feature.replace("_", " ").title())


def feature_family(feature: str) -> str:
    """Classify features for coefficient coloring."""

    return "aperiodic" if feature.startswith("aperiodic_") else "periodic"


def plot_downstream_utility(
    ttest_table: pd.DataFrame,
    classifier_scores: pd.DataFrame,
    classifier_coefficients: pd.DataFrame,
    *,
    feature_columns: list[str] | tuple[str, ...],
    groups: tuple[str, str] | list[str],
    positive_group: str,
    components: tuple[str, ...] | list[str] = ("raw", "periodic"),
    colors: dict[str, str] | None = None,
    classifier_group_label: str = "held-out groups",
    title: str = "Does decomposition make waveform features more useful downstream?",
    coefficient_component: str = "periodic",
):
    """Plot effect sizes, out-of-fold ROC curves, and model weights."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    default_colors = {"raw": "#6C757D", "periodic": "#E76F51", "aperiodic": "#457B9D"}
    colors = {**default_colors, **(colors or {})}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x = np.arange(len(feature_columns))
    width = min(0.8 / max(len(components), 1), 0.36)
    offsets = np.linspace(-width * (len(components) - 1) / 2, width * (len(components) - 1) / 2, len(components))
    for offset, component in zip(offsets, components):
        values = []
        for feature in feature_columns:
            match = ttest_table[ttest_table["component"].eq(component) & ttest_table["feature"].eq(feature)]
            values.append(match["abs_cohens_d"].iloc[0] if len(match) else np.nan)
        axes[0].bar(x + offset, values, width=width, color=colors.get(component, "0.55"), label=component)
    feature_labels = [clean_feature_name(feature) for feature in feature_columns]
    axes[0].set(
        title=f"Group effect size: {groups[0]} vs {groups[1]}",
        xticks=x,
        xticklabels=feature_labels,
        ylabel="|Cohen's d|",
    )
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend(frameon=False)

    predictions = classifier_scores.attrs.get("oof_predictions")
    if isinstance(predictions, pd.DataFrame) and not predictions.empty:
        prediction_positive_groups = set(predictions.get("positive_group", pd.Series(dtype=object)).dropna().astype(str))
        if prediction_positive_groups and prediction_positive_groups != {str(positive_group)}:
            raise ValueError(
                "ROC positive-group mismatch: "
                f"plot requested {positive_group!r}, predictions contain {sorted(prediction_positive_groups)!r}."
            )
        for component in components:
            rows = predictions[predictions["component"].eq(component)]
            rows = rows.dropna(subset=["y_true", "y_score"])
            if rows.empty or rows["y_true"].nunique() < 2:
                continue
            mean_fpr = np.linspace(0, 1, 101)
            fold_tprs = []
            fold_aucs = []
            for _, fold_rows in rows.groupby("fold"):
                if fold_rows["y_true"].nunique() < 2:
                    continue
                y_true = fold_rows["y_true"].to_numpy(dtype=int)
                y_score = fold_rows["y_score"].to_numpy(dtype=float)
                if not set(np.unique(y_true)).issubset({0, 1}):
                    raise ValueError(f"ROC y_true must be encoded as 0/1; got {sorted(np.unique(y_true))}.")
                fold_fpr, fold_tpr, _ = roc_curve(
                    y_true,
                    y_score,
                    pos_label=1,
                )
                fold_auc = roc_auc_score(y_true, y_score)
                score_auc = classifier_scores.loc[
                    classifier_scores["component"].eq(component)
                    & classifier_scores["fold"].eq(fold_rows["fold"].iloc[0]),
                    "roc_auc",
                ]
                if len(score_auc) and np.isfinite(score_auc.iloc[0]) and not np.isclose(fold_auc, score_auc.iloc[0]):
                    raise ValueError(
                        "ROC AUC mismatch between plotted predictions and score table: "
                        f"{component} fold {fold_rows['fold'].iloc[0]} predictions={fold_auc:.6f}, "
                        f"score table={score_auc.iloc[0]:.6f}."
                    )
                fold_interp = np.interp(mean_fpr, fold_fpr, fold_tpr)
                fold_interp[0] = 0.0
                fold_tprs.append(fold_interp)
                fold_aucs.append(fold_auc)
                axes[1].step(
                    fold_fpr,
                    fold_tpr,
                    where="post",
                    color=colors.get(component, "0.55"),
                    lw=0.9,
                    alpha=0.18,
                    zorder=1,
                )
            if not fold_tprs:
                continue
            mean_tpr = np.mean(fold_tprs, axis=0)
            mean_tpr[-1] = 1.0
            score_rows = classifier_scores[classifier_scores["component"].eq(component)]
            auc = float(np.mean(fold_aucs))
            if "roc_auc" in score_rows:
                table_auc = float(score_rows["roc_auc"].mean())
                if np.isfinite(table_auc) and not np.isclose(auc, table_auc):
                    raise ValueError(
                        f"Mean ROC AUC mismatch for {component}: predictions={auc:.6f}, score table={table_auc:.6f}."
                    )
            axes[1].plot(
                mean_fpr,
                mean_tpr,
                color=colors.get(component, "0.55"),
                lw=2,
                label=f"{component}: fold AUC {auc:.2f}",
                zorder=3,
            )
        axes[1].plot([0, 1], [0, 1], color="black", lw=1, ls="--")
        axes[1].set(
            title=f"Mean fold ROC\n{classifier_group_label}",
            xlabel="False positive rate",
            ylabel="True positive rate",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axes[1].legend(frameon=False, loc="lower right")
    else:
        for idx, component in enumerate(components):
            scores = classifier_scores.loc[classifier_scores["component"].eq(component), "balanced_accuracy"].to_numpy()
            axes[1].bar(idx, np.nanmean(scores), color=colors.get(component, "0.55"), width=0.65)
            axes[1].scatter(np.full(len(scores), idx), scores, color="black", s=28, zorder=3)
        axes[1].axhline(0.5, color="black", lw=1, ls="--")
        axes[1].set(
            title=f"Interpretable classifier\n{classifier_group_label}",
            xticks=np.arange(len(components)),
            xticklabels=list(components),
            ylabel="Balanced accuracy",
            ylim=(0.35, 1.0),
        )

    coef_plot = classifier_coefficients[classifier_coefficients["component"].eq(coefficient_component)].copy()
    if coef_plot.empty:
        coef_plot = classifier_coefficients.copy()
        coefficient_component = "all"
    coef_plot["abs_coef"] = coef_plot["standardized_logistic_coef"].abs()
    coef_plot["feature_label"] = coef_plot["feature"].map(clean_feature_name)
    coef_plot["feature_family"] = coef_plot["feature"].map(feature_family)
    coef_plot = coef_plot.sort_values("abs_coef")
    family_colors = {"periodic": colors["periodic"], "aperiodic": colors["aperiodic"]}
    axes[2].barh(
        coef_plot["feature_label"],
        coef_plot["standardized_logistic_coef"],
        color=coef_plot["feature_family"].map(family_colors),
    )
    axes[2].axvline(0, color="black", lw=1)
    axes[2].set(
        title=f"{coefficient_component.title()}-feature classifier\nstandardized coefficients",
        xlabel=f"log-odds toward {positive_group}",
    )
    axes[2].legend(
        handles=[
            Patch(facecolor=family_colors["periodic"], label="periodic"),
            Patch(facecolor=family_colors["aperiodic"], label="aperiodic"),
        ],
        frameon=False,
        loc="lower right",
    )

    fig.suptitle(title, y=1.05)
    fig.tight_layout()
    return fig
