"""Bycycle feature measurement and visualization helpers."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from bycycle import Bycycle
from scipy.spatial.distance import jensenshannon

from .data import SignalMatrix


def measure_bycycle_features(
    matrices: dict[str, SignalMatrix],
    band: tuple[float, float] | dict[str, tuple[float, float]],
    *,
    component: str,
    thresholds=None,
    find_extrema_kwargs=None,
    provenance_columns=None,
    burst_only: bool = False,
    min_cycles_per_row: int = 1,
    suppress_warnings: bool = True,
) -> pd.DataFrame:
    """Measure cycle-by-cycle waveform features for each condition matrix.

    ``Bycycle`` thresholds mark cycles in ``df_features["is_burst"]``. Set
    ``burst_only=True`` when the analysis should summarize only sustained,
    threshold-passing oscillatory cycles instead of every detected cycle.
    """

    thresholds = thresholds or {
        "amp_fraction": 0.2,
        "amp_consistency": 0.2,
        "period_consistency": 0.2,
        "monotonicity": 0.2,
    }
    provenance_columns = provenance_columns or []

    tables = []
    patch_bycycle_readonly_burst_mask()
    for condition, matrix in matrices.items():
        condition_band = band[condition] if isinstance(band, dict) else band
        # Bycycle may write into intermediate boolean arrays created from the
        # input. Some matrix operations produce read-only views, so pass an
        # owned C-contiguous copy here.
        X = np.array(matrix.X, dtype=np.float64, order="C", copy=True)
        for row_index, signal in enumerate(X):
            model = Bycycle(thresholds=thresholds, find_extrema_kwargs=find_extrema_kwargs)
            if suppress_warnings:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="neurodsp")
                    model.fit(signal, matrix.fs, condition_band)
            else:
                model.fit(signal, matrix.fs, condition_band)
            features = model.df_features
            if features.empty:
                continue
            features = features.copy()
            if burst_only:
                if "is_burst" not in features:
                    continue
                features = features[features["is_burst"].to_numpy(dtype=bool)].copy()
            if len(features) < min_cycles_per_row:
                continue
            features["period_ms"] = 1000 * features["period"] / matrix.fs
            features["band_low_hz"] = condition_band[0]
            features["band_high_hz"] = condition_band[1]
            features["matrix_row"] = row_index
            features["condition"] = condition
            features["component"] = component
            source = matrix.rows.iloc[row_index]
            for column in provenance_columns:
                if column in source.index:
                    features[column] = source[column]
            tables.append(features)

    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def patch_bycycle_readonly_burst_mask() -> None:
    """Patch a bycycle/pandas compatibility issue with read-only boolean masks."""

    import bycycle.burst.cycle as cycle_mod
    import bycycle.features.features as features_mod
    import bycycle.group.features as group_features_mod

    if getattr(cycle_mod.detect_bursts_cycles, "_eigvec_readonly_patch", False):
        return

    def detect_bursts_cycles(
        df_features,
        amp_fraction_threshold=0.0,
        amp_consistency_threshold=0.5,
        period_consistency_threshold=0.5,
        monotonicity_threshold=0.8,
        min_n_cycles=3,
    ):
        cycle_mod.check_param_range(amp_fraction_threshold, "amp_fraction_threshold", (0, 1))
        cycle_mod.check_param_range(amp_consistency_threshold, "amp_consistency_threshold", (0, 1))
        cycle_mod.check_param_range(period_consistency_threshold, "period_consistency_threshold", (0, 1))
        cycle_mod.check_param_range(monotonicity_threshold, "monotonicity_threshold", (0, 1))

        amp_fraction = df_features["amp_fraction"] > amp_fraction_threshold
        amp_consistency = df_features["amp_consistency"] > amp_consistency_threshold
        period_consistency = df_features["period_consistency"] > period_consistency_threshold
        monotonicity = df_features["monotonicity"] > monotonicity_threshold

        is_burst = (amp_fraction & amp_consistency & period_consistency & monotonicity).to_numpy().copy()
        if len(is_burst):
            is_burst[0] = False
            is_burst[-1] = False

        df_features = df_features.copy()
        df_features["is_burst"] = cycle_mod.check_min_burst_cycles(
            is_burst,
            min_n_cycles=min_n_cycles,
        )
        return df_features

    detect_bursts_cycles._eigvec_readonly_patch = True
    cycle_mod.detect_bursts_cycles = detect_bursts_cycles
    features_mod.detect_bursts_cycles = detect_bursts_cycles
    group_features_mod.detect_bursts_cycles = detect_bursts_cycles


def _patch_bycycle_readonly_burst_mask() -> None:
    """Backward-compatible alias for the public bycycle patch helper."""

    patch_bycycle_readonly_burst_mask()


def plot_bycycle_histograms(
    bycycle_features: pd.DataFrame,
    *,
    conditions,
    components=("raw", "periodic"),
    features=None,
    colors=None,
    reference_condition=None,
):
    """Plot condition histograms and return distribution divergence table."""

    import matplotlib.pyplot as plt

    if bycycle_features.empty:
        raise ValueError("bycycle_features is empty.")
    features = features or [
        ("period_ms", "Period (ms)"),
        ("time_rdsym", "Rise-decay symmetry"),
        ("time_ptsym", "Peak-trough symmetry"),
        ("volt_amp", "Voltage amplitude"),
        ("band_amp", "Band amplitude"),
    ]
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = {condition: cmap(index % 10) for index, condition in enumerate(conditions)}
    else:
        cmap = plt.get_cmap("tab10")
        colors = {
            condition: colors.get(condition, cmap(index % 10))
            for index, condition in enumerate(conditions)
        }
    reference_condition = conditions[0] if reference_condition is None else reference_condition

    fig, axes = plt.subplots(len(components), len(features), figsize=(4 * len(features), 3.5 * len(components)))
    axes = np.asarray(axes).reshape(len(components), len(features))
    divergence_rows = []

    for row_index, component in enumerate(components):
        component_features = bycycle_features[bycycle_features["component"].eq(component)]
        for column_index, (feature, label) in enumerate(features):
            ax = axes[row_index, column_index]
            finite = component_features[feature].replace([np.inf, -np.inf], np.nan).dropna()
            if finite.empty:
                ax.set_axis_off()
                continue
            if feature.endswith("sym"):
                bins = np.linspace(0, 1, 31)
            else:
                low, high = finite.quantile([0.01, 0.99])
                if not np.isfinite(low) or not np.isfinite(high) or low == high:
                    low, high = float(finite.min()), float(finite.max())
                bins = np.linspace(low, high, 31)

            condition_values = {}
            displayed_values = {}
            for condition in conditions:
                values = component_features.loc[component_features["condition"].eq(condition), feature]
                values = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
                condition_values[condition] = values
                displayed = values[(values >= bins[0]) & (values <= bins[-1])]
                displayed_values[condition] = displayed
                ax.hist(displayed, bins=bins, density=True, histtype="step", lw=1.8, color=colors[condition], label=condition)
                if len(values):
                    ax.axvline(np.median(values), color=colors[condition], ls="--", lw=1.2)

            ref_values = condition_values[reference_condition]
            ref_displayed = displayed_values[reference_condition]
            ref_median = np.median(ref_values) if len(ref_values) else np.nan
            max_js_distance = np.nan
            for condition in conditions:
                if condition == reference_condition:
                    continue
                condition_median = np.median(condition_values[condition]) if len(condition_values[condition]) else np.nan
                percent_change = 100 * (condition_median - ref_median) / (abs(ref_median) + 1e-12)
                hist_ref = np.histogram(ref_displayed, bins=bins)[0].astype(float)
                hist_condition = np.histogram(displayed_values[condition], bins=bins)[0].astype(float)
                js_distance = jensenshannon(hist_ref + 1e-12, hist_condition + 1e-12, base=2)
                max_js_distance = np.nanmax([max_js_distance, js_distance])
                divergence_rows.append(
                    {
                        "component": component,
                        "feature": feature,
                        "reference_condition": reference_condition,
                        "condition": condition,
                        "reference_median": ref_median,
                        "condition_median": condition_median,
                        "median_percent_change": percent_change,
                        "jensen_shannon_distance": js_distance,
                    }
                )
            title_suffix = f"\nmax JS vs {reference_condition}: {max_js_distance:.2f}" if len(conditions) > 1 else ""
            ax.set(title=f"{component}: {label}{title_suffix}", xlabel=label, ylabel="Density")
            if row_index == 0 and column_index == 0:
                ax.legend(frameon=False)

    fig.tight_layout()
    return fig, pd.DataFrame(divergence_rows)
