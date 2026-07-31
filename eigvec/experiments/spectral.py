"""Reusable AR eigenspectrum fitting and OscEig decomposition."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.optimize import least_squares

from timescales.autoreg import ARPSD
from timescales.autoreg.fit import _ar_spectrum, init_ar_from_psd

from eigvec.core import OscEig, reconstruct

from .data import SignalMatrix


def mask_peaks(powers: np.ndarray, window: int = 10, threshold: float = 0.2) -> np.ndarray:
    """Return a non-DC mask for smooth-slope spectral bins.

    The returned mask has length ``len(powers) - 1`` and aligns with
    ``freqs[1:]`` / ``powers[1:]``. High-curvature bins are treated as peaks and
    excluded from robust aperiodic fitting.
    """

    powers = np.asarray(powers, dtype=float)
    if powers.ndim != 1:
        raise ValueError("powers must be one-dimensional.")
    if len(powers) <= 1:
        return np.zeros(0, dtype=bool)
    if window < 1:
        raise ValueError("window must be >= 1.")

    safe_powers = np.maximum(powers[1:], 1e-30)
    log_powers = np.log10(safe_powers)
    finite = np.isfinite(log_powers)
    if len(log_powers) < 3:
        return finite

    d1 = np.diff(log_powers)
    d2 = np.abs(np.diff(d1))

    curvature = np.zeros(len(log_powers))
    curvature[1:-1] = d2
    kernel = np.ones(int(window), dtype=float) / float(window)

    score_backward = np.convolve(curvature, kernel, mode="full")[: len(curvature)]
    score_forward = np.convolve(curvature[::-1], kernel, mode="full")[: len(curvature)][::-1]
    score = np.minimum(score_backward, score_forward)

    cutoff = np.nanmedian(score) + threshold * np.nanstd(score)
    return (score < cutoff) & finite


class RobustAperiodicAR:
    """Robust ARPSD wrapper with explicit fit range semantics.

    AR parameters are estimated only on ``fit_min_freq <= f <= fit_max_freq``
    after removing ``exclude_bands``. For inversion, AR-predicted power is used
    only inside the fit range; outside the fit range the true eigenspectrum power
    is copied, so out-of-range power is assigned to the aperiodic component.
    Peak masking, asymmetric overfit penalties, and log-log slope penalties are
    opt-in; neutral constructor defaults do not silently apply them.
    """

    def __init__(
        self,
        fs: float,
        order: int = 10,
        fit_min_freq: float = 0.0,
        fit_max_freq: float | None = None,
        exclude_bands: list[tuple[float, float]] | None = None,
        loss_fn: str = "linear",
        f_scale: float | None = None,
        maxfev: int = 50_000,
        asymmetric_overfit_penalty: bool = False,
        overfit_weight: float = 0.0,
        low_freq_residual_weight: float = 1.0,
        low_freq_overfit_weight: float = 0.0,
        low_freq_max: float = 8.0,
        overfit_tolerance_log10: float = 0.03,
        ceiling_to_data: bool = False,
        mask_peaks_for_fit: bool = False,
        peak_mask_min_freq: float | None = None,
        peak_mask_window: int = 10,
        peak_mask_threshold: float = 0.2,
        loglog_slope_change_weight: float = 0.0,
        positive_loglog_slope_weight: float = 0.0,
        positive_loglog_slope_tolerance: float = 0.0,
        slope_penalty_grid_size: int | None = None,
    ):
        self.fs = fs
        self.order = order
        self.fit_min_freq = fit_min_freq
        self.fit_max_freq = fs / 2 if fit_max_freq is None else fit_max_freq
        self.exclude_bands = [] if exclude_bands is None else list(exclude_bands)
        self.loss_fn = loss_fn
        self.f_scale = f_scale
        self.maxfev = maxfev
        self.asymmetric_overfit_penalty = asymmetric_overfit_penalty
        self.overfit_weight = overfit_weight
        self.low_freq_residual_weight = low_freq_residual_weight
        self.low_freq_overfit_weight = low_freq_overfit_weight
        self.low_freq_max = low_freq_max
        self.overfit_tolerance_log10 = overfit_tolerance_log10
        self.ceiling_to_data = ceiling_to_data
        self.mask_peaks_for_fit = mask_peaks_for_fit
        self.peak_mask_min_freq = peak_mask_min_freq
        self.peak_mask_window = peak_mask_window
        self.peak_mask_threshold = peak_mask_threshold
        self.loglog_slope_change_weight = loglog_slope_change_weight
        self.positive_loglog_slope_weight = positive_loglog_slope_weight
        self.positive_loglog_slope_tolerance = positive_loglog_slope_tolerance
        self.slope_penalty_grid_size = slope_penalty_grid_size
        self.model = None
        self.freqs_ = None
        self.powers_ = None
        self.powers_fit = None
        self.range_mask = None
        self.fit_mask = None
        self.fit_mask_before_peak_mask = None
        self.peak_mask = None
        self.params = None

    def fit(self, freqs: np.ndarray, powers: np.ndarray) -> np.ndarray:
        freqs = np.asarray(freqs, dtype=float)
        powers = np.asarray(powers, dtype=float)
        self.freqs_ = freqs
        self.powers_ = powers

        self.range_mask = (
            np.isfinite(powers)
            & (powers > 0)
            & (freqs >= self.fit_min_freq)
            & (freqs <= self.fit_max_freq)
        )
        self.fit_mask = self.range_mask.copy()
        for low, high in self.exclude_bands:
            self.fit_mask &= ~((freqs >= low) & (freqs <= high))
        self.fit_mask_before_peak_mask = self.fit_mask.copy()
        self.peak_mask = np.ones(len(powers), dtype=bool)
        if self.mask_peaks_for_fit:
            self.peak_mask = np.zeros(len(powers), dtype=bool)
            self.peak_mask[1:] = mask_peaks(
                powers,
                window=self.peak_mask_window,
                threshold=self.peak_mask_threshold,
            )
            if self.peak_mask_min_freq is not None:
                self.peak_mask |= freqs < self.peak_mask_min_freq
            self.fit_mask &= self.peak_mask
        if self.fit_mask.sum() <= self.order + 1:
            raise RuntimeError("Too few frequency bins remain for ARPSD fitting.")

        if self._uses_regularized_fit():
            self._fit_regularized(freqs[self.fit_mask], powers[self.fit_mask])
        else:
            self.model = ARPSD(
                order=self.order,
                fs=self.fs,
                maxfev=self.maxfev,
                loss_fn=self.loss_fn,
                f_scale=self.f_scale,
            )
            self.model.fit(freqs[self.fit_mask], powers[self.fit_mask])
            self.params = np.asarray(self.model.params, dtype=float)

        k = np.arange(1, self.order + 1)
        exponent = np.exp(-2j * np.pi * np.outer(freqs, k) / self.fs).T
        ar_power = _ar_spectrum(exponent, *self.params)
        if self.ceiling_to_data:
            ar_power = ar_power.copy()
            ar_power[self.range_mask] = np.minimum(ar_power[self.range_mask], powers[self.range_mask])

        self.powers_fit = powers.copy()
        self.powers_fit[self.range_mask] = ar_power[self.range_mask]
        return self.powers_fit

    def plot(
        self,
        ax=None,
        *,
        settings_ax=None,
        show_settings: bool = True,
        show_fit_points: bool = True,
        show_rejected_points: bool = True,
        show_fit_range: bool = True,
        loglog: bool = True,
        title: str | None = None,
    ):
        """Plot the fitted AR background and the bins used for fitting.

        Returns a dictionary with ``"spectrum"`` and, when requested,
        ``"settings"`` axes. Call after ``fit``.
        """

        if self.freqs_ is None or self.powers_ is None or self.powers_fit is None:
            raise RuntimeError("Call fit before plot.")

        import matplotlib.pyplot as plt

        if ax is None:
            if show_settings:
                fig, (ax, settings_ax) = plt.subplots(
                    1,
                    2,
                    figsize=(10, 4),
                    gridspec_kw={"width_ratios": [3.0, 1.35]},
                    constrained_layout=True,
                )
            else:
                fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        else:
            fig = ax.figure
            if show_settings and settings_ax is None:
                settings_ax = ax.inset_axes([1.04, 0.0, 0.75, 1.0])

        freqs = self.freqs_
        powers = self.powers_
        powers_fit = self.powers_fit
        valid = (
            np.isfinite(freqs)
            & np.isfinite(powers)
            & np.isfinite(powers_fit)
            & (freqs > 0)
            & (powers > 0)
            & (powers_fit > 0)
        )
        if not np.any(valid):
            raise RuntimeError("No positive finite frequency/power bins are available to plot.")

        plot = ax.loglog if loglog else ax.plot
        scatter_xscale = "log" if loglog else "linear"
        scatter_yscale = "log" if loglog else "linear"

        if show_fit_range and self.range_mask is not None:
            fit_low = max(float(self.fit_min_freq), float(np.min(freqs[valid])))
            fit_high = min(float(self.fit_max_freq), float(np.max(freqs[valid])))
            if fit_high > fit_low:
                ax.axvspan(fit_low, fit_high, color="tab:blue", alpha=0.07, lw=0, label="fit range")

        for low, high in self.exclude_bands:
            ax.axvspan(low, high, color="tab:orange", alpha=0.13, lw=0, label="excluded band")

        plot(freqs[valid], powers[valid], color="0.25", lw=1.1, label="spectrum")

        if show_fit_points and self.fit_mask is not None:
            fit_points = valid & self.fit_mask
            ax.scatter(
                freqs[fit_points],
                powers[fit_points],
                s=14,
                color="tab:green",
                alpha=0.75,
                label="fit bins",
                zorder=3,
            )

        if show_rejected_points:
            rejected = np.zeros_like(valid, dtype=bool)
            if self.range_mask is not None and self.fit_mask is not None:
                rejected |= valid & self.range_mask & ~self.fit_mask
            if np.any(rejected):
                ax.scatter(
                    freqs[rejected],
                    powers[rejected],
                    s=16,
                    color="tab:purple",
                    marker="x",
                    alpha=0.8,
                    label="rejected bins",
                    zorder=3,
                )




        ax.set_xscale(scatter_xscale)
        ax.set_yscale(scatter_yscale)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        ax.set_title("RobustAperiodicAR fit" if title is None else title)
        plot(freqs[valid], powers_fit[valid], color="tab:red", lw=1.4, label="AR fit", zorder=5)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=8)

        if show_settings:
            self._plot_settings(settings_ax)

        return {"spectrum": ax, "settings": settings_ax if show_settings else None}

    def _plot_settings(self, ax) -> None:
        if ax is None:
            return

        ax.axis("off")
        lines = [
            f"order: {self.order}",
            f"fit range: {self.fit_min_freq:g}-{self.fit_max_freq:g} Hz",
            f"loss: {self.loss_fn}",
            f"f_scale: {self.f_scale}",
            f"maxfev: {self.maxfev}",
            f"range bins: {int(np.sum(self.range_mask)) if self.range_mask is not None else 'NA'}",
            f"fit bins: {int(np.sum(self.fit_mask)) if self.fit_mask is not None else 'NA'}",
        ]
        if self.exclude_bands:
            lines.append(f"exclude bands: {self.exclude_bands}")
        if self.mask_peaks_for_fit:
            removed = 0
            if self.fit_mask_before_peak_mask is not None and self.fit_mask is not None:
                removed = int(np.sum(self.fit_mask_before_peak_mask & ~self.fit_mask))
            lines.extend([
                "peak mask: True",
                f"peak min freq: {self.peak_mask_min_freq}",
                f"peak window: {self.peak_mask_window}",
                f"peak threshold: {self.peak_mask_threshold}",
                f"peak removed bins: {removed}",
            ])
        if self.asymmetric_overfit_penalty:
            lines.extend([
                "asymmetric overfit: True",
                f"overfit weight: {self.overfit_weight}",
                f"low freq residual weight: {self.low_freq_residual_weight}",
                f"low freq overfit weight: {self.low_freq_overfit_weight}",
                f"low freq max: {self.low_freq_max}",
                f"overfit tol log10: {self.overfit_tolerance_log10}",
            ])
        if self.ceiling_to_data:
            lines.append("ceiling to data: True")
        if self.loglog_slope_change_weight > 0 or self.positive_loglog_slope_weight > 0:
            lines.extend([
                f"slope change weight: {self.loglog_slope_change_weight}",
                f"positive slope weight: {self.positive_loglog_slope_weight}",
                f"positive slope tol: {self.positive_loglog_slope_tolerance}",
                f"slope grid size: {self.slope_penalty_grid_size}",
            ])

        ax.text(
            0.0,
            1.0,
            "\n".join(lines),
            va="top",
            ha="left",
            family="monospace",
            fontsize=8.5,
        )
        ax.set_title("Fit settings", fontsize=10, loc="left")

    def _uses_regularized_fit(self) -> bool:
        return bool(
            self.asymmetric_overfit_penalty
            or self.loglog_slope_change_weight > 0
            or self.positive_loglog_slope_weight > 0
        )

    def _fit_regularized(self, freqs: np.ndarray, powers: np.ndarray) -> None:
        """Fit AR parameters with optional log-space shape penalties."""

        powers = np.maximum(np.asarray(powers, dtype=float), 1e-30)
        freqs = np.asarray(freqs, dtype=float)
        k = np.arange(1, self.order + 1)
        exponent = np.exp(-2j * np.pi * np.outer(freqs, k) / self.fs).T
        slope_freqs = self._slope_penalty_freqs(freqs)
        slope_exponent = np.exp(-2j * np.pi * np.outer(slope_freqs, k) / self.fs).T

        phi_init = init_ar_from_psd(powers, self.order)
        offset_init = np.exp(np.mean(np.log(powers)))
        offset_upper = max(float(np.max(powers) * 1e6), float(offset_init * 1e6), 1e-12)

        lower = np.array([-0.9999] * self.order + [1e-30], dtype=float)
        upper = np.array([0.9999] * self.order + [offset_upper], dtype=float)
        guess = np.array([*phi_init, offset_init], dtype=float)
        guess = np.clip(guess, lower + 1e-12, upper - 1e-12)

        result_kwargs = {
            "loss": self.loss_fn,
            "max_nfev": self.maxfev,
        }
        if self.f_scale is not None:
            result_kwargs["f_scale"] = self.f_scale

        result = least_squares(
            self._regularized_residual,
            x0=guess,
            bounds=(lower, upper),
            args=(exponent, freqs, powers, slope_exponent, slope_freqs),
            **result_kwargs,
        )
        self.params = result.x
        self.model = SimpleNamespace(params=self.params, result=result)

    def _regularized_residual(
        self,
        params: np.ndarray,
        exponent: np.ndarray,
        freqs: np.ndarray,
        powers: np.ndarray,
        slope_exponent: np.ndarray,
        slope_freqs: np.ndarray,
    ) -> np.ndarray:
        pred = np.maximum(_ar_spectrum(exponent, *params), 1e-30)
        true = np.maximum(powers, 1e-30)

        resid = np.log10(pred) - np.log10(true)
        resid_weight = np.ones_like(resid)
        resid_weight[freqs <= self.low_freq_max] = np.sqrt(self.low_freq_residual_weight)
        residuals = [resid_weight * resid]

        if self.asymmetric_overfit_penalty:
            over = np.maximum(resid - self.overfit_tolerance_log10, 0.0)
            low_over = over * (freqs <= self.low_freq_max)
            residuals.extend([
                np.sqrt(self.overfit_weight) * over,
                np.sqrt(self.low_freq_overfit_weight) * low_over,
            ])

        if self.loglog_slope_change_weight > 0 or self.positive_loglog_slope_weight > 0:
            slope_pred = np.maximum(_ar_spectrum(slope_exponent, *params), 1e-30)
            slopes = self._loglog_slopes(slope_freqs, np.log10(slope_pred))
            if self.loglog_slope_change_weight > 0 and len(slopes) >= 2:
                residuals.append(
                    np.sqrt(self.loglog_slope_change_weight) * np.diff(slopes)
                )
            if self.positive_loglog_slope_weight > 0 and len(slopes):
                positive_slopes = np.maximum(
                    slopes - self.positive_loglog_slope_tolerance,
                    0.0,
                )
                residuals.append(
                    np.sqrt(self.positive_loglog_slope_weight) * positive_slopes
                )

        return np.concatenate(residuals)

    def _slope_penalty_freqs(self, freqs: np.ndarray) -> np.ndarray:
        positive = np.asarray(freqs, dtype=float)
        positive = positive[np.isfinite(positive) & (positive > 0)]
        if len(positive) < 2:
            return positive

        size = self.slope_penalty_grid_size
        if size is None:
            size = max(len(positive), 64)
        size = int(size)
        if size <= 2:
            return np.unique(positive)
        return np.geomspace(float(positive.min()), float(positive.max()), size)

    @staticmethod
    def _loglog_slopes(freqs: np.ndarray, log_powers: np.ndarray) -> np.ndarray:
        log_freqs = np.log10(np.asarray(freqs, dtype=float))
        log_powers = np.asarray(log_powers, dtype=float)
        finite = np.isfinite(log_freqs) & np.isfinite(log_powers)
        log_freqs = log_freqs[finite]
        log_powers = log_powers[finite]
        if len(log_freqs) < 2:
            return np.zeros(0, dtype=float)
        dx = np.diff(log_freqs)
        keep = dx > 0
        return np.diff(log_powers)[keep] / dx[keep]


def fit_condition_decompositions(
    matrices: dict[str, SignalMatrix],
    *,
    periodic_mode: str = "quadrature",
    quadrature_balance_band: tuple[float, float] | None = None,
    order: int = 10,
    fit_min_freq: float = 0.0,
    fit_max_freq: float | None = None,
    exclude_bands: list[tuple[float, float]] | None = None,
    loss_fn: str = "linear",
    f_scale: float | None = None,
    maxfev: int = 50_000,
    asymmetric_overfit_penalty: bool = False,
    overfit_weight: float = 0.0,
    low_freq_residual_weight: float = 1.0,
    low_freq_overfit_weight: float = 0.0,
    low_freq_max: float = 8.0,
    overfit_tolerance_log10: float = 0.03,
    ceiling_to_data: bool = False,
    mask_peaks_for_fit: bool = False,
    peak_mask_min_freq: float | None = None,
    peak_mask_window: int = 10,
    peak_mask_threshold: float = 0.2,
    loglog_slope_change_weight: float = 0.0,
    positive_loglog_slope_weight: float = 0.0,
    positive_loglog_slope_tolerance: float = 0.0,
    slope_penalty_grid_size: int | None = None,
) -> tuple[dict[str, OscEig], dict[str, SignalMatrix], dict[str, SignalMatrix]]:
    """Fit OscEig per condition and return models, aperiodic, and periodic matrices.

    ``quadrature_balance_band`` is used only by ``quadrature_phase``; each
    condition derives its own pivot from its fitted eigenspectrum.
    """

    models = {}
    aperiodic = {}
    periodic = {}
    for condition, matrix in matrices.items():
        psd_model = RobustAperiodicAR(
            fs=matrix.fs,
            order=order,
            fit_min_freq=fit_min_freq,
            fit_max_freq=fit_max_freq,
            exclude_bands=exclude_bands,
            loss_fn=loss_fn,
            f_scale=f_scale,
            maxfev=maxfev,
            asymmetric_overfit_penalty=asymmetric_overfit_penalty,
            overfit_weight=overfit_weight,
            low_freq_residual_weight=low_freq_residual_weight,
            low_freq_overfit_weight=low_freq_overfit_weight,
            low_freq_max=low_freq_max,
            overfit_tolerance_log10=overfit_tolerance_log10,
            ceiling_to_data=ceiling_to_data,
            mask_peaks_for_fit=mask_peaks_for_fit,
            peak_mask_min_freq=peak_mask_min_freq,
            peak_mask_window=peak_mask_window,
            peak_mask_threshold=peak_mask_threshold,
            loglog_slope_change_weight=loglog_slope_change_weight,
            positive_loglog_slope_weight=positive_loglog_slope_weight,
            positive_loglog_slope_tolerance=positive_loglog_slope_tolerance,
            slope_penalty_grid_size=slope_penalty_grid_size,
        )
        model = OscEig(
            fs=matrix.fs,
            psd_model=psd_model,
            periodic_mode=periodic_mode,
            quadrature_balance_band=quadrature_balance_band,
        )
        model.fit(matrix.X)
        X_aperiodic, X_periodic = model.decompose()
        if not np.allclose(matrix.X, X_aperiodic + X_periodic, atol=1e-5):
            raise RuntimeError(f"Decomposition does not reconstruct {condition}.")

        models[condition] = model
        aperiodic[condition] = SignalMatrix(X_aperiodic, matrix.rows, matrix.fs, condition)
        periodic[condition] = SignalMatrix(X_periodic, matrix.rows, matrix.fs, condition)

    return models, aperiodic, periodic


def decompose_with_fitted_model(
    model: OscEig,
    matrix: SignalMatrix,
    *,
    condition: str | None = None,
) -> tuple[SignalMatrix, SignalMatrix]:
    """Apply a train-fitted OscEig power curve to held-out rows.

    ``OscEig`` powers are scatter powers summed across matrix rows. When the
    number of held-out rows differs from the fit matrix, scale the fitted
    scatter by ``n_heldout / n_fit`` so reconstructed row amplitudes stay on the
    same per-row scale.
    """

    if model.powers_decomp_ is None or model.X_ is None:
        raise RuntimeError("Call fit on the OscEig model before transforming held-out rows.")

    row_scale = matrix.X.shape[0] / model.X_.shape[0]
    powers = np.asarray(model.powers_decomp_, dtype=float) * row_scale
    X_aperiodic, X_periodic, _, _, _ = reconstruct(
        matrix.X,
        powers,
        method=model.method,
        eps=model.eps,
        return_factors=False,
        periodic_mode=model.periodic_mode,
        fs=matrix.fs,
        quadrature_signs=model.quadrature_signs_,
    )
    matrix_condition = matrix.condition if condition is None else condition
    return (
        SignalMatrix(X_aperiodic, matrix.rows.copy(), matrix.fs, matrix_condition),
        SignalMatrix(X_periodic, matrix.rows.copy(), matrix.fs, matrix_condition),
    )


def fit_pooled_decomposition_model(
    matrices: dict[str, SignalMatrix],
    *,
    periodic_mode: str = "quadrature",
    quadrature_balance_band: tuple[float, float] | None = None,
    order: int = 10,
    fit_min_freq: float = 0.0,
    fit_max_freq: float | None = None,
    exclude_bands: list[tuple[float, float]] | None = None,
    loss_fn: str = "linear",
    f_scale: float | None = None,
    maxfev: int = 50_000,
    asymmetric_overfit_penalty: bool = False,
    overfit_weight: float = 0.0,
    low_freq_residual_weight: float = 1.0,
    low_freq_overfit_weight: float = 0.0,
    low_freq_max: float = 8.0,
    overfit_tolerance_log10: float = 0.03,
    ceiling_to_data: bool = False,
    mask_peaks_for_fit: bool = False,
    peak_mask_min_freq: float | None = None,
    peak_mask_window: int = 10,
    peak_mask_threshold: float = 0.2,
    loglog_slope_change_weight: float = 0.0,
    positive_loglog_slope_weight: float = 0.0,
    positive_loglog_slope_tolerance: float = 0.0,
    slope_penalty_grid_size: int | None = None,
) -> OscEig:
    """Fit one unsupervised decomposition model after pooling conditions.

    For ``quadrature_phase``, the pooled training eigenspectrum selects the
    pivot in ``quadrature_balance_band`` and held-out transforms reuse it.
    """

    if not matrices:
        raise ValueError("At least one matrix is required.")
    fs_values = {float(matrix.fs) for matrix in matrices.values()}
    if len(fs_values) != 1:
        raise ValueError(f"All matrices must share one sampling rate; got {sorted(fs_values)}.")

    psd_model = RobustAperiodicAR(
        fs=fs_values.pop(),
        order=order,
        fit_min_freq=fit_min_freq,
        fit_max_freq=fit_max_freq,
        exclude_bands=exclude_bands,
        loss_fn=loss_fn,
        f_scale=f_scale,
        maxfev=maxfev,
        asymmetric_overfit_penalty=asymmetric_overfit_penalty,
        overfit_weight=overfit_weight,
        low_freq_residual_weight=low_freq_residual_weight,
        low_freq_overfit_weight=low_freq_overfit_weight,
        low_freq_max=low_freq_max,
        overfit_tolerance_log10=overfit_tolerance_log10,
        ceiling_to_data=ceiling_to_data,
        mask_peaks_for_fit=mask_peaks_for_fit,
        peak_mask_min_freq=peak_mask_min_freq,
        peak_mask_window=peak_mask_window,
        peak_mask_threshold=peak_mask_threshold,
        loglog_slope_change_weight=loglog_slope_change_weight,
        positive_loglog_slope_weight=positive_loglog_slope_weight,
        positive_loglog_slope_tolerance=positive_loglog_slope_tolerance,
        slope_penalty_grid_size=slope_penalty_grid_size,
    )
    pooled_X = np.vstack([matrix.X for matrix in matrices.values()])
    return OscEig(
        fs=psd_model.fs,
        psd_model=psd_model,
        periodic_mode=periodic_mode,
        quadrature_balance_band=quadrature_balance_band,
    ).fit(pooled_X)
