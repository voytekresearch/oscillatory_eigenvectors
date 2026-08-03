"""Core oscillatory eigenspectrum model."""

from typing import Optional

import numpy as np

from eigvec.circulant import compute_kappa as _compute_kappa


__all__ = [
    "OscEig",
    "PERIODIC_MODES",
    "decompose_fourier",
    "fit",
    "mirror_pos_to_full",
    "reconstruct",
]


PERIODIC_MODES = {
    "additive",
    "quadrature",
    "quadrature_phase",
}

_QUADRATURE_PHASE_MODE = "quadrature_phase"


def mirror_pos_to_full(d_pos, n):
    """Mirror positive-frequency values into full FFT order.

    Parameters
    ----------
    d_pos : 1d array
        Values paired with ``np.fft.rfftfreq(n, 1 / fs)``.
    n : int
        Number of time samples in the original signal.

    Returns
    -------
    d_full : 1d array
        Values paired with ``np.fft.fftfreq(n, 1 / fs)``.
    """
    d_pos = np.asarray(d_pos)
    n_pos = n // 2 + 1

    if d_pos.shape != (n_pos,):
        raise ValueError("d_pos must have length n // 2 + 1.")

    d_full = np.zeros(n, dtype=float)

    if n % 2 == 0:
        d_full[0] = d_pos[0]
        d_full[1:n//2] = d_pos[1:-1]
        d_full[n//2] = d_pos[-1]
        d_full[n//2+1:] = d_pos[1:-1][::-1]
    else:
        d_full[0] = d_pos[0]
        d_full[1:n//2+1] = d_pos[1:]
        d_full[n//2+1:] = d_pos[1:][::-1]

    return d_full


def decompose_fourier(
    X,
    d_raw,
    d_fit,
    *,
    periodic_mode="quadrature",
    eps=1e-12,
    fs=None,
    quadrature_balance_band=None,
    quadrature_pivot_frequency=None,
    quadrature_signs=None,
):
    """Split signals using raw and fitted full-order Fourier powers.

    ``quadrature`` is the default and retains the original fixed-branch
    quadrature construction. It returns exactly additive components whose
    non-edge aggregate powers are the fitted aperiodic power and the positive
    power excess. ``quadrature_phase`` preserves those identities while
    reversing the quadrature branch above a training-derived frequency pivot.
    The pivot balances leading and lagging phase contributions in a requested
    frequency band without changing either component spectrum.

    Parameters
    ----------
    X : 2d array
        Input data, arranged as observations by time.
    d_raw, d_fit : 1d arrays
        Reference raw and fitted scatter powers in full FFT order.
    periodic_mode : {"additive", "quadrature", "quadrature_phase"}, optional
        ``"additive"`` preserves Fourier phase and defines the periodic
        component as ``X - X_aperiodic``. ``"quadrature"`` gives positive
        excess power and exact signal additivity using the original fixed
        quadrature branch. ``"quadrature_phase"`` chooses opposite quadrature
        branches below and above a frequency pivot to reduce systematic
        band-limited lag while retaining the same component powers.
    eps : float, optional
        Small floor used when dividing by raw power.
    fs : float, optional
        Sampling frequency used to interpret quadrature phase settings.
    quadrature_balance_band : tuple of float, optional
        Frequency interval in which ``quadrature_phase`` derives a pivot that
        balances leading and lagging contributions.
    quadrature_pivot_frequency : float, optional
        Explicit pivot frequency for ``quadrature_phase``. Mutually exclusive
        with ``quadrature_balance_band`` and ``quadrature_signs``.
    quadrature_signs : 1d array, optional
        Explicit full-order branch signs for ``quadrature_phase``. This is the
        representation reused for held-out data after fitting.

    Returns
    -------
    X_aperiodic, X_periodic : 2d arrays
        Time-domain signal components.
    """
    X = _validate_X(X)
    input_is_real = np.isrealobj(X)
    n = X.shape[1]
    d_raw = _validate_full_powers(d_raw, n, "d_raw")
    d_fit = _validate_full_powers(d_fit, n, "d_fit")
    periodic_mode = _validate_periodic_mode(periodic_mode)
    _validate_quadrature_phase_arguments_for_mode(
        periodic_mode,
        quadrature_balance_band,
        quadrature_pivot_frequency,
        quadrature_signs,
    )

    raw_floor = np.maximum(d_raw, eps)
    Z = np.fft.fft(X, axis=1) / np.sqrt(n)

    if periodic_mode == "additive":
        aperiodic_filter = np.sqrt(d_fit / raw_floor)
        periodic_filter = 1.0 - aperiodic_filter
    else:
        aperiodic_fraction = np.divide(
            d_fit,
            d_raw,
            out=np.ones_like(d_raw),
            where=d_raw > 0.0,
        )
        aperiodic_fraction = np.clip(aperiodic_fraction, 0.0, 1.0)
        quadrature = np.sqrt(aperiodic_fraction * (1.0 - aperiodic_fraction))

        if periodic_mode == _QUADRATURE_PHASE_MODE:
            signs, _ = _resolve_quadrature_phase_signs(
                d_raw,
                d_fit,
                fs=fs,
                balance_band=quadrature_balance_band,
                pivot_frequency=quadrature_pivot_frequency,
                signs=quadrature_signs,
            )
        else:
            signs = np.sign(np.fft.fftfreq(n))
            signs[0] = 0.0
            if n % 2 == 0:
                signs[n // 2] = 0.0

        if not input_is_real:
            signs = signs.copy()
            signs[0] = 1.0
            if n % 2 == 0:
                signs[n // 2] = 1.0

        quadrature *= signs
        aperiodic_filter = aperiodic_fraction + 1j * quadrature
        periodic_filter = 1.0 - aperiodic_filter

    Z_aperiodic = Z * aperiodic_filter[None, :]
    Z_periodic = Z * periodic_filter[None, :]

    if input_is_real and periodic_mode in {"quadrature", _QUADRATURE_PHASE_MODE}:
        _validate_symmetric_powers(d_raw, "d_raw")
        _validate_symmetric_powers(d_fit, "d_fit")
        # A scalar real-valued DC or Nyquist coefficient cannot be divided
        # into two nonzero orthogonal components. Keep those isolated bins in
        # the aperiodic signal. Crucially, this preserves row independence;
        # decomposition must never mix observations to manufacture an exact
        # aggregate edge-bin power split.
        edge_indices = [0]
        if n % 2 == 0:
            edge_indices.append(n // 2)
        Z_aperiodic[:, edge_indices] = Z[:, edge_indices]
        Z_periodic[:, edge_indices] = 0.0

    X_aperiodic = np.fft.ifft(Z_aperiodic * np.sqrt(n), axis=1)
    X_periodic = np.fft.ifft(Z_periodic * np.sqrt(n), axis=1)
    if input_is_real:
        X_aperiodic = X_aperiodic.real
        X_periodic = X_periodic.real
    return X_aperiodic, X_periodic


def reconstruct(
    X,
    d_fit,
    method=None,
    eps=1e-12,
    return_factors=True,
    periodic_mode="quadrature",
    fs=None,
    quadrature_balance_band=None,
    quadrature_pivot_frequency=None,
    quadrature_signs=None,
):
    """Decompose data using fixed Fourier vectors and fitted reference powers.

    Parameters
    ----------
    X : 2d array
        Input data, arranged as observations by time.
    d_fit : 1d array
        Fitted scatter powers in full FFT order.
    method : {None, "polar"}, optional
        Reconstruction method. ``None`` applies the selected Fourier-domain
        decomposition while preserving the observed basis. ``"polar"`` uses
        a polar factor for the left vectors and requires at least as many
        observations as time samples.
    eps : float, optional, default: 1e-12
        Small floor for powers and column norms.
    return_factors : bool, optional, default: True
        If True, also return the explicit Fourier factors ``U``, ``S``, and
        ``F``. Internal callers set this to False because the reconstructed
        time series are all they need, and materializing dense Fourier factors
        is much slower than the FFT/IFFT path.
    periodic_mode : {"additive", "quadrature", "quadrature_phase"}, optional
        Definition of the periodic component. ``method="polar"`` supports
        only ``periodic_mode="additive"``; all other modes require
        the FFT reconstruction method (``method=None``).
    fs : float, optional
        Sampling frequency used by ``quadrature_phase`` settings.
    quadrature_balance_band, quadrature_pivot_frequency, quadrature_signs : optional
        Phase-balancing settings passed to :func:`decompose_fourier`.

    Returns
    -------
    X_ap : 2d array
        Aperiodic component.
    X_periodic : 2d array
        Periodic component. It equals ``X - X_ap`` in every mode.
    U : 2d array
        Left factors.
    S : 2d array
        Diagonal matrix of the realized ``X_ap`` Fourier magnitudes. For
        quadrature modes, non-edge bins match the clipped fitted powers.
    F : 2d array
        Orthonormal Fourier right vectors.
    """
    X = _validate_X(X)
    m, n = X.shape

    d_fit = _validate_full_powers(d_fit, n, "d_fit")
    periodic_mode = _validate_periodic_mode(periodic_mode)
    _validate_quadrature_phase_arguments_for_mode(
        periodic_mode,
        quadrature_balance_band,
        quadrature_pivot_frequency,
        quadrature_signs,
    )
    if method is not None and periodic_mode != "additive":
        raise ValueError(
            "method='polar' supports only periodic_mode='additive'."
        )

    if method is None:
        # Equivalent to X @ F, but avoids building the dense n x n Fourier
        # matrix and doing O(m n^2) matrix multiplies.
        Z = np.fft.fft(X, axis=1) / np.sqrt(n)
        d_raw = np.sum(np.abs(Z) ** 2, axis=0).real
        X_ap, X_periodic = decompose_fourier(
            X,
            d_raw,
            d_fit,
            periodic_mode=periodic_mode,
            eps=eps,
            fs=fs,
            quadrature_balance_band=quadrature_balance_band,
            quadrature_pivot_frequency=quadrature_pivot_frequency,
            quadrature_signs=quadrature_signs,
        )
        if return_factors:
            fitted_fourier = np.fft.fft(X_ap, axis=1) / np.sqrt(n)
            actual_ap_power = np.sum(np.abs(fitted_fourier) ** 2, axis=0).real
            sqrt_actual_ap_power = np.sqrt(np.maximum(actual_ap_power, eps))
            U = fitted_fourier / sqrt_actual_ap_power[None, :]
            F = _fft_matrix(n)
            S = np.diag(sqrt_actual_ap_power)
        else:
            U = S = F = None

    elif method == "polar":
        if m < n:
            raise ValueError("Need n_observations >= n_time for polar reconstruction.")

        sqrt_d = np.sqrt(d_fit)
        F = _fft_matrix(n)
        S = np.diag(sqrt_d)
        Z = X @ F
        M = Z @ S
        P, _, Rh = np.linalg.svd(M, full_matrices=False)
        U = P @ Rh
        X_ap = U @ S @ F.conj().T
        X_periodic = X - X_ap

    else:
        raise ValueError("method must be None or 'polar'.")
    if np.isrealobj(X):
        X_ap = X_ap.real

    return X_ap, X_periodic, U, S, F


def fit(
    X,
    fs,
    psd_model=None,
    method=None,
    eps=1e-12,
    periodic_mode="quadrature",
    quadrature_balance_band=None,
    quadrature_pivot_frequency=None,
    quadrature_signs=None,
):
    """Functional wrapper around :class:`OscEig`.

    Parameters
    ----------
    X : 2d array
        Input data, arranged as observations by time.
    fs : float
        Sampling frequency.
    psd_model : object, optional
        External PSD model implementing ``fit(freqs, powers)`` and exposing
        fitted powers either from the return value or a ``powers_fit``
        attribute.
    method : {None, "polar"}, optional
        Reconstruction method passed to :func:`reconstruct`.
    eps : float, optional, default: 1e-12
        Small numerical floor.
    periodic_mode : str, optional
        Periodic-component definition passed to :class:`OscEig`.
    quadrature_balance_band, quadrature_pivot_frequency, quadrature_signs : optional
        Phase-balancing settings passed to :class:`OscEig`.

    Returns
    -------
    X_ap, X_periodic, U, S, F, d_fit : arrays
        Decomposition results and full-order fitted powers.
    """
    model = OscEig(
        fs=fs,
        psd_model=psd_model,
        method=method,
        eps=eps,
        periodic_mode=periodic_mode,
        quadrature_balance_band=quadrature_balance_band,
        quadrature_pivot_frequency=quadrature_pivot_frequency,
        quadrature_signs=quadrature_signs,
    )
    model.fit(X)

    return (
        model.X_circ_,
        model.X_noncirc_,
        model.U_,
        model.S_,
        model.F_,
        model.powers_decomp_,
    )


class OscEig:
    """Oscillatory eigenspectrum model.

    Parameters
    ----------
    fs : float, optional
        Sampling frequency. If omitted, ``psd_model.fs`` is used when
        available.
    psd_model : object, optional
        External PSD model. It must implement ``fit(freqs, powers)`` and
        provide fitted powers either as ``fit`` output or as a ``powers_fit``
        attribute. This class does not implement or import ARPSD.
    method : {None, "polar"}, optional
        Reconstruction method passed to :func:`reconstruct`.
    fit_dc : bool, optional, default: False
        If True, include DC in the PSD model fit. By default, DC is copied
        from the raw eigenspectrum and the model fits positive non-DC bins.
    split_circulant_first : bool, optional, default: False
        If True, enable an opt-in hierarchical decomposition that first
        separates circulant and non-circulant structure, then decomposes the
        circulant part into aperiodic fit and oscillatory residual. This
        requires ``method="polar"``.
    periodic_mode : {"additive", "quadrature", "quadrature_phase"}, optional
        Definition used to separate the fitted aperiodic and periodic signals.
        The default quadrature construction exactly realizes the fitted and
        positive-excess aggregate powers away from DC and Nyquist. The
        ``quadrature_phase`` construction changes quadrature branch once across
        frequency to balance leading and lagging contributions in a requested
        oscillatory band while retaining the same component powers.
        ``method="polar"`` and ``split_circulant_first=True`` require
        explicitly selecting ``periodic_mode="additive"``.
    quadrature_balance_band : tuple of float, optional
        Frequency band used to derive the ``quadrature_phase`` pivot.
    quadrature_pivot_frequency : float, optional
        Explicit ``quadrature_phase`` pivot instead of deriving one.
    quadrature_signs : 1d array, optional
        Explicit full-order branch signs, normally reused from a fitted model.
    eps : float, optional, default: 1e-12
        Small numerical floor.
    """

    def __init__(
        self,
        fs: Optional[float] = None,
        psd_model=None,
        method=None,
        fit_dc=False,
        split_circulant_first=False,
        periodic_mode="quadrature",
        eps=1e-12,
        quadrature_balance_band=None,
        quadrature_pivot_frequency=None,
        quadrature_signs=None,
    ):
        self.fs = fs
        self.psd_model = psd_model
        self.method = method
        self.fit_dc = fit_dc
        self.split_circulant_first = split_circulant_first
        self.periodic_mode = _validate_periodic_mode(periodic_mode)
        self.eps = eps
        self.quadrature_balance_band = quadrature_balance_band
        self.quadrature_pivot_frequency = quadrature_pivot_frequency
        self.quadrature_signs = quadrature_signs
        _validate_quadrature_phase_arguments_for_mode(
            self.periodic_mode,
            self.quadrature_balance_band,
            self.quadrature_pivot_frequency,
            self.quadrature_signs,
        )

        self.X_ = None
        self.freqs_ = None
        self.freqs_pos_ = None
        self.powers_ = None
        self.powers_pos_ = None
        self.powers_fit_pos_ = None
        self.powers_fit_ = None
        self.powers_decomp_ = None
        self.X_circ_ = None
        self.X_noncirc_ = None
        self.X_circulant_full_ = None
        self.X_aperiodic_ = None
        self.X_periodic_ = None
        self.X_oscillatory_ = None
        self.X_noncirculant_ = None
        self.U_ = None
        self.S_ = None
        self.F_ = None
        self.quadrature_pivot_frequency_ = None
        self.quadrature_signs_ = None

    def fit(self, X, fs=None):
        """Compute the Fourier eigenspectrum, fit PSD powers, and decompose.

        Parameters
        ----------
        X : 2d array
            Input data, arranged as observations by time.
        fs : float, optional
            Sampling frequency. Overrides ``self.fs`` for this fit.

        Returns
        -------
        self : OscEig
            Fitted model.
        """
        X = _validate_X(X)
        fs = self._resolve_fs(fs)

        if self.split_circulant_first and self.method != "polar":
            raise ValueError(
                "split_circulant_first=True requires method='polar' so the "
                "first-stage reconstruction has a Fourier-diagonal covariance."
            )

        freqs, powers = _compute_fourier_scatter(X, fs)
        freqs_pos = np.fft.rfftfreq(X.shape[1], d=1/fs)
        powers_pos = powers[:len(freqs_pos)]

        self.X_ = X
        self.fs = fs
        self.freqs_ = freqs
        self.freqs_pos_ = freqs_pos
        self.powers_ = powers
        self.powers_pos_ = powers_pos

        self.powers_fit_pos_ = self._fit_psd(freqs_pos, powers_pos)

        if self.powers_fit_pos_ is None:
            self.powers_fit_ = None
            self.powers_decomp_ = powers
        else:
            self.powers_fit_ = mirror_pos_to_full(self.powers_fit_pos_, X.shape[1])
            self.powers_decomp_ = self.powers_fit_

        if self.periodic_mode == _QUADRATURE_PHASE_MODE:
            (
                self.quadrature_signs_,
                self.quadrature_pivot_frequency_,
            ) = _resolve_quadrature_phase_signs(
                self.powers_,
                self.powers_decomp_,
                fs=fs,
                balance_band=self.quadrature_balance_band,
                pivot_frequency=self.quadrature_pivot_frequency,
                signs=self.quadrature_signs,
            )
        else:
            _validate_quadrature_phase_arguments_for_mode(
                self.periodic_mode,
                self.quadrature_balance_band,
                self.quadrature_pivot_frequency,
                self.quadrature_signs,
            )

        (
            X_aperiodic,
            X_periodic,
            self.U_,
            self.S_,
            self.F_,
        ) = reconstruct(
            X,
            self.powers_decomp_,
            method=self.method,
            eps=self.eps,
            return_factors=False,
            periodic_mode=self.periodic_mode,
            fs=fs,
            quadrature_signs=self.quadrature_signs_,
        )
        self.X_aperiodic_ = X_aperiodic
        self.X_periodic_ = X_periodic
        self.X_circ_ = self.X_aperiodic_
        self.X_noncirc_ = self.X_periodic_

        return self

    def compute_kappa(self, X=None, **kwargs):
        """Compute how Fourier-circulant the covariance of ``X`` is."""
        if X is None:
            X = self._require_fit("compute_kappa").X_

        return _compute_kappa(X, **kwargs)

    def plot_spectrum(self, ax=None, include_fit=True, positive=True, **plot_kwargs):
        """Plot the eigenspectrum and fitted powers when available.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes on which to draw. Created if omitted.
        include_fit : bool, optional, default: True
            Plot fitted powers if a PSD model was supplied and fit.
        positive : bool, optional, default: True
            Plot positive frequencies only.
        **plot_kwargs
            Additional keyword arguments passed to the raw spectrum plot.

        Returns
        -------
        ax : matplotlib Axes
            Axes containing the plot.
        """
        self._require_fit("plot_spectrum")

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        if positive:
            freqs = self.freqs_pos_
            powers = self.powers_pos_
            powers_fit = self.powers_fit_pos_
        else:
            freqs = self.freqs_
            powers = self.powers_
            powers_fit = self.powers_fit_

        mask = (freqs > 0) & (powers > 0)
        raw_kwargs = {"label": "Eigenspectrum"}
        raw_kwargs.update(plot_kwargs)
        ax.loglog(freqs[mask], powers[mask], **raw_kwargs)

        if include_fit and powers_fit is not None:
            fit_mask = (freqs > 0) & (powers_fit > 0)
            ax.loglog(freqs[fit_mask], powers_fit[fit_mask], label="Fit")

        ax.set_xlabel("Frequency")
        ax.set_ylabel("Power")
        ax.legend()

        return ax

    def decompose(self, X=None, split_circulant_first=None):
        """Return ``(X_aperiodic, X_periodic)`` decomposition components.

        Parameters
        ----------
        X : array, optional
            New signal matrix to decompose with the fixed training spectral
            gain and training-derived ``quadrature_phase`` signs. Otherwise,
            returns the components from ``fit``.
        """
        self._require_fit("decompose")

        if split_circulant_first is None:
            split_circulant_first = self.split_circulant_first
        if split_circulant_first:
            raise NotImplementedError("split_circulant_first is not implemented for decompose.")

        if X is not None:
            X = _validate_X(X)
            n = X.shape[1]
            if self.powers_.shape != (n,) or self.powers_decomp_.shape != (n,):
                raise ValueError("X must have the same number of time samples used during fit.")

            return decompose_fourier(
                X,
                self.powers_,
                self.powers_decomp_,
                periodic_mode=self.periodic_mode,
                eps=self.eps,
                fs=self.fs,
                quadrature_signs=self.quadrature_signs_,
            )

        return self.X_aperiodic_, self.X_periodic_

    def _resolve_fs(self, fs=None):
        fs = self.fs if fs is None else fs

        if fs is None and self.psd_model is not None:
            fs = getattr(self.psd_model, "fs", None)

        if fs is None:
            raise ValueError("Sampling frequency fs is required.")

        return fs

    def _fit_psd(self, freqs_pos, powers_pos):
        if self.psd_model is None:
            return None

        fit_slice = slice(None) if self.fit_dc else slice(1, None)
        fit_freqs = freqs_pos[fit_slice]
        fit_powers = powers_pos[fit_slice]

        fit_result = self.psd_model.fit(fit_freqs, fit_powers)
        powers_fit = _extract_powers_fit(fit_result, self.psd_model)
        powers_fit = np.asarray(powers_fit, dtype=float)

        powers_fit_pos = powers_pos.copy()

        if powers_fit.shape == fit_powers.shape:
            powers_fit_pos[fit_slice] = powers_fit
        elif powers_fit.shape == powers_pos.shape:
            powers_fit_pos = powers_fit
        else:
            raise ValueError(
                "PSD model fitted powers must match either the fitted "
                "frequency bins or all positive-frequency bins."
            )

        return powers_fit_pos

    def _require_fit(self, method_name):
        if self.X_ is None:
            raise RuntimeError(f"Call fit before {method_name}.")

        return self


def _compute_fourier_scatter(X, fs):
    n = X.shape[1]
    freqs = np.fft.fftfreq(n, d=1/fs)
    Z = np.fft.fft(X, axis=1) / np.sqrt(n)
    powers = np.sum(np.abs(Z) ** 2, axis=0).real

    return freqs, powers


def _extract_powers_fit(fit_result, psd_model):
    if isinstance(fit_result, tuple) and len(fit_result) >= 2:
        return fit_result[1]

    if hasattr(fit_result, "powers_fit"):
        return fit_result.powers_fit

    if hasattr(psd_model, "powers_fit"):
        return psd_model.powers_fit

    raise AttributeError(
        "PSD model must return fitted powers or expose a powers_fit attribute."
    )


def _fft_matrix(n):
    return np.fft.fft(np.eye(n), axis=0) / np.sqrt(n)


def _validate_full_powers(powers, n, name):
    powers = np.asarray(powers, dtype=float)
    if powers.shape != (n,):
        raise ValueError(f"{name} must have one value per time sample.")
    if not np.all(np.isfinite(powers)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(powers < 0):
        raise ValueError(f"{name} must be nonnegative.")
    return powers


def _validate_periodic_mode(periodic_mode):
    if periodic_mode not in PERIODIC_MODES:
        choices = ", ".join(sorted(PERIODIC_MODES))
        raise ValueError(f"periodic_mode must be one of: {choices}.")
    return periodic_mode


def _validate_quadrature_phase_arguments_for_mode(
    periodic_mode,
    balance_band,
    pivot_frequency,
    signs,
):
    if periodic_mode == _QUADRATURE_PHASE_MODE:
        return
    if any(value is not None for value in (balance_band, pivot_frequency, signs)):
        raise ValueError(
            "quadrature_balance_band, quadrature_pivot_frequency, and "
            "quadrature_signs are supported only by "
            "periodic_mode='quadrature_phase'."
        )


def _resolve_quadrature_phase_signs(
    d_raw,
    d_fit,
    *,
    fs=None,
    balance_band=None,
    pivot_frequency=None,
    signs=None,
):
    """Resolve one conjugate-symmetric quadrature branch transition."""
    d_raw = np.asarray(d_raw, dtype=float)
    d_fit = np.asarray(d_fit, dtype=float)
    if d_raw.shape != d_fit.shape or d_raw.ndim != 1:
        raise ValueError("d_raw and d_fit must be one-dimensional and equal length.")

    supplied = sum(
        value is not None for value in (balance_band, pivot_frequency, signs)
    )
    if supplied > 1:
        raise ValueError(
            "Specify only one of quadrature_balance_band, "
            "quadrature_pivot_frequency, or quadrature_signs."
        )

    n = len(d_raw)
    if signs is not None:
        resolved = _validate_quadrature_signs(signs, n)
        return resolved, _infer_quadrature_pivot(resolved, fs)

    if fs is None:
        if balance_band is not None or pivot_frequency is not None:
            raise ValueError("fs is required for frequency-based quadrature settings.")
        fs = 1.0
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be a positive finite number.")

    positive_indices = np.arange(1, (n + 1) // 2)
    positive_freqs = np.fft.fftfreq(n, d=1.0 / fs)[positive_indices]
    if len(positive_freqs) == 0:
        return np.zeros(n, dtype=float), None

    if pivot_frequency is not None:
        pivot_frequency = float(pivot_frequency)
        if (
            not np.isfinite(pivot_frequency)
            or pivot_frequency <= 0
            or pivot_frequency >= fs / 2
        ):
            raise ValueError("quadrature_pivot_frequency must lie between 0 and Nyquist.")
        return (
            _quadrature_signs_from_pivot(n, fs, pivot_frequency),
            pivot_frequency,
        )

    if balance_band is None:
        keep = np.ones(len(positive_freqs), dtype=bool)
    else:
        if len(balance_band) != 2:
            raise ValueError("quadrature_balance_band must contain two frequencies.")
        low, high = map(float, balance_band)
        if not np.isfinite(low) or not np.isfinite(high) or low < 0 or high <= low:
            raise ValueError(
                "quadrature_balance_band must be a finite increasing interval."
            )
        keep = (positive_freqs >= low) & (positive_freqs <= high)
        if not np.any(keep):
            raise ValueError("quadrature_balance_band contains no Fourier frequencies.")

    fitted = np.minimum(d_fit, d_raw)
    excess = np.maximum(d_raw - d_fit, 0.0)
    weights = (
        2.0
        * np.pi
        * positive_freqs
        * np.sqrt(fitted[positive_indices] * excess[positive_indices])
    )
    selected_freqs = positive_freqs[keep]
    selected_weights = weights[keep]
    if np.sum(selected_weights) <= 0:
        pivot_frequency = float(np.median(selected_freqs))
    else:
        cumulative = np.cumsum(selected_weights)
        pivot_index = int(np.searchsorted(cumulative, cumulative[-1] / 2.0))
        pivot_frequency = float(selected_freqs[pivot_index])

    return (
        _quadrature_signs_from_pivot(n, fs, pivot_frequency),
        pivot_frequency,
    )


def _quadrature_signs_from_pivot(n, fs, pivot_frequency):
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    signs = np.sign(freqs)
    signs[np.abs(freqs) > pivot_frequency] *= -1.0
    signs[0] = 0.0
    if n % 2 == 0:
        signs[n // 2] = 0.0
    return signs


def _validate_quadrature_signs(signs, n):
    signs = np.asarray(signs, dtype=float)
    if signs.shape != (n,):
        raise ValueError("quadrature_signs must have one value per time sample.")
    if not np.all(np.isfinite(signs)):
        raise ValueError("quadrature_signs must contain only finite values.")

    positive_indices = np.arange(1, (n + 1) // 2)
    if not np.all(np.isin(signs[positive_indices], (-1.0, 1.0))):
        raise ValueError("quadrature_signs must be -1 or +1 away from edge bins.")
    negative_indices = (-positive_indices) % n
    if not np.array_equal(signs[negative_indices], -signs[positive_indices]):
        raise ValueError("quadrature_signs must be antisymmetric across frequencies.")
    if signs[0] != 0.0 or (n % 2 == 0 and signs[n // 2] != 0.0):
        raise ValueError("quadrature_signs must be zero at DC and Nyquist.")
    return signs.copy()


def _infer_quadrature_pivot(signs, fs):
    if fs is None:
        return None
    n = len(signs)
    positive_indices = np.arange(1, (n + 1) // 2)
    positive_signs = signs[positive_indices]
    positive_freqs = np.fft.fftfreq(n, d=1.0 / fs)[positive_indices]
    transitions = np.flatnonzero(np.diff(positive_signs) != 0)
    if len(transitions) == 0:
        return None
    if len(transitions) > 1:
        raise ValueError("quadrature_signs may contain only one positive-frequency transition.")
    return float(positive_freqs[transitions[0]])


def _validate_symmetric_powers(powers, name):
    mirrored = powers[(-np.arange(len(powers))) % len(powers)]
    scale = max(float(np.max(powers)), 1.0)
    if not np.allclose(powers, mirrored, rtol=1e-10, atol=1e-12 * scale):
        raise ValueError(
            f"{name} must be conjugate-frequency symmetric for real input."
        )


def _validate_X(X):
    X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.ndim != 2:
        raise ValueError("X must be 1D or 2D, arranged as observations by time.")

    return X
