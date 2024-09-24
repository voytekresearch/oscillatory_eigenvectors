"""Eigenspectra via FFT."""

import numpy as np


def compute_eigenspectrum(X, fs, return_neg=True):
    """Compute PSD from circulant eigendecomposition.

    Parameters
    ----------
    X : 2d array
        Matrix of signals.
    return_neg : bool, default: True
        Returns negative and positive frequencies if True.
        Return only positive if False

    Returns
    -------
    freqs : 1d array
        Frequencies.
    powers : 1d array
        Power spectral density.

    Notes
    -----
    - Uses SVD to accelerate.
    - eigenvals = US.conj().T @ US
    - This varies from what is in spectral.py,
      which relies on the covariance matrix.
    """

    # Compute singular values
    n = len(X[0])
    US = np.fft.fft(X, axis=1) / np.sqrt(n)
    S = np.linalg.norm(US, axis=0)

    powers = S**2 / n
    freqs = np.fft.fftfreq(n, 1/fs)

    if not return_neg:
        inds = freqs >= 0
        freqs = freqs[inds]
        powers = powers[inds]

    return freqs, powers


def compute_kappa(X):
    """Compute kappa, a measure of how circulant a matrix is.

    Parameters
    ----------
    X : 2d array
        An n-by-m matrix.
    """
    # Use SVD to compute eigenvalues
    n = len(X[0])
    US = np.fft.fft(X, axis=1) / np.sqrt(n)

    # (U S)^T U S == S U^T U S == S^2 == eigenvalues
    vals = (US.conj().T @ US).real / n

    # Kappa
    vals = np.abs(vals)
    trace = np.diag(vals).sum()
    diag_mean = trace / n
    off_diag_mean = (vals.sum() - trace) / (n**2 - n)
    kappa = diag_mean / (diag_mean + off_diag_mean)

    return kappa