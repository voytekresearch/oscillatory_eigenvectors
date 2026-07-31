"""Cirulant matrices, measures and reshaping."""

from typing import Optional
import numpy as np


def sim_circulant(sig: np.ndarray) -> np.ndarray:
    """Simulates a circulant matrix.

    Parameters
    ----------
    sig : 1d array
        Signal time series.

    Returns
    -------
    X : 2d array
        Circulant matrix from rolling sig.
    """
    n = len(sig)
    X = np.zeros((n, n))
    for i in range(len(sig)):
        X[i] = np.roll(sig, i)
    return X


def compute_kappa(
    X,
    normalize=True,
    use_fft=True,
    eps=1e-12,
):
    """
    Estimate population-level Fourier/circulant diagonality.

    Uses:
        A = F* C F

    and compares observed off-diagonal Fourier covariance energy to the
    expected finite-sample off-diagonal energy under a Fourier-diagonal
    population covariance.

    This makes white noise score near 1 across matrix sizes.
    """
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must be 2D: rows=observations, cols=time.")

    m, n = X.shape
    df = m
    norm = m if normalize else 1.0

    if df <= 0:
        raise ValueError("Need at least two rows.")

    if use_fft:
        Z = np.fft.fft(X, axis=1) / np.sqrt(n)
        A = (Z.conj().T @ Z) / norm
    else:
        F = np.fft.fft(np.eye(n), axis=0) / np.sqrt(n)
        C = (X.T @ X) / norm
        A = F.conj().T @ C @ F

    d = np.real(np.diag(A))
    d = np.maximum(d, eps)

    total_sq = np.sum(np.abs(A) ** 2)
    diag_sq = np.sum(d ** 2)
    off_sq = total_sq - diag_sq

    # Expected finite-sample off-diagonal energy under diagonal Fourier covariance.

    # For i != j:
    #   E |A_ij|^2 ≈ d_i d_j / df

    # Summed over i != j:
    #   sum_{i != j} d_i d_j / df
    expected_off_sq = ((d.sum() ** 2) - np.sum(d ** 2)) / df

    # Excess off-diagonal energy beyond finite-sample floor
    excess_off_sq = max(off_sq - expected_off_sq, 0.0)

    kappa = diag_sq / (diag_sq + excess_off_sq + eps)
    kappa = float(np.real(kappa))

    return kappa


def compute_svd(X, V=None):
    n = len(X[0])
    if V is None:
        V = np.fft.fft(np.eye(n)) / np.sqrt(n)
    U = X @ V
    S = U.conj().T @ U
    return U, S, V

def compute_eig(cov, vecs=None):
    n = len(cov)
    if vecs is None:
        vecs = np.fft.fft(np.eye(n)) / np.sqrt(n)
    vals = vecs.conj().T @ cov @ vecs
    return vals, vecs

