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

def compute_kappa(X, method='svd', vecs=None, normalize=True):
    """Compute kappa, a measure of how circulant a matrix is.

    Parameters
    ----------
    X : 2d array
        An n-by-m matrix.
    method : {'svd', 'eig'}
        How to compute the eigenvalues. Approximately equivalent.
        'svd' is orders of magnitude faster when m is large.
    vecs : 2d array, default: None
        Matrix of eigenvectors or, equivalently, right singular vectors.
        Decreases compute time if provided, useful for computing many kappa
        with many X that have same number columns.
    normalize : bool, optional default: True
        How to normalize X. True is equivalent to np.cov(X.T). False is
        equivalent to X.T @ X.
    """
    n = len(X[1]) - 1

    if normalize:
        X = X - X.mean(axis=0)
        norm = len(X[1]) - 1
    else:
        norm = 1

    if method == 'eig':
        cov = X.T @ X
        cov = cov / norm
        vals, vecs = compute_eig(cov)

    elif method == 'svd':
        X = X / norm
        U, S, V = compute_svd(X)
        vals = S

    # Kappa
    vals = np.abs(vals)
    n = len(X[0])
    trace = np.diag(vals).sum()
    diag_mean = trace / n
    off_diag_mean = (vals.sum() - trace) / (n**2 - n)
    kappa = diag_mean / (diag_mean + off_diag_mean)

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

