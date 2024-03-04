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

def compute_epsilon(X: np.ndarray, vecs: Optional[np.ndarray]=None) -> float:
    """Compute epsilon, a measure of how circulant a matrix is.

    Parameters
    ----------
    X : 2d array
        Arbitrary square matrix.
        If from sim_circulant, epilson -> 0.
    vecs : 2d array, optional, default: None
        Eigenvectors to assume. Defaults to Fourier modes.

    Returns
    -------
    epsilon : float
        A measure of how diagonal the eigenvalues are.
        Off-digonal values increase episolon.

    Notes
    -----
    \epsilon = \frac{1}{n} ||\Lambda - \lambda_ij \odot I||^2
    """
    n = len(X)

    # Fourier modes (e.g. cosines)
    if vecs is None:
        vecs = np.fft.fft(np.eye(n)) / np.sqrt(n)

    # Eigenvalues given cosines as eigenvectors
    Lambda = (vecs.conj().T @ X @ vecs).real

    # Error measure
    epsilon = (((Lambda * np.eye(len(Lambda))) - Lambda)**2).mean()

    return epsilon
