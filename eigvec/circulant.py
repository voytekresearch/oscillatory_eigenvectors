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


def compute_mse(X: np.ndarray, vecs: Optional[np.ndarray]=None) -> float:
    """Compute MSE, a measure of how circulant a matrix is.

    Parameters
    ----------
    X : 2d array
        Arbitrary square matrix.
        If from sim_circulant, epilson -> 0.
    vecs : 2d array, optional, default: None
        Eigenvectors to assume. Defaults to Fourier modes.

    Returns
    -------
    mse : float
        A measure of how diagonal the eigenvalues are.
        Off-digonal values increase episolon.

    Notes
    -----
    \text{MSE} = \frac{1}{n} ||\Lambda - \Lambda \odot I||^2
    """

    # Fourier modes (e.g. cosines)
    if vecs is None:
        n = len(X)
        vecs = np.fft.fft(np.eye(n)) / np.sqrt(n)

    # Eigenvalues given cosines as eigenvectors
    Lambda = (vecs.conj().T @ X @ vecs).real

    # Error measure
    mse = (((Lambda * np.eye(len(Lambda))) - Lambda)**2).mean()

    return mse


def compute_kappa(X: np.ndarray, vecs: Optional[np.ndarray]=None) -> float:
    """Compute kappa, a more stringent measure of how circulant a matrix is.

    Parameters
    ----------
    X : 2d array
        Arbitrary square matrix.
        If from sim_circulant, kapppa -> 1.
    vecs : 2d array, optional, default: None
        Eigenvectors to assume. Defaults to Fourier modes.

    Returns
    -------
    kappa : float
        A measure of how diagonal the eigenvalues are.
        Off-digonal values decreases kappa.

    Notes
    -----
    \kappa &= \frac{\sum_{i=0}^{n} \Lambda^2_{i, i}}{\sum_{i=0}^{n}\sum_{j=0}^{n}|\Lambda^2_{i, j}|}
    """

    # Fourier modes (e.g. cosines)
    if vecs is None:
        n = len(X)
        vecs = np.fft.fft(np.eye(n)) / np.sqrt(n)

    # Eigenvalues given cosines as eigenvectors
    Lambda = (vecs.conj().T @ X @ vecs).real

    # Kappa
    kappa = (np.diag(Lambda)**2).sum() / (Lambda**2).sum()

    return kappa