"""Powers spectral density."""

import numpy as np


def fft_to_eigvals_to_psd(X):
    """Compute PSD using: fft -> eigvals -> PSD.

    Parameters
    ----------
    X : 2d array
        Input matrix, e.g. covariance.

    Returns
    -------
    freqs : 1d array
        Frequencies.
    powers : 1d array
        Power spectral density.
    """
    # 2d fft
    coefs = np.fft.fft2(X, norm="ortho")

    # fft -> eigenvalues
    #   eigvals of cov == power
    powers = fftcoefs_to_eigvals(coefs).real
    powers = powers[:len(coefs)//2-1]

    # frequencies
    freqs = np.fft.fftfreq(len(coefs), 1)[:len(coefs)//2]
    freqs= freqs[1:]

    return freqs, powers


def fftcoefs_to_eigvals(coefs):
    """Extracts eigenvalues out of the 2d fft.

    Parameters
    ----------
    coefs : 2d array
        2d FFT coefficients.

    Returns
    -------
    1d array
        Diagonal, from lower left to upper right.

    Notes
    -----
    1. Gets these indices out of the 2d fft:
       [1, -1], [2, -2], [3, -3] ...
    2. The DC offset is not included, expect n-1 values.
    """
    return np.diag(coefs[:, ::-1], -1)
