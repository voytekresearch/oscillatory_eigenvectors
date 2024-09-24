"""Powers spectral density."""

import numpy as np


def compute_eigenspectrum(cov, fs):
    """Compute PSD using: fft -> eigvals -> PSD.

    Parameters
    ----------
    cov : 2d array
        Covariance matrix.

    Returns
    -------
    freqs : 1d array
        Frequencies.
    powers : 1d array
        Power spectral density.
    """

    # 2d fft
    coefs = np.fft.fft2(cov, norm="ortho")

    # fft -> eigenvalues
    #   eigvals of cov == power
    powers = np.diag(np.roll(coefs.real[:, ::-1], 1))
    powers = powers.copy()
    powers[0] = np.sum(cov) / len(cov)

    # Take positive powers
    n = int(np.ceil(len(cov)/2))
    powers = powers[:n]

    # Frequencies
    freqs = np.fft.fftfreq(len(coefs), 1/fs)[:n]

    return freqs, powers
