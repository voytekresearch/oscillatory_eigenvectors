"""Singular spectrum analysis (SSA)."""

import numpy as np
from scipy import linalg
from typing import Optional, Callable

class SSA:
    """Singular spectrum analysis.

    Attributes
    ----------
    sig_components : 2d array.
        Signal components reconstructed from SVD.
    U, S, V : 2d array, 1d array, 2d array, optional
        SVD components.
    """
    def __init__(self, window_len:int, group_size:int, n_components:Optional[int]=None,
                 svd_solver:Optional[Callable]=None, store_svd:Optional[bool]=True):
        """Initialize.

        Parameters
        ----------
        window_len : int
            Size of window to slide along time series.
            Same as number of columns in the Hankel matrix.
        group_size : int, optional, default: 1
            Number of components to group together (e.g. subsets of U, S, V).
        n_components : int
            Max number of components, e.g. rank-1 matrices, to keep.
        svd_solver : func, optional, default: None
            Function to solve SVD. Defaults to scipy.linalg.svd.
            Should return numpy arrays as U, S, V.
        stores_svd : bool, optional, default: False
            Keeps U, S, V in associated attributes if True.
            Use False if they are not needed or there are memory limits.
        """
        # Parameters / options
        self.window_len = window_len
        self.group_size = group_size
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.store_svd = store_svd

        # Results: right/left singular vectors and singular values
        self.U = None
        self.S = None
        self.V = None

        # Results: signal components
        self.sig_components = None


    def fit(self, X:np.ndarray):
        """Run the decomposition and reconstruction.

        Parameters
        ----------
        X : 1d or 2d array
            Signal or signals in rows.
        """
        if X.ndim == 1:
            # Enforce 2d
            X = X.reshape(1, -1)

        # Diagonal counts
        n_avg = diagonal_counts(len(X[0])-self.window_len+1, self.window_len)

        # Run SSA
        for i, x in enumerate(X):

            _sig_components, _U, _S, _V = singular_spectrum_analysis(
                x, self.window_len, self.group_size, self.n_components, return_svd=True,
                svd_solver=self.svd_solver, n_avg=n_avg
            )

            if i == 0:
                # Initalize arrays
                self.sig_components = np.zeros((len(X), len(_sig_components)))

                if self.store_svd:
                    self.U = np.zeros((len(X), *_U.shape))
                    self.S = np.zeros((len(X), *_S.shape))
                    self.V = np.zeros((len(X), *_V.shape))

            self.sig_components[i] = _sig_components

            if self.store_svd:
                self.U[i] = _U
                self.S[i] = _S
                self.V[i] = _V.T


def singular_spectrum_analysis(sig:np.ndarray, window_len:int, n_components:int, group_size:Optional[int]=1,
                               svd_solver:Optional[Callable]=None, diag_counts:Optional[np.ndarray]=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute singular spectrum analysis.

    Parameters
    ----------
    sig : 1d array
        Signal time series.
    window_len : int
        Size of window to slide along time series.
        Same as number of columns in the Hankel matrix.
    n_components : int
        Max number of components, e.g. rank-1 matrices, to keep.
    group_size : int, optional, default: 1
        Number of components to group together (e.g. subsets of U, S, V).
    svd_solver : callable, optional, default: None
        Function to solve SVD. Defaults to scipy.linalg.svd.
        Should return numpy arrays as U, S, V.
    diag_counts : 1d array, optional, default None
        Number of elements along each diagonal. Computes if None.

    Returns
    -------
    sig_components : 2d array.
        Signal components reconstructed from SVD.
    U, S, V : 2d array, 1d array, 2d array, optional
        SVD components when return_svd is True.
    """
    # 1.1 Decomposition: Embedding
    X_hankel = np.lib.stride_tricks.sliding_window_view(sig, window_len, axis=0)

    # 1.2 Decomposition: SVD
    if svd_solver is None:
        U, S, Vt = linalg.svd(X_hankel, full_matrices=False)
        V = Vt.T

    if n_components is not None:
        U = U[:, :n_components]
        S = S[:n_components]
        V = V[:, :n_components]
    else:
        n_components = len(S)

    if diag_counts is None:
        counts = diagonal_counts(X_hankel.shape[0], X_hankel.shape[1])

    # 2. Reconstruction
    sig_components = np.zeros((int(np.ceil(n_components/group_size)), len(sig)))

    for i_sig, i_group in enumerate(range(0, n_components, group_size)):

        # 2.1 Reconstruction: Grouping
        X_hat = U[:, i_group:i_group+group_size] * S[i_group:i_group+group_size] \
            @ V[:, i_group:i_group+group_size].T

        # 2.2 Reconstruction: Diagonal averaging
        diag = np.zeros(len(counts))
        for i in range(X_hat.shape[0]):
            for j in range(X_hat.shape[1]):
                diag[i+j] += X_hat[i, j]

        sig_components[i_sig] = diag / counts

    return sig_components, U, S, V


def diagonal_counts(n:int, m:int) -> np.ndarray:
    """Get number of elements along diagonal of matrix.

    Parameters
    ----------
    n : int
        Number of rows in the Hankel matrix.
    m : int
        Number of columns in the Hankel matrix.

    Returns
    -------
    counts : 1d array
        Number of elements on each diagonal.

    Notes
    -----
    Count n elements along each diagonal of the Hankel matrix,
    e.g. [[x1, x2, x3]
          [x2, x3, x4], -> [1, 2, 3, 2, 1]
          [x3, x4, x5]]
    """
    n_diag = n+m-1
    n, m = n-m+1, m
    dmin = min(n, m)
    counts = np.zeros(n_diag)
    for i in range(dmin):
        # Increasing to max dim
        counts[i] = i+1
        counts[-(i+1)] = i+1
    counts[(i+1):-(i+1)] = counts[i]
    return counts
