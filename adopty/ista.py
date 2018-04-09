"""Ista algorithm for the LASSO
"""

import numpy as np
from time import time

from .utils import cost


def ista(D, x, reg, z_init=None, max_iter=100):
    """ISTA for resolution of the sparse coding

    Parameters
    ----------
    D : array, shape (n_atoms, n_dimensions)
        Dictionary used for the sparse coding
    x : array, shape (n_trials, n_dimensions)
        Signal to encode on D
    reg : float
        Regularization parameter of the sparse coding as a ratio of lambda_max
    z_init : array, shape (n_trial, n_atoms) or None
        Initial value of the activation codes
    max_iter : int
        Maximal number of iteration for ISTA

    Returns
    -------
    z_hat : array, shape (n_trial, n_atoms)
        Estimated sparse codes
    cost_ista : list
        Cost accross the iterations
    times : list
        Time taken by each iteration
    """
    n_trials = x.shape[0]
    n_atoms = D.shape[0]

    # Compute the effective regularization
    lmbd_max = x.dot(D.T).max()
    lmbd = lmbd_max * reg

    L = np.linalg.norm(D.dot(D.T), 2)
    step_size = 1 / L

    # Generate an initial point
    if z_init:
        z_hat = np.copy(z_init)
    else:
        z_hat = np.zeros((n_trials, n_atoms))

    times = []
    cost_ista = [cost(z_hat, D, x, lmbd)]
    for _ in range(max_iter):
        t_start_iter = time()
        z_hat -= step_size * (z_hat.dot(D) - x).dot(D.T)
        z_hat = np.maximum(abs(z_hat) - lmbd * step_size, 0) * np.sign(z_hat)
        times += [time() - t_start_iter]

        cost_ista += [cost(z_hat, D, x, lmbd)]

    return z_hat, cost_ista, times, lmbd