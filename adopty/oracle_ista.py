"""Ista algorithm for the LASSO
"""

import numpy as np
from time import time

from .loss_and_gradient import cost_lasso, grad, soft_thresholding


def oracle_step(D):
    """
    Gives the L constant of D
    """
    L = np.linalg.norm(D.dot(D.T), 2)
    return 1 / L


def one_ista(x, D, z, lmbd, step_size):
    """
    One soft thresholing step
    """
    y = z - step_size * grad(z, D, x)
    return soft_thresholding(y, lmbd * step_size)


def oracle_ista(D, x, lmbd, z_init=None, max_iter=100):
    """Oracle ISTA for resolution of the sparse coding

    Parameters
    ----------
    D : array, shape (n_atoms, n_dim)
        Dictionary used for the sparse coding
    x : array, shape (n_samples, n_dim)
        Signal to encode on D
    lmbd : float
        Regularization parameter of the sparse coding.
    z_init : array, shape (n_trial, n_atoms) or None
        Initial value of the activation codes
    max_iter : int
        Maximal number of iteration for ISTA

    Returns
    -------
    z_hat : array, shape (n_trial, n_atoms)
        Estimated sparse codes
    cost_ista : list
        Cost across the iterations
    times : list
        Time taken by each iteration
    """
    n_samples = x.shape[0]
    n_atoms = D.shape[0]

    L = np.linalg.norm(D.dot(D.T), 2)
    # Generate an initial point
    if z_init is not None:
        z_hat = np.copy(z_init)
    else:
        z_hat = np.zeros(n_atoms)
    times = []
    cost_ista = [cost_lasso(z_hat, D, x, lmbd) * n_samples]
    steps = []
    for _ in range(max_iter):
        t_start_iter = time()
        support = z_hat != 0
        size = np.sum(support)
        if size == 0:  # anoying case
            step_size = 1 / L
        else:
            idx = np.where(support)[0]
            step_size = oracle_step(D[idx])
        y_hat = one_ista(x, D, z_hat, lmbd, step_size)
        support_y = y_hat != 0
        if np.sum(support_y * support) == np.sum(support_y):  # good step
            z_hat = y_hat
        else:   # bad step
            step_size = 1 / L
            z_hat = one_ista(x, D, z_hat, lmbd, step_size)
            cost_ista += [cost_lasso(z_hat, D, x, lmbd) * n_samples]
            steps.append(step_size)
        steps.append(step_size)
        times += [time() - t_start_iter]
        cost_ista += [cost_lasso(z_hat, D, x, lmbd) * n_samples]
    return z_hat, cost_ista, times, steps
