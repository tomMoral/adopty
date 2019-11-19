"""Ista algorithm for the LASSO
"""

import numpy as np
from time import time

from .loss_and_gradient import l2, cost_lasso, grad, soft_thresholding


def backtracking_ista(D, x, lmbd, eta=0.99, z_init=None, max_iter=100,
                      stopping_criterion=None):
    """ISTA for resolution of the sparse coding with backtracking line search

    Parameters
    ----------
    D : array, shape (n_atoms, n_dim)
        Dictionary used for the sparse coding
    x : array, shape (n_samples, n_dim)
        Signal to encode on D
    lmbd : float
        Regularization parameter of the sparse coding.
    eta : float
        Backtracking parameter.
    z_init : array, shape (n_trial, n_atoms) or None
        Initial value of the activation codes
    max_iter : int
        Maximal number of iteration for ISTA
    stopping_criterion: callable or None
        If it is a callable, it is call with the list of the past costs
        and the algorithm is stopped if it returns True.

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

    step_size = 1

    # Generate an initial point
    if z_init is not None:
        z_hat = np.copy(z_init)
    else:
        z_hat = np.zeros((n_samples, n_atoms))

    times = []
    steps = []
    cost_ista = [cost_lasso(z_hat, D, x, lmbd)]
    for _ in range(max_iter):

        l2_0 = l2(z_hat, D, x)
        grad_l2_0 = grad(z_hat, D, x)

        step_size = 1000
        while not is_valid_t(z_hat, D, x, lmbd, l2_0, grad_l2_0, step_size):
            step_size *= eta

        t_start_iter = time()
        z_hat -= step_size * grad(z_hat, D, x)
        z_hat = soft_thresholding(z_hat, lmbd * step_size)
        times += [time() - t_start_iter]
        steps.append(step_size)

        cost_ista += [cost_lasso(z_hat, D, x, lmbd)]

        # Stopping criterion for the convergence
        if callable(stopping_criterion) and stopping_criterion(cost_ista):
            break

    return z_hat, cost_ista, times, steps


def is_valid_t(z_hat, D, x, lmbd, l2_0, grad_l2_0, t):
    xt = soft_thresholding(z_hat - t * grad_l2_0, lmbd * t)
    Gt = (z_hat - xt) / t
    cost_t = l2(xt, D, x)
    surrogate_t = (l2_0 - t * Gt.ravel().dot(grad_l2_0.ravel())
                   + .5 * t * np.sum(Gt * Gt))
    return cost_t <= surrogate_t
