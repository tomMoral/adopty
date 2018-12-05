import numpy as np


def cost(z_hat, D, x, lmbd, flatten=False):
    """Cost of the LASSO
    """
    if flatten:
        n_trials = x.shape[0]
        z_hat = z_hat.reshape((n_trials, -1))
    res = z_hat.dot(D) - x
    return .5 * (res * res).sum() + lmbd * abs(z_hat).sum()


def grad(z_hat, D, x, lmbd=None, flatten=False, return_func=False):
    """Gradient of the LASSO
    """
    if flatten:
        n_trials = x.shape[0]
        z_hat = z_hat.reshape((n_trials, -1))
    grad = (z_hat.dot(D) - x).dot(D.T)

    if flatten:
        grad = grad.ravel()

    if return_func:
        return cost(z_hat, D, x, lmbd), grad

    return grad


def soft_thresholding(z, mu):
    return np.maximum(abs(z) - mu, 0) * np.sign(z)


def get_lmbd_max(D, x):
    # Compute the effective regularization
    return x.dot(D.T).max()
