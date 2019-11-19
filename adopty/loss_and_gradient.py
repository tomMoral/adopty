import numpy as np


def l2(z_hat, D, x):
    """Cost for the data fit term"""
    res = z_hat.dot(D) - x
    return .5 * (res * res).sum()


def cost_lasso(z_hat, D, x, lmbd, flatten=False):
    """Cost of the LASSO
    """
    n_samples = x.shape[0]
    if flatten:
        z_hat = z_hat.reshape((n_samples, -1))
    return (l2(z_hat, D, x) + lmbd * abs(z_hat).sum()) / n_samples


def grad(z_hat, D, x, lmbd=None, flatten=False, return_func=False):
    """Gradient of the LASSO
    """
    n_samples = x.shape[0]
    if flatten:
        z_hat = z_hat.reshape((n_samples, -1))
    grad = (z_hat.dot(D) - x).dot(D.T)

    if flatten:
        grad = grad.ravel()

    if return_func:
        return cost_lasso(z_hat, D, x, lmbd), grad

    return grad


def soft_thresholding(z, mu):
    return np.maximum(abs(z) - mu, 0) * np.sign(z)


def get_lmbd_max(D, x):
    # Compute the effective regularization
    return x.dot(D.T).max()
