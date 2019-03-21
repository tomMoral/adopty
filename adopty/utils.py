import numpy as np


def cost(z_hat, D, x, lmbd, flatten=False):
    """Cost of the LASSO
    """
    if flatten:
        n_samples = x.shape[0]
        z_hat = z_hat.reshape((n_samples, -1))
    res = z_hat.dot(D) - x
    return .5 * (res * res).sum() + lmbd * abs(z_hat).sum()


def grad(z_hat, D, x, lmbd=None, flatten=False, return_func=False):
    """Gradient of the LASSO
    """
    if flatten:
        n_samples = x.shape[0]
        z_hat = z_hat.reshape((n_samples, -1))
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


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
