"""Dataset utilities, for simulated and real examples
"""
import numpy as np
from .utils import check_random_state


def make_coding(n_samples=1000, n_atoms=10, n_dim=3, random_state=None):
    """Simulate a sparse coding problem  with no noise


    Parameters
    ----------
    n_samples : int (default: 1000)
        Number of samples in X
    n_atoms : int (default: 3)
        Number of atoms in the dictionary
    n_dim : in (default: 10)
        Number of dimension for the observation space
    random_state: None or int or RandomState
        Random state for the random number generator.

    Return
    ------
    x : ndarray, shape (n_samples, n_dim)
        observation
    D : ndarray, shape (n_atoms, n_dim)
        dictionary of atoms used to generate the observation
    z : ndarray, shape (n_samples, n_atoms)
        activation associated to each observation for the dictionary D
    lmbd_max : float
        Minimal value of lmbd_max for which 0 is solution of the LASSO for
        x and D fixed.
    """

    rng = check_random_state(random_state)

    # Generate a problem
    D = rng.randn(n_atoms, n_dim)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    z = rng.randn(n_samples, n_atoms)
    x = z.dot(D)

    # Compute the effective regularization
    lmbd_max = x.dot(D.T)
    x /= abs(lmbd_max).max(axis=1, keepdims=True)

    lmbd_max = x.dot(D.T)

    return x, D, z
