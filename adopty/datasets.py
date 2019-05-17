"""Dataset utilities, for simulated and real examples
"""
import numpy as np
from .utils import check_random_state

from .ista import ista
# from .lista import Lista


def make_coding(n_samples=1000, n_atoms=10, n_dim=3, normalize=True,
                random_state=None):
    """Simulate a sparse coding problem  with no noise


    Parameters
    ----------
    n_samples : int (default: 1000)
        Number of samples in X
    n_atoms : int (default: 3)
        Number of atoms in the dictionary
    n_dim : in (default: 10)
        Number of dimension for the observation space
    normalize : bool (default: True)
        If set to True, normalize each atom in the dictionary
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
    if normalize:
        D /= np.linalg.norm(D, axis=1, keepdims=True)
    z = rng.randn(n_samples, n_atoms)
    x = z.dot(D)

    # Compute the effective regularization
    lmbd_max = x.dot(D.T)
    x /= abs(lmbd_max).max(axis=1, keepdims=True)

    lmbd_max = x.dot(D.T)

    return x, D, z


def make_sparse_coding(n_samples=1000, n_atoms=10, n_dim=3, reg=.1,
                       sparsity_filter="<2", normalize=True,
                       random_state=None):
    """Simulate a sparse coding problem  with no noise


    Parameters
    ----------
    n_samples : int (default: 1000)
        Number of samples in X
    n_atoms : int (default: 3)
        Number of atoms in the dictionary
    n_dim : int (default: 10)
        Number of dimension for the observation space
    reg : float (default: .1)
        Regularization level
    sparsity_filter: str (default: '<2')
        Filter to select the sparsity of the solution given by ISTA for the
        given reg level. The first character of the string is an operator in
        '=', '<' or '>' and the rest of the string must be convertible to an
        integer. For instance, '<2' will return all samples with solution with
        only one non-zero coefficient.
    normalize : bool (default: True)
        If set to True, normalize each atom in the dictionary
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
    if normalize:
        D /= np.linalg.norm(D, axis=1, keepdims=True)
    z = 10 * rng.randn(n_samples * 5, n_atoms)
    x = z.dot(D)

    # Compute the effective regularization
    lmbd_max = x.dot(D.T)
    x /= abs(lmbd_max).max(axis=1, keepdims=True)

    mask = filter_sparse_set(x, D, reg, sparsity_filter)

    return x[mask][:n_samples], D, z[mask][:n_samples]


def filter_sparse_set(x, D, lmbd, sparsity_filter="=1"):

    # z_hat = Lista(D, n_layers=30).transform(x, lmbd)
    z_hat, _, _ = ista(D, x, lmbd, max_iter=10000, tol=0)
    operator = sparsity_filter[0]
    sparsity = int(sparsity_filter[1:])
    z_sparsity = np.sum(abs(z_hat) > 1e-2, axis=1)
    if operator == "=":
        mask = z_sparsity == sparsity
    elif operator == "<":
        mask = z_sparsity < sparsity
    elif operator == ">":
        mask = z_sparsity > sparsity
    else:
        raise NotImplementedError("operator should be '=', '<' or '>'. "
                                  "Got '{}'".format(operator))

    return mask
