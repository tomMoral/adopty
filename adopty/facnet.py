from .utils import grad
from .utils import soft_thresholding


def facnet_step(z_q, D, x, lmbd, A, S):
    """Facnet step

    Parameters
    ----------
    z_q : ndarray, shape (n_trials, n_atoms)
        current solution estimate
    D : ndarray, shape (n_atoms, n_dims)
        Dictionary used to encode the data
    x : ndarray, shape (n_trial, n_dims)
        Data encoded with Facnet
    lmbd : float
        Regularization parameter
    A : ndarray, shape (n_atoms, n_atoms)
        Rotation parameter for Facnet step
    S : ndarray, shape (1, n_atoms)
        Diagonal matrix considered for Facnet step
    """
    G_q = grad(z_q, D, x)
    y_q = z_q.dot(A) - G_q.dot(A) / S

    return soft_thresholding(y_q, lmbd / S).dot(A.T)
