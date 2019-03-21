"""Plot the initial cost function
=================================

Plot the initial cost function for all the methods:
    * :func:ista
    * :func:fista
    * :func:lista
    * ...
"""
import numpy as np
import matplotlib.pyplot as plt

from adopty.ista import ista
from adopty.datasets import make_coding


if __name__ == "__main__":

    reg = .5
    n_iters = 100

    n_dim = 64
    n_atoms = 100
    n_samples = 1000
    random_state = 42

    # Generate a problem
    x, D, z, lmbd_max = make_coding(n_samples=n_samples, n_atoms=n_atoms,
                                    n_dim=n_dim, random_state=random_state)

    # Compute the effective regularization
    lmbd = lmbd_max * reg

    _, cost_ista, _ = ista(D, x, lmbd, max_iter=100)

    cost_ista = np.array(cost_ista)
    plt.loglog(cost_ista - cost_ista.min() + 1e-16)
    plt.show()
