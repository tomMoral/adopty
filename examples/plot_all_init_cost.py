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


if __name__ == "__main__":

    n_trials = 1000
    n_atoms = 100
    n_dimensions = 64
    reg = .5

    n_iters = 100

    random_state = 42

    rng = np.random.RandomState(random_state)

    # Generate a problem
    D = rng.randn(n_atoms, n_dimensions)
    z = rng.randn(n_trials, n_atoms)
    x = z.dot(D)

    z_hat, cost_ista, _ = ista(D, x, reg, max_iter=100)

    cost_ista = np.array(cost_ista)
    plt.loglog(cost_ista - cost_ista.min() + 1e-16)
    plt.show()
