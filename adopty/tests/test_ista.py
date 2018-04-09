import pytest
import numpy as np
from scipy import optimize


from adopty.ista import ista
from adopty.lista import Lista
from adopty.utils import cost, grad
from adopty.tests.utils import gradient_checker


@pytest.mark.parametrize('reg', [.1, .3, .5])
def test_ista(reg):

    n_trials = 1000
    n_atoms = 100
    n_dimensions = 64

    n_iters = 100

    random_state = 42

    rng = np.random.RandomState(random_state)

    # Generate a problem
    D = rng.randn(n_atoms, n_dimensions)
    z = rng.randn(n_trials, n_atoms)
    x = z.dot(D)

    z_hat, cost_ista, *_ = ista(D, x, reg, max_iter=n_iters)

    assert all(np.diff(cost_ista) <= 0)


@pytest.mark.parametrize('reg', [.1, .3, .5, 2])
def test_lista(reg):

    n_trials = 1000
    n_atoms = 100
    n_dimensions = 64

    n_iters = 100

    n_layers = 10
    random_state = 42

    rng = np.random.RandomState(random_state)

    # Generate a problem
    D = rng.randn(n_atoms, n_dimensions)
    z = rng.randn(n_trials, n_atoms)
    x = z.dot(D)

    z_hat, cost_ista, _, lmbd = ista(D, x, reg, max_iter=n_layers)

    lista = Lista(D, n_layers)
    z_lista = lista(x, lmbd)

    z_lista = z_lista.data.numpy()
    assert np.isclose(cost_ista[n_layers], cost(z_lista, D, x, lmbd))


def test_grad():

    n_trials = 10
    n_atoms = 20
    n_dimensions = 64

    random_state = 1729

    rng = np.random.RandomState(random_state)

    # Generate a problem
    D = rng.randn(n_atoms, n_dimensions)
    z = rng.randn(n_trials, n_atoms)
    x = z.dot(D)

    z = z.ravel()

    gradient_checker(cost, grad, n_trials * n_atoms, args=(D, x),
                     kwargs=dict(lmbd=0, flatten=True), n_checks=100,
                     debug=True)
