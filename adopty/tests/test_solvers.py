import pytest
import numpy as np


from adopty.ista import ista
from adopty.fista import fista
from adopty.lista import Lista
from adopty.utils import cost, grad
from adopty.facnet import facnet_step
from adopty.datasets import make_coding
from adopty.tests.utils import gradient_checker


@pytest.mark.parametrize('reg', [.1, .3, .5])
def test_ista(reg):

    n_dim = 20
    n_atoms = 10
    n_samples = 1000
    random_state = 42

    n_iters = 100

    # Generate a problem
    x, D, z, lmbd_max = make_coding(n_samples=n_samples, n_atoms=n_atoms,
                                    n_dim=n_dim, random_state=random_state)

    # Compute the effective regularization
    lmbd = lmbd_max * reg

    z_hat, cost_ista, *_ = ista(D, x, lmbd, max_iter=n_iters)

    assert all(np.diff(cost_ista) <= 1e-8)


@pytest.mark.parametrize('reg', [.1, .3, .5])
def test_fista(reg):

    n_dim = 20
    n_atoms = 10
    n_samples = 1000
    random_state = 42

    n_iters = 100

    # Generate a problem
    x, D, z, lmbd_max = make_coding(n_samples=n_samples, n_atoms=n_atoms,
                                    n_dim=n_dim, random_state=random_state)

    # Compute the effective regularization
    lmbd = lmbd_max * reg

    z_hat_ista, cost_ista, *_ = ista(D, x, lmbd, max_iter=n_iters)
    z_hat_fista, cost_fista, *_ = fista(D, x, lmbd, max_iter=n_iters)

    assert np.isclose(cost_ista[-1], cost_fista[-1])

    diff = z_hat_fista - z_hat_ista
    print(diff[abs(diff) > 1e-2])


@pytest.mark.parametrize('reg', [.1, .3, .5])
def test_facnet(reg):

    n_dim = 20
    n_atoms = 10
    n_samples = 1000
    random_state = 42

    # Generate a problem
    x, D, z, lmbd_max = make_coding(n_samples=n_samples, n_atoms=n_atoms,
                                    n_dim=n_dim, random_state=random_state)

    # Compute the effective regularization
    lmbd = lmbd_max * reg

    z_hat_ista, cost_ista, *_ = ista(D, x, lmbd, max_iter=1)

    L = np.linalg.norm(D.dot(D.T), 2)
    A = np.eye(n_atoms)
    S = L * np.ones((1, n_atoms))
    z_hat = facnet_step(np.zeros_like(z), D, x, lmbd, A, S)

    assert np.allclose(z_hat_ista, z_hat)


@pytest.mark.parametrize('reg', [.1, .3, .5, 2])
@pytest.mark.parametrize('n_layers', [1, 3, 5, 10, 50, 100])
@pytest.mark.parametrize('parametrization', ['lista', 'hessian', 'coupled'])
def test_lista(reg, n_layers, parametrization):
    n_dim = 20
    n_atoms = 10
    n_samples = 1000
    random_state = 42

    # Generate a problem
    x, D, z, lmbd_max = make_coding(n_samples=n_samples, n_atoms=n_atoms,
                                    n_dim=n_dim, random_state=random_state)

    # Compute the effective regularization
    lmbd = lmbd_max * reg

    z_hat, cost_ista, _ = ista(D, x, lmbd, max_iter=n_layers)

    lista = Lista(D, n_layers, parametrization=parametrization)
    z_lista = lista(x, lmbd)

    z_lista = z_lista.data.numpy()
    assert np.isclose(cost_ista[n_layers], cost(z_lista, D, x, lmbd))


def test_grad():

    n_dim = 20
    n_atoms = 10
    n_samples = 10
    random_state = 42

    # Generate a problem
    x, D, _, _ = make_coding(n_samples=n_samples, n_atoms=n_atoms, n_dim=n_dim,
                             random_state=random_state)

    gradient_checker(cost, grad, n_samples * n_atoms, args=(D, x),
                     kwargs=dict(lmbd=0, flatten=True), n_checks=100,
                     debug=True, rtol=1e-5)
