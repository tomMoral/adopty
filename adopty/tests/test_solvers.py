import pytest
import numpy as np


from adopty.ista import ista
from adopty.fista import fista
from adopty.datasets import make_coding
from adopty.loss_and_gradient import cost_lasso, grad

from adopty.tests.utils import gradient_checker


@pytest.mark.parametrize('reg', [.1, .3, .5])
def test_ista(reg):

    n_dim = 20
    n_atoms = 10
    n_samples = 1000
    random_state = 42

    n_iters = 100

    # Generate a problem
    x, D, z = make_coding(n_samples=n_samples, n_atoms=n_atoms, n_dim=n_dim,
                          random_state=random_state)

    z_hat, cost_ista, *_ = ista(D, x, reg, max_iter=n_iters)

    assert all(np.diff(cost_ista) <= 0)
    assert all(np.diff(cost_ista) <= 1e-8)


@pytest.mark.parametrize('reg', [.1, .3, .5])
def test_fista(reg):

    n_dim = 20
    n_atoms = 10
    n_samples = 1000
    random_state = 42

    n_iters = 100

    # Generate a problem
    x, D, z = make_coding(n_samples=n_samples, n_atoms=n_atoms, n_dim=n_dim,
                          random_state=random_state)

    z_hat_ista, cost_ista, *_ = ista(D, x, reg, max_iter=n_iters * 10)
    z_hat_fista, cost_fista, *_ = fista(D, x, reg, max_iter=n_iters)

    assert np.isclose(cost_ista[1], cost_fista[1])
    assert np.isclose(cost_ista[-1], cost_fista[-1])

    diff = z_hat_fista - z_hat_ista
    print(diff[abs(diff) > 1e-2])


def test_grad():

    n_dim = 20
    n_atoms = 10
    n_samples = 1
    random_state = 42

    # Generate a problem
    x, D, _ = make_coding(n_samples=n_samples, n_atoms=n_atoms, n_dim=n_dim,
                          random_state=random_state)

    gradient_checker(cost_lasso, grad, n_samples * n_atoms, args=(D, x),
                     kwargs=dict(lmbd=0, flatten=True), n_checks=100,
                     debug=True, rtol=1e-5)
