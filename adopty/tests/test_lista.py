import pytest
import numpy as np


from adopty.ista import ista
from adopty.datasets import make_coding
from adopty.lista import Lista, PARAMETRIZATIONS


@pytest.mark.parametrize('reg', [.1, .3, .5, 2])
@pytest.mark.parametrize('learn_th', [True, False])
@pytest.mark.parametrize('n_layers', [1, 10, 50, 100])
@pytest.mark.parametrize('parametrization', PARAMETRIZATIONS)
def test_lista_init(reg, n_layers, parametrization, learn_th):

    if parametrization == "alista":
        pytest.skip(msg="ALISTA is not initialized to match ISTA.")
    if "step" in parametrization and not learn_th:
        pytest.skip(msg="For parametrization 'step' and 'coupled_step', "
                        "learn_th need to be set to True.")
    n_dim = 20
    n_atoms = 10
    n_samples = 1000
    random_state = 42

    # Generate a problem
    x, D, z = make_coding(n_samples=n_samples, n_atoms=n_atoms, n_dim=n_dim,
                          random_state=random_state)

    z_hat, cost_ista, _ = ista(D, x, reg, max_iter=n_layers)

    lista = Lista(D, n_layers, parametrization=parametrization)
    cost_lista = lista.score(x, reg)
    assert np.isclose(cost_ista[n_layers], cost_lista)


@pytest.mark.parametrize('learn_th', [True, False])
@pytest.mark.parametrize('parametrization', PARAMETRIZATIONS)
def test_save(parametrization, learn_th):
    n_dim = 20
    n_atoms = 10
    n_samples = 1000
    random_state = 42

    reg = .1
    n_layers = 4

    if "step" in parametrization and not learn_th:
        pytest.skip(msg="For parametrization 'step' and 'coupled_step', "
                        "learn_th need to be set to True.")

    # Generate a problem
    x, D, z = make_coding(n_samples=n_samples, n_atoms=n_atoms, n_dim=n_dim,
                          random_state=random_state)

    lista = Lista(D, n_layers, parametrization=parametrization,
                  learn_th=learn_th, max_iter=15)
    lista.fit(x, reg)
    parameters = lista.export_parameters()

    lista_ = Lista(D, n_layers, parametrization=parametrization,
                   learn_th=learn_th)
    lista_.init_network_parameters(parameters_init=parameters)
    parameters_ = lista_.export_parameters()
    assert np.all([np.allclose(pl[k], pl_[k])
                   for pl, pl_ in zip(parameters, parameters_) for k in pl])

    cost_lista = lista.score(x, reg)
    cost_lista_ = lista_.score(x, reg)
    assert np.allclose(cost_lista, cost_lista_)

    z_lista = lista.transform(x, reg)
    z_lista_ = lista_.transform(x, reg)
    atol = abs(z_lista).max() * 1e-6
    assert np.allclose(z_lista, z_lista_, atol=atol)
