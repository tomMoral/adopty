import pytest
import numpy as np
from adopty.sinkhorn import Sinkhorn
from adopty.datasets.optimal_transport import make_ot


@pytest.mark.parametrize(
    'eps', [.1, 1, 10]
)
@pytest.mark.parametrize(
    'n_layers', [1, 5, 10, 100, 1000]
)
def test_log_domain(eps, n_layers):
    """Test that the log domain computation is equivalent to classical sinkhorn
    """
    p = 2
    n, m = 10, 15

    alpha, beta, C, *_ = make_ot(n, m, p, random_state=0)

    snet1 = Sinkhorn(n_layers, log_domain=True)
    f1, g1 = snet1.transform(alpha, beta, C, eps)
    snet2 = Sinkhorn(n_layers, log_domain=False)
    f2, g2 = snet2.transform(alpha, beta, C, eps)
    assert np.allclose(f1, f2)
    assert np.allclose(g1, g2)

    # Check that the scores are well computed
    assert np.isclose(
        snet1.score(alpha, beta, C, eps),
        snet2.score(alpha, beta, C, eps)
    )


def sinkhorn(a, b, K, n_iter):
    n, m = K.shape
    v = np.ones(m)
    for i in range(n_iter):
        u = a / np.dot(K, v)
        v = b / np.dot(u, K)
    return u, v


def test_sinkhorn_np():
    p = 2
    n, m = 10, 15
    eps = .1
    n_layers = 500

    alpha, beta, C, *_ = make_ot(n, m, p, random_state=0)

    snet = Sinkhorn(n_layers=n_layers, log_domain=True)
    f, g = snet.transform(alpha, beta, C, eps)

    u, v = sinkhorn(alpha, beta, np.exp(-C / eps), n_layers)
    assert np.allclose(f, eps * np.log(u))
    assert np.allclose(g, eps * np.log(v))


def test_gradient_beta():
    p = 2
    n, m = 10, 15
    eps = .1
    n_layers = 500

    alpha, beta, C, *_ = make_ot(n, m, p, random_state=0)

    snet = Sinkhorn(n_layers=n_layers)

    f1, g1 = snet.transform(alpha, beta, C, eps)
    f2, g2 = snet.transform(alpha, beta, C, eps, output_layer=n_layers-2)
    err_norm = np.sqrt(np.linalg.norm(f1 - f2) ** 2 +
                       np.linalg.norm(g1 - g2) ** 2)
    assert err_norm < 1e-6

    # Get the gradient with analytic formula and autodiff
    G1 = snet.gradient_beta_analytic(alpha, beta, C, eps)
    G2 = snet.gradient_beta(alpha, beta, C, eps)

    assert np.allclose(G1, G2)
