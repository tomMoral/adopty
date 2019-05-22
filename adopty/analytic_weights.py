import numpy as np
import matplotlib.pyplot as plt


def obj_func(W, D):
    """Cost function to minimize to obtain the analytic weights

    Parameters
    ----------
    D : ndarray, shape (n_atoms, n_dim)
        Dictionary for the considered sparse coding problem.
    """
    n_atoms, n_dim = D.shape
    Ik = np.eye(n_atoms)
    WtD = W.dot(D.T) - Ik
    Q = np.ones((n_atoms, n_atoms)) - Ik
    WtD *= np.sqrt(Q)
    return np.sum(WtD * WtD)


def proj(W, D):
    """Projection for W on the constraint set s.t. W_i^TD_i = 1
    """
    aw = np.diag(W.dot(D.T))
    return W + (1-aw)[:, None] * D


def get_alista_weights(D, max_iter=10000, step_size=1e-2, tol=1e-12):
    """Cost function to minimize to obtain the analytic weights

    Parameters
    ----------
    D : ndarray, shape (n_atoms, n_dim)
        Dictionary for the considered sparse coding problem.
    """
    n_atoms, n_dim = D.shape
    W = np.copy(D)
    pobj = [obj_func(W, D)]
    for i in range(max_iter):
        res = W.dot(D.T) - np.eye(n_atoms)
        grad = res.dot(D)
        W -= step_size * grad

        W = proj(W, D)

        pobj.append(obj_func(W, D))
        assert pobj[-1] <= pobj[-2] + 1e-8, (pobj[-2] - pobj[-1])
        if 1 - pobj[-1] / pobj[-2] < tol:
            break

    assert np.allclose(np.diag(W.dot(D.T)), 1)
    return W


def plot_weights(W, D):

    n_atoms = D.shape[0]
    Ik = np.eye(n_atoms, dtype=np.bool)

    fig, axes = plt.subplots(1, 2)

    res = W.dot(D.T) - Ik
    res0 = D.dot(D.T) - Ik

    ax = axes[0]
    ax.hist(res[Ik])
    # ax = axes[0, 1]
    ax.hist(res0[Ik], alpha=.3)

    ax = axes[1]
    nIk = Ik == 0
    ax.hist(res[nIk])
    # ax = axes[1, 1]
    ax.hist(res0[nIk], alpha=.3)

    plt.show()


if __name__ == "__main__":
    from adopty.datasets import make_coding

    n_atoms = 100
    n_dim = 64

    _, D, _ = make_coding(n_atoms=n_atoms, n_dim=n_dim)
    W = get_alista_weights(D)
    plot_weights(W, D)
