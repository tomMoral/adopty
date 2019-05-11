import autograd.numpy as np

from sklearn.utils import check_random_state


def generate(p, k, n_test, n_train, binary=False, K=None, D=None, rng=None,
             sigma=0.01, density='normal'):
    rng = check_random_state(rng)
    if K is None:
        K = np.eye(p)
    if D is None:
        D = np.dot(rng.randn(k, p), K)
    if density == 'laplace':
        z = rng.laplace(size=(p, n_test + n_train))
    else:
        z = rng.randn(p, n_test + n_train)
    X = (np.dot(D, z) +
         sigma * rng.randn(k, n_test+n_train))
    if binary:
        X = 2. * (X > 0) - 1.
    return D, X[:, :n_train], X[:, n_train:]


def get_lambda_max(D, X, fit='l2', maximum=True):
    lbda = np.max(np.abs(np.dot(D.T, X)), axis=0)
    if maximum:
        lbda = np.max(lbda)
    if fit == 'logreg':
        lbda /= 2.
    return lbda
