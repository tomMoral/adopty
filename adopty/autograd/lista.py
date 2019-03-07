import autograd.numpy as np

from autograd import grad
from sklearn.utils import check_random_state

from functions import lasso_loss, logreg_loss, lasso_der, logreg_der


def st(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0)


def _forward(x, weights, p, n_layers, D, level, der_function):
    if len(x.shape) == 1:
        z = np.zeros(p)
    else:
        _, n_samples = x.shape
        z = np.zeros((p, n_samples))
    for W in weights:
        z = st(z - np.dot(W, der_function(np.dot(D, z), x)), level)
    return z


def sgd(weights, X, gradient_function, full_loss, logger, l_rate=0.001,
        max_iter=100, batch_size=None, log=True, verbose=False):
    _, n_samples = X.shape
    if batch_size is None:
        batch_size = n_samples
    for i in range(max_iter):
        sl = np.arange(i * batch_size, (i + 1) * batch_size) % n_samples
        x = X[:, sl]
        gradients = gradient_function(weights, x)
        if i % 100 == 0 and (log or verbose):
            loss_value = full_loss(weights, X)
            gradient_value = np.sum((np.linalg.norm(g) for g in gradients))
            if log:
                logger['loss'].append(loss_value)
                logger['grad'].append(gradient_value)
            if verbose:
                print('it %d, loss = %.3e, grad = %.2e' %
                      (i, loss_value, gradient_value))
        for weight, gradient in zip(weights, gradients):
            weight -= l_rate * gradient
    return weights


class LISTA(object):
    def __init__(self, D, lbda, n_layers=2, model='lasso', rng=None,
                 l_star=0.):
        self.D = D
        self.n, self.p = D.shape
        self.lbda = lbda
        self.n_layers = n_layers
        self.rng = check_random_state(rng)
        self.L = np.linalg.norm(D, ord=2) ** 2
        if model == 'logreg':
            self.L /= 4.
        self.level = lbda / self.L
        self.B = np.dot(D.T, D)
        self.model = model
        loss, der_function = {'lasso': (lasso_loss, lasso_der),
                              'logreg': (logreg_loss, logreg_der)}[self.model]
        self.loss = loss
        self.der_function = der_function
        self.weights = [self.D.T / self.L, ] * n_layers  # Init weights
        self.logger = {}
        self.logger['loss'] = []
        self.logger['grad'] = []

    def transform(self, x):
        return _forward(x, self.weights, self.p, self.n_layers, self.D,
                        self.level, self.der_function)

    def fit(self, X, solver='sgd', *args, **kwargs):

        def full_loss(weights, x):
            z = _forward(x, weights, self.p, self.n_layers, self.D, self.level,
                         self.der_function)
            return self.loss(z, x, self.D, self.lbda)

        gradient_function = grad(full_loss)
        if solver == 'sgd':
            weights = sgd(self.weights, X, gradient_function, full_loss,
                          self.logger, *args, **kwargs)
        self.weights = weights
        return self
