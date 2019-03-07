import autograd.numpy as np
from autograd import grad

from sklearn.utils import check_random_state


def st(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0)


def lasso_loss(z, x, D, lbda):
    shape = x.shape
    if len(shape) == 2:
        _, n_samples = shape
    else:
        n_samples = 1
    return (0.5 * np.sum((x - np.dot(D, z)) ** 2) +
            lbda * np.sum(np.abs(z))) / n_samples


def lasso_der(x, y):
    return x - y


def logreg_loss(z, x, D, lbda):
    shape = x.shape
    if len(shape) == 2:
        _, n_samples = shape
    else:
        n_samples = 1
    return (np.sum(np.log1p(np.exp(- x * np.dot(D, z)))) +
            lbda * np.sum(np.abs(z))) / n_samples


def logreg_der(x, y):
    exp = np.exp(x * y)
    return -y / (1. + exp)


def _forward(x, weights, p, n_layers, D, level, der_function):
    if len(x.shape) == 1:
        z = np.zeros(p)
    else:
        _, n_samples = x.shape
        z = np.zeros((p, n_samples))
    for W in weights:
        z = st(z - np.dot(W, der_function(np.dot(D, z), x)), level)
    return z


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
        self.l_star = l_star
        self.logger = {}
        self.logger['loss'] = []
        self.logger['grad'] = []

    def transform(self, x):
        return _forward(x, self.weights, self.p, self.n_layers, self.D,
                        self.level, self.der_function)

    def fit(self, X, l_rate=1e-5, max_iter=100, batch_size=3, verbose=False,
            log=True):
        _, n_samples = X.shape

        def full_loss(weights, x):
            z = _forward(x, weights, self.p, self.n_layers, self.D, self.level,
                         self.der_function)

            return self.loss(z, x, self.D, self.lbda)

        gradient_function = grad(full_loss)
        weights = self.weights
        for i in range(max_iter):
            sl = np.arange(i * batch_size, (i + 1) * batch_size) % n_samples
            x = X[:, sl]
            gradients = gradient_function(weights, x)
            if i % 100 == 0 and (log or verbose):
                loss_value = full_loss(weights, X)
                gradient_value = np.sum((np.linalg.norm(g) for g in gradients))
                if log:
                    self.logger['loss'].append(loss_value)
                    self.logger['grad'].append(gradient_value)
                if verbose:
                    print('it %d, loss = %.2e, grad = %.2e' %
                          (i, loss_value - self.l_star, gradient_value))
            for weight, gradient in zip(weights, gradients):
                weight -= l_rate * gradient
        self.weights = weights
        return self
