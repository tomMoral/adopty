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


def _sgd(weights, X, gradient_function, full_loss, logger, l_rate=0.001,
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
    def __init__(self, D, lbda, n_layers=2, model='lasso'):
        '''
        Parameters
        ----------
        D : array, shape (k, p)
            Dictionnary
        lbda : float
            regularization
        n_layers : int
            Number of layers
        model : str
            model to use. Either 'lasso' or 'logreg'
        '''
        self.D = D
        self.k, self.p = D.shape
        self.lbda = lbda
        self.n_layers = n_layers
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
        '''
        Parameters
        ----------
        x : array, shape (k, n_samples) or (k,)
            Input array

        Returns
        -------
        z : array, shape (p, n_samples)
            The output of the network, close from the minimum of the Lasso
        '''
        return _forward(x, self.weights, self.p, self.n_layers, self.D,
                        self.level, self.der_function)

    def fit(self, X, solver='sgd', *args, **kwargs):
        '''
        Parameters
        ----------
        X : array, shape (k, n_samples) or (k,)
            train array
        solver : str
            solver to use
        *args, **kwargs : other arguments to pass to the solver

        Returns
        -------
        self
        '''
        def full_loss(weights, x):
            z = _forward(x, weights, self.p, self.n_layers, self.D, self.level,
                         self.der_function)
            return self.loss(z, x, self.D, self.lbda)

        gradient_function = grad(full_loss)
        if solver == 'sgd':
            weights = _sgd(self.weights, X, gradient_function, full_loss,
                           self.logger, *args, **kwargs)
        self.weights = weights
        return self
