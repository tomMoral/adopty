import autograd.numpy as np

from autograd import grad
from sklearn.utils import check_random_state

from functions import (l2_fit, l2_der, logreg_fit, logreg_der, l1_pen, l1_prox,
                       l2_pen, l2_prox, no_prox, no_pen)


def _forward(x, weights, p, n_layers, level, der_function, prox):
    '''
    defines the propagation in the network
    '''
    if len(x.shape) == 1:
        z = np.zeros(p)
    else:
        _, n_samples = x.shape
        z = np.zeros((p, n_samples))
    n_weights = len(weights) // 2
    for i in range(n_weights):
        W1 = weights[2 * i]
        W2 = weights[2 * i + 1]
        z = prox(z - np.dot(W1, der_function(np.dot(W2, z), x)), level)
    return z


def _sgd(weights, X, gradient_function, full_loss, logger, variables,
         l_rate=0.001, max_iter=100, batch_size=None, log=True, verbose=False):
    _, n_samples = X.shape
    if batch_size is None:
        batch_size = n_samples

    idx_to_skip = {'W1': 1, 'W2': 0, 'both': 2}[variables]
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
        for j, (weight, gradient) in enumerate(zip(weights, gradients)):
            if j % 2 == idx_to_skip:
                continue
            weight -= l_rate * gradient
    return weights


def make_loss(fit, pen):
    def loss(z, x, D, lbda):
        return fit(z, x, D) + lbda * pen(z)

    return loss


class LISTA(object):
    def __init__(self, D, lbda, n_layers=2, fit_loss='l2', reg='l1',
                 variables='W1'):
        '''
        Parameters
        ----------
        D : array, shape (k, p)
            Dictionnary
        lbda : float
            regularization strength
        n_layers : int
            Number of layers
        fit_loss : str
            data fit term. 'l2' or 'logreg'
        reg : str or None
            regularization function. 'l1', 'l2' or None
        variables : str
            'W1', 'W2', or 'both'. The weights to learn.
        '''
        self.D = D
        self.k, self.p = D.shape
        self.lbda = lbda
        self.n_layers = n_layers
        self.L = np.linalg.norm(D, ord=2) ** 2
        if fit_loss == 'logreg':
            self.L /= 4.
        self.level = lbda / self.L
        self.B = np.dot(D.T, D)
        self.fit_loss = fit_loss
        self.reg = reg
        fit_function, der_function = {
                                      'l2': (l2_fit, l2_der),
                                      'logreg': (logreg_fit, logreg_der)
                                      }[self.fit_loss]
        reg_function, prox = {
                              'l2': (l2_pen, l2_prox),
                              'l1': (l1_pen, l1_prox),
                              None: (no_pen, no_prox)
                              }[self.reg]
        self.fit_function = fit_function
        self.der_function = der_function
        self.reg_function = reg_function
        self.loss = make_loss(self.fit_function, self.reg_function)
        self.prox = prox
        self.weights = [self.D.T / self.L,  # W1
                        self.D.copy()] * n_layers  # W2
        self.variables = variables
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
        return _forward(x, self.weights, self.p, self.n_layers,
                        self.level, self.der_function, self.prox)

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
            z = _forward(x, weights, self.p, self.n_layers, self.level,
                         self.der_function, self.prox)
            return self.loss(z, x, self.D, self.lbda)

        gradient_function = grad(full_loss)
        if solver == 'sgd':
            weights = _sgd(self.weights, X, gradient_function, full_loss,
                           self.logger, self.variables, *args, **kwargs)
        self.weights = weights
        return self
