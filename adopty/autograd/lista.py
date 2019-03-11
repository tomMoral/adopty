import warnings

import autograd.numpy as np

from autograd import grad
from sklearn.utils import check_random_state

from functions import (l2_fit, l2_der, logreg_fit, logreg_der, l1_pen, l1_prox,
                       l2_pen, l2_prox, no_prox, no_pen)


def _forward(x, weights, p, n_layers, levels, der_function, prox):
    '''
    defines the propagation in the network
    '''
    if len(x.shape) == 1:
        z = np.zeros(p)
    else:
        _, n_samples = x.shape
        z = np.zeros((p, n_samples))
    n_layers = len(weights) // 2
    for i in range(n_layers):
        W1 = weights[2 * i]
        W2 = weights[2 * i + 1]
        level = levels[i]
        z = prox(z - np.dot(W1, der_function(np.dot(W2, z), x)), level)
    return z


def _forward_sag(x, weights, p, n_layers, levels, der_function, prox):
    '''
    defines the propagation in the network using sag
    '''
    if len(x.shape) == 1:
        z = np.zeros(p)
        K = len(x)
    else:
        K, n_samples = x.shape
        z = np.zeros((p, n_samples))
    n_layers = len(weights) // 2
    avg_gradient = np.zeros((p, n_samples))
    memory_gradient = np.zeros((K, p, n_samples))
    backprop = not type(weights[0]) is np.ndarray
    for i in range(n_layers):
        W1 = weights[2 * i]
        W2 = weights[2 * i + 1]
        level = levels[i]
        for k in range(K):
            old_gradient = memory_gradient[k]
            one_dim_der = der_function(np.dot(W2[k, :], z), x[k])
            new_gradient = W1[:, k][:, None] * one_dim_der[None, :]
            avg_gradient += (new_gradient - old_gradient) / K
            if backprop:
                memory_gradient[k] = new_gradient._value
            else:
                memory_gradient[k] = new_gradient
            z = prox(z - new_gradient + old_gradient - avg_gradient, level)
    return z


def _forward_cd(x, weights, p, n_layers, levels, der_function, prox):
    '''
    defines the propagation in the network using coordinate descent
    '''
    if len(x.shape) == 1:
        z = np.zeros(p)
        K = len(x)
    else:
        K, n_samples = x.shape
        z = np.zeros((p, n_samples))
    n_layers = len(weights) // 2
    residual = np.zeros((K, n_samples))
    backprop = not type(weights[0]) is np.ndarray
    for i in range(n_layers):
        W1 = weights[2 * i]
        W2 = weights[2 * i + 1]
        level = levels[i]
        residual = der_function(np.dot(W2, z), x)
        for j in range(p):
            if backprop:
                if type(z[j]) is np.ndarray:
                    z_old = np.copy(z[j])
                else:
                    z_old = np.copy(z[j]._value)
            else:
                z_old = np.copy(z[j])
            coord_update = np.dot(W1[j], residual)
            # Inefficient mask trick
            mask = np.zeros((p, n_samples))
            mask[j] = np.ones(n_samples)
            z += mask * (prox(z[j] - coord_update, level) - z[j])
            if backprop:
                z_new = z[j]._value
            else:
                z_new = z[j]
            W2j = W2[:, j]
            residual += der_function(W2j[:, None] * z_new[None, :], x)
            residual -= der_function(W2j[:, None] * z_old[None, :], x)
    return z


def _sgd(weights, levels, X, gradient_function, full_loss, logger, variables,
         learn_levels, l_rate=0.001, max_iter=100, batch_size=None, log=True,
         verbose=False, backtrack=True):
    _, n_samples = X.shape
    if batch_size is None:
        batch_size = n_samples
    old_loss = np.inf
    loss_value = full_loss((weights, levels), X)
    idx_to_skip = {'W1': 1, 'W2': 0, 'both': 2}[variables]
    for i in range(max_iter):
        sl = np.arange(i * batch_size, (i + 1) * batch_size) % n_samples
        x = X[:, sl]
        gradients = gradient_function((weights, levels), x)
        if i % 100 == 0 and (log or verbose):
            gradient_W = np.sum((np.linalg.norm(g) for g in gradients[0]))
            gradient_l = np.sum((np.linalg.norm(g) for g in gradients[1]))
            if loss_value > old_loss:
                warnings.warn('loss increasing')
            old_loss = loss_value
            if log:
                logger['loss'].append(loss_value)
                logger['grad'].append(gradient_W)
            if verbose:
                print('it %d, loss = %.3e, grad_W = %.2e, grad_l = %.2e' %
                      (i, loss_value, gradient_W, gradient_l))
        # Update weights
        if backtrack:
            (weights, levels), loss_value, l_rate =\
                backtracking(optim_vars, loss, gradient, l_rate, x,
                             loss_init=loss_value)
        else:
            for j, (weight, gradient) in enumerate(zip(weights, gradients[0])):
                if j % 2 == idx_to_skip:
                    continue
                weight -= l_rate * np.array(gradient)
            # Update levels
            if learn_levels:
                levels -= l_rate * np.array(gradients[1])
    return weights, levels


def backtracking(optim_vars, loss, gradient, l_rate, x, loss_init=None,
                 cst_mul=2., n_tries=20):
    if loss_init is None:
        loss_init = loss(optim_vars, x)
    l_rate *= cst_mul
    for _ in range(n_tries):
        # new_vars = optim_vars - l_rate * gradient
        new_loss = loss(new_vars, x)
        if new_loss < loss_init:
            break
        l_rate /= cst_mul
    return new_vars, new_loss, l_rate


def make_loss(fit, pen):
    def loss(z, x, D, lbda):
        return fit(z, x, D) + lbda * pen(z)

    return loss


class LISTA(object):
    def __init__(self, D, lbda, n_layers=2, fit_loss='l2', reg='l1',
                 variables='W1', learn_levels=False, architecture='pgd'):
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
        learn_levels : bool
            wether the prox levels should be learned
        architecture : str
            defines which optim algorithm to mimic. Either 'pgd' or 'sag'
        '''
        self.D = D
        self.k, self.p = D.shape
        self.lbda = lbda
        self.n_layers = n_layers
        self.architecture = architecture
        if architecture == 'pgd':
            self.L = np.linalg.norm(D, ord=2) ** 2
            if fit_loss == 'logreg':
                self.L /= 4.
            self.forward = _forward
        elif architecture == 'sag':
            self.L = np.max(np.sum(D ** 2, axis=1))
            self.forward = _forward_sag
        elif architecture == 'cd':
            self.L = np.max(np.sum(D ** 2, axis=0))
            self.forward = _forward_cd
        self.levels = [lbda / self.L, ] * n_layers
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
        self.weights = []
        for _ in range(n_layers):
            self.weights.append(self.D.T.copy() / self.L)
            self.weights.append(self.D.copy())
        self.variables = variables
        self.learn_levels = learn_levels
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
        return self.forward(x, self.weights, self.p, self.n_layers,
                            self.levels, self.der_function, self.prox)

    def compute_loss(self, x):
        return self.loss(self.transform(x), x, self.D, self.lbda)

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
        def full_loss(optim_vars, x):
            weights, levels = optim_vars
            z = self.forward(x, weights, self.p, self.n_layers, levels,
                             self.der_function, self.prox)
            return self.loss(z, x, self.D, self.lbda)

        gradient_function = grad(full_loss)
        if solver == 'sgd':
            weights, levels = _sgd(self.weights, self.levels, X,
                                   gradient_function, full_loss, self.logger,
                                   self.variables, self.learn_levels, *args,
                                   **kwargs)
        self.weights = weights
        self.levels = levels
        return self
