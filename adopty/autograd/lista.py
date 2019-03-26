import warnings
from copy import deepcopy

import autograd.numpy as np

from autograd import grad
from sklearn.utils import check_random_state

from functions import (l2_fit, l2_der, logreg_fit, logreg_der, l1_pen, l1_prox,
                       l2_pen, l2_prox, no_prox, no_pen)
from generation import get_lambda_max


def _forward(x, weights, p, n_layers, levels, der_function, prox, lbda,
             scaling):
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
        level = levels[i] * lbda
        z = prox(z - np.dot(W1, der_function(np.dot(W2, z), x)),
                 level[:, None])
        z = scaling[i][:, None] * z
    return z


def _forward_sag(x, weights, p, n_layers, levels, der_function, prox, lbda):
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
        level = levels[i] * lbda
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


def _forward_cd(x, weights, p, n_layers, levels, der_function, prox, lbda):
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
        level = levels[i] * lbda
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
            if backprop:
                # Inefficient mask trick
                mask = np.zeros((p, n_samples))
                mask[j] = np.ones(n_samples)
                z += mask * (prox(z[j] - coord_update, level[j]) - z[j])
            else:
                z[j] = prox(z[j] - coord_update, level[j])
            if backprop:
                z_new = z[j]._value
            else:
                z_new = z[j]
            W2j = W2[:, j]
            residual += der_function(W2j[:, None] * z_new[None, :], x)
            residual -= der_function(W2j[:, None] * z_old[None, :], x)
    return z


def _sgd(weights, levels, scaling, X, gradient_function, full_loss, logger,
         variables, learn_levels, learn_scale, l_rate=0.001, l_rate_min=1e-6,
         max_iter=100, thres=0, batch_size=None, log=True, verbose=False,
         backtrack=False):
    _, n_samples = X.shape
    if batch_size is None:
        batch_size = n_samples
    old_loss = np.inf
    loss_value = full_loss((weights, levels, scaling), X)
    idx_to_skip = {'W1': 1, 'W2': 0, 'both': 2, 'spectrum': 1}[variables]
    for i in range(max_iter):
        sl = np.arange(i * batch_size, (i + 1) * batch_size) % n_samples
        x = X[:, sl]
        gradients = gradient_function((weights, levels, scaling), x)
        if verbose:
            if i % verbose == 0:
                if not backtrack:
                    loss_value = full_loss((weights, levels), X)
                gradient_W = np.sum((np.linalg.norm(g) for g in gradients[0]))
                gradient_l = np.sum((np.linalg.norm(g) for g in gradients[1]))
                gradient_s = np.sum((np.linalg.norm(g) for g in gradients[2]))
                if loss_value > old_loss:
                    warnings.warn('loss increasing')
                old_loss = loss_value
                if log:
                    logger['loss'].append(loss_value)
                    logger['grad'].append(gradient_W)
                if verbose:
                    print('it %d, loss = %.3e, grad_W = %.2e, grad_l = %.2e, '
                          'grad_s = %.2e, l_rate = %.2e' %
                          (i, loss_value, gradient_W, gradient_l, gradient_s,
                           l_rate))
        # Update weights
        old_loss = loss_value
        if backtrack:
            (weights, levels, scaling), loss_value, l_rate =\
                backtracking(weights, levels, scaling, full_loss, gradients,
                             l_rate, x, idx_to_skip, learn_levels,
                             learn_scale, variables, loss_init=loss_value,
                             l_rate_min=l_rate_min)
        else:
            weights, levels, scaling = change_params(weights, levels, scaling,
                                                     l_rate, gradients,
                                                     idx_to_skip, learn_levels,
                                                     learn_scale, variables)
        # if learn_scale:
        #     scaling = levels / weights ** 2
        if np.abs(old_loss - loss_value) / np.abs(loss_value) < thres:
            break
    return weights, levels, scaling


def change_params(weights, levels, scaling, l_rate, gradients, idx_to_skip,
                  learn_levels, learn_scale, variables):
    if variables != 'spectrum':
        for j, (weight, gradient) in enumerate(zip(weights, gradients[0])):
            if j % 2 == idx_to_skip:
                continue
            weight -= l_rate * np.array(gradient)
    else:
        weights -= l_rate * np.array(gradients[0])
    if learn_levels:
        levels -= l_rate * np.array(gradients[1])
    if learn_scale:
        scaling -= l_rate * np.array(gradients[2])
    return weights, levels, scaling


def backtracking(weights, levels, scaling, loss, gradients, l_rate, x,
                 idx_to_skip, learn_levels, learn_scale, variables,
                 loss_init=None, cst_mul=2.,  n_tries=30,  l_rate_min=1e-6):
    if loss_init is None:
        loss_init = loss(optim_vars, x)
    l_rate *= cst_mul
    for _ in range(n_tries):
        new_vars = change_params(deepcopy(weights), np.copy(levels),
                                 np.copy(scaling), l_rate,
                                 gradients, idx_to_skip, learn_levels,
                                 learn_scale, variables)
        new_loss = loss(new_vars, x)
        if new_loss < loss_init or l_rate < l_rate_min:
            break
        l_rate /= cst_mul
    return new_vars, new_loss, l_rate


def make_loss(fit, pen):
    def loss(z, x, D, lbda):
        return fit(z, x, D) + lbda * pen(z)

    return loss


def spectrum_to_weights(spectra, U, V, D):
    weights = []
    for spectrum in spectra:
        weights.append(np.dot(U * spectrum, V))
        weights.append(D.copy())
    return weights


class LISTA(object):
    def __init__(self, D, lbda, n_layers=2, fit_loss='l2', reg='l1',
                 variables='W1', learn_levels=False, architecture='pgd',
                 one_lbda_each=False, learn_scale=False, lbda_weights=None):
        '''
        Parameters
        ----------
        D : array, shape (k, p)
            Dictionnary
        lbda : float or array
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
        self.compute_lbda_path = type(lbda) in [np.ndarray, list, tuple]
        if self.compute_lbda_path:
            if lbda_weights is None:
                lbda_weights = np.ones(len(lbda))
            self.lbda_weights = lbda_weights
        self.n_layers = n_layers
        self.architecture = architecture
        self.learn_scale = learn_scale
        self.scaling = np.ones((n_layers, self.p))
        if architecture == 'pgd':
            self.L = np.linalg.norm(D, ord=2) ** 2
            if fit_loss == 'logreg':
                self.L /= 4.
            self.forward = _forward
        elif architecture == 'sag':
            self.L = 3 * np.max(np.sum(D ** 2, axis=1))
            self.forward = _forward_sag
        elif architecture == 'cd':
            self.L = np.sum(D ** 2, axis=0)
            self.forward = _forward_cd
        self.levels = np.ones((n_layers, self.p)) / self.L
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
            if self.architecture == 'cd':
                self.weights.append(self.D.T.copy() / self.L[:, None])
            else:
                self.weights.append(self.D.T.copy() / self.L)
            self.weights.append(self.D.copy())
        self.one_lbda_each = one_lbda_each
        self.variables = variables
        if self.variables == 'spectrum':
            self.D_svd = np.linalg.svd(D.T, full_matrices=False)
            self.U = self.D_svd[0][:, ::-1]
            self.V = self.D_svd[2][::-1]
            self.spectra = np.array([(self.D_svd[1][::-1] / self.L)
                                     for _ in range(n_layers)])
        self.learn_levels = learn_levels
        self.logger = {}
        self.logger['loss'] = []
        self.logger['grad'] = []

    def transform(self, x, lbda=None):
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
        if lbda is None:
            lbda = self.lbda
        if self.one_lbda_each:
            lbda_max = get_lambda_max(self.D, x, self.fit_loss, False)
            lbda = lbda_max * lbda
        return self.forward(x, self.weights, self.p, self.n_layers,
                            self.levels, self.der_function, self.prox,
                            lbda, self.scaling)

    def compute_loss(self, x):
        if self.compute_lbda_path:
            return [self.loss(self.transform(x, lbda), x, self.D, lbda) for
                    lbda in self.lbda]
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
            if self.variables == 'spectrum':
                spectra, levels, scaling = optim_vars
                weights = spectrum_to_weights(spectra, self.U, self.V, self.D)
            else:
                weights, levels, scaling = optim_vars
            if self.compute_lbda_path:
                loss = 0.
                for lbda, w in zip(self.lbda, self.lbda_weights):
                    z = self.forward(x, weights, self.p, self.n_layers,
                                     levels, self.der_function, self.prox,
                                     lbda)
                    loss += w * self.loss(z, x, self.D, lbda)
                return loss
            z = self.forward(x, weights, self.p, self.n_layers, levels,
                             self.der_function, self.prox, self.lbda, scaling)
            return self.loss(z, x, self.D, self.lbda)

        gradient_function = grad(full_loss)
        if self.variables == 'spectrum':
            variables = self.spectra
        else:
            variables = self.weights
        if solver == 'sgd':
            outputs, levels, scaling =\
                _sgd(variables, self.levels, self.scaling, X,
                     gradient_function, full_loss, self.logger, self.variables,
                     self.learn_levels, self.learn_scale, *args, **kwargs)
        if self.variables == 'spectrum':
            self.spectra = outputs
            self.weights = spectrum_to_weights(self.spectra, self.U, self.V,
                                               self.D)
        else:
            self.weights = outputs
        self.levels = levels
        self.scaling = scaling
        return self
