import autograd.numpy as np


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
