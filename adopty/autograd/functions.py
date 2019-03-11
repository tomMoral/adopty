import autograd.numpy as np


def l2_fit(z, x, D):
    n_samples = get_n_samples(z)
    return 0.5 * np.sum((x - np.dot(D, z)) ** 2) / n_samples


def l2_der(x, y):
    return x - y


def logreg_fit(z, x, D):
    n_samples = get_n_samples(z)
    return np.sum(np.log1p(np.exp(- x * np.dot(D, z)))) / n_samples


def logreg_der(x, y):
    exp = np.exp(x * y)
    return -y / (1. + exp)


def l1_pen(z):
    n_samples = get_n_samples(z)
    return np.sum(np.abs(z)) / n_samples


def l1_prox(x, level):
    return np.sign(x) * np.maximum(np.abs(x) - level, 0.)


def l2_pen(z):
    n_samples = get_n_samples(z)
    return 0.5 * np.sum(z ** 2) / n_samples


def l2_prox(x, level):
    return x / (1. + level)


def no_pen(z):
    return 0.


def no_prox(x, level):
    return x


def get_n_samples(z):
    shape = z.shape
    if len(shape) == 2:
        _, n_samples = shape
    else:
        n_samples = 1
    return n_samples
