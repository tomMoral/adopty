import numpy as np
from functools import partial
from scipy.optimize import approx_fprime


def gradient_checker(func, grad, shape, args=(), kwargs={}, n_checks=10,
                     rtol=1e-5, debug=False, random_seed=None):
    """Check that the gradient correctly approximate the finite difference
    """

    rng = np.random.RandomState(random_seed)

    msg = ("Computed gradient did not match gradient computed with finite "
           "difference. Relative error is {}")

    func = partial(func, **kwargs)

    def test_grad(z0):
        grad_approx = approx_fprime(z0, func, 1e-8, *args)
        grad_compute = grad(z0, *args, **kwargs)
        error = np.sqrt(np.sum((grad_approx - grad_compute) ** 2))
        error /= np.sqrt(np.sum(grad_compute * grad_approx))
        try:
            assert error < rtol, msg.format(error)
        except AssertionError:
            if debug:
                import matplotlib.pyplot as plt
                plt.plot(grad_approx)
                plt.plot(grad_compute)
                plt.show()
            raise

    z0 = np.zeros(shape)
    test_grad(z0)

    for _ in range(n_checks):
        z0 = rng.randn(shape)
        test_grad(z0)
