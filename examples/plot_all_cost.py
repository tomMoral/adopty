import numpy as np
import matplotlib.pyplot as plt

from adopty.lista import Lista
from adopty.datasets import make_coding
from adopty.loss_and_gradient import cost_lasso


def get_c_star(x, D, z, reg, n_iter=10000):
    ista = Lista(D, n_iter, parametrization="coupled")
    z_star = ista.transform(x, reg, z0=z).numpy()
    return cost_lasso(z_star, D, x, reg)


x, D, z = make_coding(n_samples=1000)
reg = .5
n_layers = 3

x_test = np.random.randn(*x.shape)


format_cost = "{}: {} cost = {:.3e}"
c_star = get_c_star(x, D, z, reg)


saved_model = {}
for parametrization in ['hessian', 'coupled']:
    lista = Lista(D, n_layers, parametrization=parametrization, max_iter=3000)
    z_hat_test = lista.transform(x_test, reg).numpy()
    c_star_test = get_c_star(x_test, D, z_hat_test, reg)
    cost_test = cost_lasso(z_hat_test, D, x_test, reg) - c_star_test
    print(format_cost.format("Un-trained[{}]".format(parametrization), "test",
                             cost_test))
    lista.fit(x, reg)
    z_hat_test = lista.transform(x_test, reg).numpy()
    c_star_test = get_c_star(x_test, D, z_hat_test, reg)
    cost_test = cost_lasso(z_hat_test, D, x_test, reg) - c_star_test
    print(format_cost.format("Trained[{}]".format(parametrization), "test",
                             cost_test))
    saved_model[parametrization] = lista

    z_hat = lista.transform(x, reg).numpy()
    c_star = min(c_star, get_c_star(x, D, z_hat, reg))
    plt.loglog(lista.training_loss_ - c_star, label=parametrization)

plt.legend()

plt.show()
