import numpy as np
from joblib import Memory
import matplotlib.pyplot as plt

from adopty.lista import Lista
from adopty.datasets import make_coding
from adopty.loss_and_gradient import cost_lasso

from itertools import combinations


mem = Memory(location='.', verbose=0)

seed = np.random.randint(0, 1000)
print(seed)
rng = np.random.RandomState(seed)
n_dim = 2
n_atoms = 3
n_s = 10000

n_layers = 1
x, D, z = make_coding(n_s, n_atoms, n_dim, rng)

x_d = np.copy(x)


def loss_lasso(z, x, reg):
    res = np.dot(z, D) - x
    return 0.5 * np.sum(res ** 2, axis=1) + reg * np.sum(np.abs(z), axis=1)


x_m, y_m = np.min(x, axis=0)
x_M, y_M = np.max(x, axis=0)

x_min = min(-2, x_m)
x_max = max(2, x_M)
y_min = min(-2, y_m)
y_max = max(2, y_M)
n_samples = 500


x_list = np.linspace(x_min, x_max, n_samples)
y_list = np.linspace(y_min, y_max, n_samples)

grid = np.meshgrid(x_list, y_list)
x_plot = np.zeros((n_samples ** 2, 2))
for i in [0, 1]:
    x_plot[:, i] = grid[i].ravel()


n_reg = 100
for enum, reg in enumerate(np.linspace(0, 1, n_reg)[1:]):
    print(enum / n_reg)
    x_r = x_d * reg
    z_lasso = Lista(D, 1000).transform(x_plot, reg)
    loss_fit = loss_lasso(z_lasso, x_plot, reg)


    ista = Lista(D, n_layers)
    z_ista = ista.transform(x_plot, reg)
    z_ista_train = ista.transform(x, reg)
    loss_ista = loss_lasso(z_ista, x_plot, reg)
    avg_loss = np.mean(loss_lasso(z_ista_train, x, reg))
    print('Ista training loss: {}'.format(avg_loss))


    # @mem.cache
    def get_trained_lista(D, x, reg, n_layers, max_iter):

        lista = Lista(D, n_layers, max_iter=max_iter)
        lista.fit(x, reg)
        return lista


    lista = get_trained_lista(D, x, reg, n_layers, max_iter=1000)
    z_lista = lista.transform(x_plot, reg)
    z_lista_train = lista.transform(x, reg)
    loss_lista = loss_lasso(z_lista, x_plot, reg)
    avg_loss = np.mean(loss_lasso(z_lista_train, x, reg))
    print('Lista training loss: {}'.format(avg_loss))

    for layer in [n_layers, ]:
        z_lista = lista.transform(x_plot, reg, output_layer=layer)
        loss_lista = loss_lasso(z_lista, x_plot, reg)
        z_ista = ista.transform(x_plot, reg, output_layer=layer)
        loss_ista = loss_lasso(z_ista, x_plot, reg)
        Z_plot = ((loss_ista - loss_lista) / loss_fit).reshape(n_samples,
                                                               n_samples)
        Z_plot = np.sign(Z_plot) * np.abs(Z_plot) ** .2
        M = np.max(np.abs(Z_plot))
        plt.figure()
        plt.set_cmap('PiYG')
        Cs = plt.contourf(grid[0], grid[1], Z_plot, 50, vmin=-M, vmax=M)
        D_ = np.sign(D.dot(D[0]))[:, None] * D
        for dk in D_:
            plt.arrow(0, 0, dk[0], dk[1], color='cornflowerblue')

        U, S, V = np.linalg.svd(D, full_matrices=False)
        M = V
        for i in range(1):
            plt.arrow(0, 0, M[i, 0], M[i, 1], color='darkslategrey')
        plt.colorbar(Cs)
        # plt.scatter(x[:, 0], x[:, 1], color='cornflowerblue', s=1)
        plt.scatter(x_d[:, 0], x_d[:, 1], color='cornflowerblue', s=.5)
        plt.scatter(x_r[:, 0], x_r[:, 1], color='gainsboro', s=.5)
        # plt.title('Loss ista - loss lista (green= lista is better), '
        #           'layer {} / {}'.format(layer, n_layers))
        plt.title('lambda = %.2e' % reg)
        plt.savefig('examples/figures/%0*d.png' % (3, enum))
plt.show()
