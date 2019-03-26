import numpy as np
import matplotlib.pyplot as plt

from adopty.lista import Lista
from adopty.datasets import make_coding
from adopty.loss_and_gradient import cost_lasso

from itertools import combinations


seed = np.random.randint(0, 10)
rng = np.random.RandomState(seed)
n_dim = 2
n_atoms = 3
n_s = 1000
reg = .05

n_layers = 2
x, D, z = make_coding(n_s, n_atoms, n_dim, rng)

# D = np.random.randn(n_atoms, n_dim)
x = rng.randn(n_s, 2)
x /= np.linalg.norm(x, axis=1, keepdims=True)

# angles = np.angle(x[:, 0] + 1j * x[:, 1])
# eps = 0.3
# target_angle = 0.75
# mask = ((angles < (np.pi * (target_angle + eps))) *
#         (angles > (np.pi * (target_angle - eps))))
# x = x[mask, :]


def loss_lasso(z, x, reg):
    res = np.dot(z, D) - x
    return 0.5 * np.sum(res ** 2, axis=1) + reg * np.sum(np.abs(z), axis=1)


x_m, y_m = np.min(x, axis=0)
x_M, y_M = np.max(x, axis=0)

x_min = min(-2, x_m)
x_max = max(2, x_M)
y_min = min(-2, y_m)
y_max = max(2, y_M)
n_samples = 100


x_list = np.linspace(x_min, x_max, n_samples)
y_list = np.linspace(y_min, y_max, n_samples)

grid = np.meshgrid(x_list, y_list)
x_plot = np.zeros((n_samples ** 2, 2))
for i in [0, 1]:
    x_plot[:, i] = grid[i].ravel()

z_lasso = Lista(D, 1000).transform(x_plot, reg)
loss_fit = loss_lasso(z_lasso, x_plot, reg)


ista = Lista(D, n_layers)
z_ista = ista.transform(x_plot, reg)
z_ista_train = ista.transform(x, reg)
loss_ista = loss_lasso(z_ista, x_plot, reg)
avg_loss = np.mean(loss_lasso(z_ista_train, x, reg))
print('Ista training loss: {}'.format(avg_loss))

lista = Lista(D, n_layers)
lista.fit(x, reg)
z_lista = lista.transform(x_plot, reg)
z_lista_train = lista.transform(x, reg)
loss_lista = loss_lasso(z_lista, x_plot, reg)
avg_loss = np.mean(loss_lasso(z_lista_train, x, reg))
print('Lista training loss: {}'.format(avg_loss))

Z_plot = ((loss_ista - loss_lista) / loss_fit).reshape(n_samples, n_samples)
Z_plot = np.sign(Z_plot) * np.abs(Z_plot) ** .2
M = np.max(np.abs(Z_plot))
plt.figure()
plt.set_cmap('PiYG')
Cs = plt.contourf(grid[0], grid[1], Z_plot, 50, vmin=-M, vmax=M)
for dk in D:
    plt.arrow(0, 0, dk[0], dk[1])


# for c in combinations(np.arange(n_atoms), 2):
#     dm = D[c, :].mean(axis=0)
#     dm /= np.linalg.norm(dm)
#     plt.arrow(0, 0, dm[0], dm[1], linewidth=1, color='orange')
#
#     dm2 = [dm[1], -dm[0]]
#     plt.arrow(0, 0, dm2[0], dm2[1], linewidth=1, color='blue')

U, S, V = np.linalg.svd(D, full_matrices=False)
M = V
for i in range(1):
    plt.arrow(0, 0, M[i, 0], M[i, 1], color='blue')
plt.colorbar(Cs)
plt.scatter(x[:, 0], x[:, 1], color='r', s=1)

plt.title('Loss ista - loss lista (green= lista is better)')
plt.show()
