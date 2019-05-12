import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from time import time

from adopty.lista import Lista
from adopty.datasets import make_coding
from adopty.loss_and_gradient import cost_lasso

n_dim = 2
n_atoms = 4
n_samples = 100
n_layers = 40
reg = 0.1
rng = 0

x, D, z = make_coding(n_samples=n_samples, n_atoms=n_atoms, n_dim=n_dim,
                      random_state=rng)
lista = Lista(D, n_layers, parametrization='step',
              max_iter=1000)
print('Initial loss: {}'.format(lista.score(x, reg)))
lista.fit(x, reg)
print('Final loss: {}'.format(lista.score(x, reg)))
z_ = lista.transform(x, reg)
supports = z_ != 0
# Compute the sparse pcas of D
S_pca = []
for support in supports:
    D_s = D[support]
    G_s = D_s.T.dot(D_s)
    S_pca.append(np.linalg.eigvalsh(G_s)[-1])
L = np.linalg.norm(D, ord=2) ** 2  # Lipschitz constant
steps = [p.detach().numpy()[0] for p in lista.parameters()]

plt.figure()
plt.plot(steps, label='steps')
plt.hlines(1 / L, 0, n_layers, linestyle='--', label='1 / L')
plt.hlines(2 / np.max(S_pca), 0, n_layers, color='r',
           linestyle='--', label='2 / L_S')
plt.legend()
plt.show()
