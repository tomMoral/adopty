import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from time import time

from adopty.lista import Lista
from adopty.datasets import make_coding
from adopty.loss_and_gradient import cost_lasso
import torch

n_dim = 2
n_atoms = 4
n_samples = 1000
n_test = 1000
n_layers = 30
reg = 0.5
rng = 0
init_print = 'Initial loss: {:2e}'
final_print = 'Final loss: {:2e}, delta= {:2e}'

x, D, z = make_coding(n_samples=n_samples + n_test, n_atoms=n_atoms,
                      n_dim=n_dim, random_state=rng)
x_test = x[n_samples:]
x = x[:n_samples]
lista = Lista(D, n_layers, parametrization='step',
              max_iter=1000)
init_score = lista.score(x_test, reg)
print(init_print.format(init_score))
lista.fit(x, reg)
losses = np.array([lista.score(x_test, reg, output_layer=layer)
                   for layer in range(1, n_layers + 1)])
final_score = lista.score(x_test, reg)
print(final_print.format(final_score, init_score - final_score))

lista_ = Lista(D, n_layers, parametrization='step',
               max_iter=100)
init_score = lista_.score(x_test, reg)
print(init_print.format(init_score))
lista_.fit(x, reg, per_layer=True)
losses_ = np.array([lista_.score(x_test, reg, output_layer=layer)
                    for layer in range(1, n_layers + 1)])
final_score_ = lista_.score(x_test, reg)
print(final_print.format(final_score, init_score - final_score_))


L_s_list = []
for layer in range(n_layers):
    z_ = lista.transform(x, reg, output_layer=layer+1)
    supports = z_ != 0
    # Compute the sparse pcas of D
    S_pca = []
    for support in supports:
        D_s = D[support]
        G_s = D_s.T.dot(D_s)
        S_pca.append(np.linalg.eigvalsh(G_s)[-1])
    L_s_list.append(np.max(S_pca))
L_s_list = np.array(L_s_list)
# hack_params = lista_.params
#
# hack_params[-1][0] = torch.Tensor([1 / L_s_list[-1]]).double()
#
# lista_hack = Lista(D, n_layers,
#                    parametrization='step',
#                    max_iter=100)
# lista_hack.params = hack_params
# final_score_ = lista_hack.score(x_test, reg)
# print(final_print.format(final_score, init_score - final_score_))
L = np.linalg.norm(D, ord=2) ** 2  # Lipschitz constant
steps = [p.detach().numpy()[0] for p in lista.parameters()]

plt.figure()
plt.plot(steps, label='one shot. Test loss = {:2e}'.format(final_score))
plt.plot([p.detach().numpy()[0] for p in lista_.parameters()],
         label='recursive. Test loss = {:2e}'.format(final_score_))
plt.hlines(1 / L, 0, n_layers, linestyle='--', label='1 / L')
plt.hlines(2 / L_s_list[-1], 0, n_layers, color='r',
           linestyle='--', label='2 / L_S')
plt.ylabel('Step')
plt.xlabel('Layer')
plt.legend()
plt.show()

c_star = Lista(D, 1000).score(x_test, reg)
plt.figure()
plt.semilogy(losses - c_star, label='one shot')
plt.semilogy(losses_ - c_star, label='recursive')
plt.ylabel('Loss')
plt.xlabel('Layer')
plt.legend()
plt.show()


plt.figure()
spar = [np.mean(lista.transform(x_test, reg, output_layer=i) != 0)
        for i in range(1, n_layers)]
spar_ = [np.mean(lista_.transform(x_test, reg, output_layer=i) != 0)
         for i in range(1, n_layers)]
plt.plot(spar, label='one shot')
plt.plot(spar_, label='recursive')
plt.ylabel('Sparsity')
plt.xlabel('Layer')
plt.legend()
plt.show()
