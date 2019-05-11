from sys import getsizeof
import numpy as np
import matplotlib.pyplot as plt
import torch
from adopty.lista import Lista
from adopty.datasets import make_coding
from adopty.loss_and_gradient import cost_lasso


from itertools import combinations, permutations
from copy import deepcopy
from pympler.asizeof import asizeof

seed = np.random.randint(0, 1000)
print(seed)
rng = np.random.RandomState(seed)

def loss_lasso(z, x, D, reg):
    res = np.dot(z, D) - x
    return 0.5 * np.sum(res ** 2, axis=1) + reg * np.sum(np.abs(z), axis=1)


n_dim = 2
n_atoms = 8
n_s = 1000

x, D, z = make_coding(n_s, n_atoms, n_dim, rng)
L = np.linalg.norm(D, ord=2) ** 2
reg = 0.5


def spca_bourin(A, k_=None):
    k, _ = A.shape
    if k_ is None:
        k_ = k
    l_list = []
    for i in range(1, k_+1):
        l_max = 0.
        for idx in permutations(range(k), i):
            B = A[idx, :][:, idx]
            l_max = max(l_max, np.linalg.norm(B, ord=2))
        l_list.append(l_max)
    return l_list


step_list = spca_bourin(D.dot(D.T), n_atoms)


z_star = Lista(D, 1000).transform(x, reg)
l_star = loss_lasso(z_star, x, D, reg)


l_lista = []
l_ista = []
layers = [30]
#layers = [2]
for n_layers in layers:
    ista = Lista(D, n_layers)
    z_ista = ista.transform(x, reg)
    loss_ista = loss_lasso(z_ista, x, D, reg)
    lista = Lista(D, n_layers, parametrization='coupled', learn_threshold=True, max_iter=500).fit(x, reg)
    z_lista = lista.transform(x, reg)
    loss_lista = loss_lasso(z_lista, x, D, reg)
    l_lista.append(np.mean(loss_lista - l_star))
    l_ista.append(np.mean(loss_ista - l_star))

l_lista = np.array(l_lista)
l_ista = np.array(l_ista)
