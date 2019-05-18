import numpy as np

from copy import deepcopy
from joblib import Memory
import matplotlib.pyplot as plt

from adopty.lista import Lista
from adopty.datasets import make_coding

mem = Memory(location='.', verbose=0)

rc = {"pdf.fonttype": 42, 'text.usetex': True}
plt.rcParams.update(rc)

n_dim = 5
n_atoms = 10
n_samples = 1000
n_test = 1000
n_layers = 30
reg = 0.2
rng = 1
max_iter = 100


@mem.cache
def get_steps(n_dim, n_atoms, n_samples, n_test, n_layers, reg, rng, max_iter):
    lista_kwargs = dict(
        n_layers=n_layers,
        max_iter=max_iter)
    x, D, z = make_coding(n_samples=n_samples + n_test, n_atoms=n_atoms,
                          n_dim=n_dim, random_state=rng)
    x_test = x[n_samples:]
    x = x[:n_samples]

    c_star = Lista(D, 1000).score(x_test, reg)
    L = np.linalg.norm(D, ord=2) ** 2  # Lipschitz constant
    network = Lista(D, **lista_kwargs,
                    parametrization='step', per_layer='oneshot')
    init_score = network.score(x_test, reg)
    print(init_score)
    network.fit(x, reg)
    print(network.score(x_test, reg))
    steps = network.get_parameters(name='step_size')
    z_ = network.transform(x, reg)
    supports = np.unique(z_ != 0, axis=0)
    S_pca = []
    for support in supports:
        D_s = D[support]
        G_s = D_s.T.dot(D_s)
        S_pca.append(np.linalg.eigvalsh(G_s)[-1])
    L_s = np.max(S_pca)
    return steps, L, L_s


steps, L, L_S = get_steps(n_dim, n_atoms, n_samples, n_test, n_layers, reg,
                          rng, max_iter)
plt.figure(figsize=(4, 2))
plt.plot(steps, color='indianred', label='Learned steps')
plt.hlines(1 / L, 0, n_layers, color='k', linestyle='--',
           label=r'$\frac{1}{L}$')
plt.hlines(2 / L_S, 0, n_layers, color='b', linestyle='--',
           label=r'$\frac{2}{L_S}$')
lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.6, 0.7))
x_ = plt.xlabel('Layer')
y_ = plt.ylabel('Step')
plt.savefig('learned_steps.pdf', bbox_extra_artists=[lgd, x_, y_],
            bbox_inches='tight')
plt.show()
