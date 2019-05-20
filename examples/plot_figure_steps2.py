import numpy as np

from copy import deepcopy
from joblib import Memory
import matplotlib.pyplot as plt

from adopty.lista import Lista
from adopty.datasets import make_coding
from setup import colors, rc

mem = Memory(location='.', verbose=0)


plt.rcParams.update(rc)

n_dim = 10
n_atoms = 20
n_samples = 1000
n_test = 1000
n_layers = 20
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
    L_s = np.zeros((n_layers, n_samples))
    for layer in range(1, n_layers+1):
        z_ = network.transform(x, reg, output_layer=layer)
        supports = z_ != 0
        S_pca = []
        for support in supports:
            idx = np.where(support)[0]
            D_s = D[idx]
            G_s = D_s.T.dot(D_s)
            S_pca.append(np.linalg.eigvalsh(G_s)[-1])
        L_s[layer-1] = np.array(S_pca)
    return steps, L, L_s, S_pca


steps, L, L_S, S_pca = get_steps(n_dim, n_atoms, n_samples, n_test, n_layers,
                                 reg, rng, max_iter)
ls_steps = 1 / L_S
n_quantiles = 11
quantiles = np.linspace(0, 0.95, n_quantiles)
# avg_ls = np.mean(ls_steps, axis=1)
quants = np.array([np.quantile(ls_steps, q, axis=1) for q in quantiles])
f, ax = plt.subplots(1, 2, figsize=(3, 2), sharey=True,
                     gridspec_kw={'width_ratios': [3, 1]})
plt.subplots_adjust(wspace=0)
xlim = np.arange(1, n_layers + 1)
ax[0].plot(xlim, steps,
           color=colors['SLISTA'], label='Learned steps', linewidth=3)
ax[0].plot(xlim, quants[n_quantiles // 2], color='darkgoldenrod',
           label=r'Median $\frac{1}{L_S}$', linewidth=3)
for i in range(n_quantiles // 2):
    ax[0].fill_between(xlim, quants[i], quants[n_quantiles - i - 1],
                       color='sandybrown', alpha=1.5 * (i + 1)/n_quantiles)
ax[0].hlines(1 / L, 1, 9, color='k', linestyle='--')
ax[0].hlines(1 / L, 11, n_layers, color='k', linestyle='--')
ax[0].text(9.5, 1/L * 0.93, r'$\frac{1}{L}$')
# lgd_ = ax[0].legend(loc='upper left')
x_ = ax[0].set_xlabel('Layer')
y_ = ax[0].set_ylabel('Step')
ax[0].set_xticks([1, 10, 20])
ax[0].set_xlim([1, n_layers])
ax[0].grid()
ax[1].hist(1 / np.array(S_pca), orientation='horizontal', bins=30,
           density='normed', color='sandybrown', alpha=0.8,
           histtype='bar', label=r'$\frac{1}{L_S}$')
# ax[1].legend(handlelength=0.5)
ax[1].set_xticks([])
ax[1].set_ylim([0.18, 0.67])
# plt.hlines(2 / L_S, 0, n_layers, color='b', linestyle='--',
#            label=r'$\frac{2}{L_S}$')
lgd_ = f.legend(ncol=3, loc='upper center', handletextpad=0.1,
                handlelength=1, columnspacing=.8)
plt.subplots_adjust(top=0.75)
plt.savefig('examples/figures/learned_steps.pdf',
            bbox_extra_artists=[lgd_, x_, y_],
            bbox_inches='tight')
# plt.show()
