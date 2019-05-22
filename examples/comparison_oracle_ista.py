import numpy as np

import matplotlib.pyplot as plt
from adopty.ista import ista
from adopty.fista import fista
from adopty.oracle_ista import oracle_ista

from setup import colors, rc

plt.rcParams.update(rc)

seed = np.random.randint(100000)
seed = 92518
# seed = 3416
# seed = 51656
rng = np.random.RandomState(seed)

n_atoms = 50
n_dims = 10
max_iter = 200
D = rng.randn(n_atoms, n_dims)
x = rng.randn(n_dims)
x /= np.max(np.abs(D.dot(x)))
lmbd = 0.5

_, cost_ista, times_ista = ista(D, x[None, :], lmbd, max_iter=max_iter)
_, cost_fista, times_fista = fista(D, x[None, :], lmbd, max_iter=max_iter)
_, cost_oista, times_oista, steps = oracle_ista(D, x, lmbd, max_iter=max_iter)
cost_ista = np.array(cost_ista)
cost_oista = np.array(cost_oista)
cost_fista = np.array(cost_fista)
z_hat, c_star, _ = ista(D, x[None, :], lmbd, max_iter=3000)
c_star = c_star[-1]


f, axes = plt.subplots(2, 1, figsize=(4, 2.5), sharex=True,
                       gridspec_kw={'height_ratios': [3, 1]})
ax = axes[0]
ax.semilogy(cost_ista - c_star, label=r'ISTA', color=colors['ISTA'],
            linewidth=2)
ax.semilogy(cost_fista - c_star, label=r'FISTA', color=colors['FISTA'],
            linewidth=2)
ax.semilogy(cost_oista - c_star,
            label=r'OISTA (proposed)',
            color=colors['OISTA'], linewidth=2)
ax.grid()
ax.set_ylim([1e-15, cost_ista[0] - c_star])
ax.set_yticks([1e-6, 1e-12])
lgd = f.legend(ncol=3, loc='upper center', handletextpad=0.1, handlelength=0.9,
               columnspacing=.7)
ax2 = axes[1]
ax2.plot(steps / steps[0], color=colors['OISTA'], linewidth=2)
textx = len(steps) / 2 - 10
text_width = 24
ax2.hlines(1., 0, textx - 2, color='k', linestyle='--')
ax2.hlines(1., textx + text_width, len(steps), color='k', linestyle='--')
ax2.text(textx, .7, r'$1/L$', fontsize=15)
ax2.set_ylim(0.1, max(steps/steps[0]) + .2)
ax2.set_ylabel('Oracle \n step')
ax2.set_xlim(0, len(steps)-1)
ax2.set_yticks([1, 3])
x = ax2.set_xlabel(r'Number of iterations')
y = ax.set_ylabel(r'$F_x - F_x^*$')
# ax2.yaxis.grid(True)
plt.subplots_adjust(top=0.82)
plt.savefig('examples/figures/comparison_oista_ista.pdf',
            bbox_extra_artists=[lgd, x, y], bbox_inches='tight')
# plt.show()
