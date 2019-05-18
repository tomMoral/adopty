import numpy as np

import matplotlib.pyplot as plt
from adopty.ista import ista
from adopty.fista import fista
from adopty.oracle_ista import oracle_ista

rc = {"pdf.fonttype": 42, 'text.usetex': True}
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


f, axes = plt.subplots(2, 1, figsize=(4, 3), sharex=True,
                       gridspec_kw={'height_ratios': [3, 1]})
ax = axes[0]
ax.semilogy(cost_ista - c_star, label=r'ISTA', color='cornflowerblue')
ax.semilogy(cost_fista - c_star, label=r'FISTA', color='orange')
ax.semilogy(cost_oista - c_star,
            label=r'Oracle ISTA',
            color='indianred')
ax.grid()
lgd = ax.legend(ncol=3, bbox_to_anchor=(0.5, 1.2), loc='upper center')
ax2 = axes[1]
ax2.plot(steps / steps[0], color='indianred')
textx = len(steps) / 2 - 3
text_width = 10
ax2.hlines(1., 0, textx - 4, color='k', linestyle='--')
ax2.hlines(1., textx + text_width, len(steps), color='k', linestyle='--')
ax2.text(textx, .4, r'$\frac{1}{L}$', fontsize=12)
ax2.set_ylim(0.1, max(steps/steps[0]) + .2)
ax2.set_ylabel(r'Oracle step')
ax2.set_xlim(0, len(steps)-1)
x = ax2.set_xlabel(r'Number of iterations')
y = ax.set_ylabel(r'$F_x - F_x^*$')
ax2.grid()
plt.savefig('comparison_oista_ista.pdf', bbox_extra_artists=[lgd, x, y],
            bbox_inches='tight')
plt.show()
