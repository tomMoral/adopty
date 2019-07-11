"""
Compare the different iterative solvers on one LASSO problem.

- Use example/comparison_oracle_ista.py to run the benchmark.
  The results are saved in figures/.
"""

import numpy as np
import matplotlib.pyplot as plt

from adopty.ista import ista
from adopty.fista import fista
from adopty.oracle_ista import oracle_ista


############################################
# Configure matplotlib and joblib cache
############################################
from setup import colors, rc
plt.rcParams.update(rc)


###########################################
# Fix random seed for reproducible figures
###########################################
seed = np.random.randint(100000)
seed = 92518
rng = np.random.RandomState(seed)


###########################################
# Parameters of the simulation
###########################################
n_atoms = 50
n_dims = 10
max_iter = 200
D = rng.randn(n_atoms, n_dims)
x = rng.randn(n_dims)
x /= np.max(np.abs(D.dot(x)))
lmbd = 0.5


############################################
# Benchmark computation function
############################################
_, cost_ista, times_ista = ista(D, x[None, :], lmbd, max_iter=max_iter)
_, cost_fista, times_fista = fista(D, x[None, :], lmbd, max_iter=max_iter)
_, cost_oista, times_oista, steps = oracle_ista(D, x, lmbd, max_iter=max_iter)
cost_ista = np.array(cost_ista)
cost_oista = np.array(cost_oista)
cost_fista = np.array(cost_fista)
z_hat, c_star, _ = ista(D, x[None, :], lmbd, max_iter=3000)
c_star = c_star[-1]


############################################
# Plot the results
############################################
f, axes = plt.subplots(2, 1, figsize=(4, 3), sharex=True,
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
textx = len(steps) / 2 - 3
text_width = 10
ax2.hlines(1., 0, textx - 4, color='k', linestyle='--')
ax2.hlines(1., textx + text_width, len(steps), color='k', linestyle='--')
ax2.text(textx, .7, r'$\frac{1}{L}$', fontsize=15)
ax2.set_ylim(0.1, max(steps/steps[0]) + .2)
ax2.set_ylabel(r'Oracle step')
ax2.set_xlim(0, len(steps)-1)
x = ax2.set_xlabel(r'Number of iterations')
y = ax.set_ylabel(r'$F_x - F_x^*$')
ax2.grid()
plt.subplots_adjust(top=0.85)
plt.savefig('figures/comparison_oista_ista.pdf',
            bbox_extra_artists=[lgd, x, y], bbox_inches='tight')

# Separate plot for step sizes in presentations
fig = plt.figure(figsize=(6, 3))
ax3 = fig.subplots()
ax3.plot(steps / steps[0], color=colors['OISTA'], linewidth=2)
textx = len(steps) / 2 - 3
text_width = 10
ax3.hlines(1., 0, textx - 4, color='k', linestyle='--')
ax3.hlines(1., textx + text_width, len(steps), color='k', linestyle='--')
ax3.text(textx, .7, r'$\frac{1}{L}$', fontsize=15)
ax3.set_ylim(0.5, max(steps/steps[0]) + .2)
ax3.set_ylabel(r'Oracle step')
ax3.set_xlim(0, len(steps)-1)
x = ax3.set_xlabel(r'Number of iterations')
y = ax.set_ylabel(r'$F_x - F_x^*$')
ax3.grid()
ax3.set_xticks([0, 50, 100, 150])
plt.savefig('figures/comparison_oista_ista_steps.pdf')

plt.show()
