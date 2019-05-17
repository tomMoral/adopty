import numpy as np

import matplotlib.pyplot as plt
from adopty.ista import ista
from adopty.fista import fista
from adopty.oracle_ista import oracle_ista

seed = np.random.randint(100000)
# seed = 3416
# seed = 51656
rng = np.random.RandomState(seed)

n_atoms = 100
n_dims = 10
max_iter = 1000
D = rng.randn(n_atoms, n_dims)
x = rng.randn(n_dims)
x /= np.max(np.abs(D.dot(x)))
lmbd = 0.1

_, cost_ista, times_ista = ista(D, x[None, :], lmbd, max_iter=max_iter)
_, cost_fista, times_fista = fista(D, x[None, :], lmbd, max_iter=max_iter,
                                   tol=0)
_, cost_oista, times_oista, steps = oracle_ista(D, x, lmbd, max_iter=max_iter)
cost_ista = np.array(cost_ista)
cost_oista = np.array(cost_oista)
cost_fista = np.array(cost_fista)
z_hat, c_star, _ = ista(D, x[None, :], lmbd, max_iter=3000)
c_star = c_star[-1]
f, axes = plt.subplots(2, 1, figsize=(4, 3), sharex=True)
ax = axes[0]
ax.semilogy(cost_ista - c_star, label='ISTA')
ax.semilogy(cost_fista - c_star, label='FISTA')
ax.semilogy(cost_oista - c_star,
            label='Oracle ISTA',
            color='indianred')
ax.legend()
ax2 = axes[1]
ax2.plot(steps / steps[0], color='indianred')
ax2.hlines(1., 0, len(steps), color='k', linestyle='--')
ax2.set_ylabel('Oracle steps')
ax.set_xlabel('Number of ISTA iterations')
ax.set_ylabel(r'$F - F^*$')
plt.show()
