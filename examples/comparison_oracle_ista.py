import numpy as np

import matplotlib.pyplot as plt
from adopty.ista import ista
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
lmbd = 0.01

_, cost_ista, times_ista = ista(D, x, lmbd, max_iter=max_iter)
_, cost_oista, times_oista, steps = oracle_ista(D, x, lmbd, max_iter=max_iter)
cost_ista = np.array(cost_ista)
cost_oista = np.array(cost_oista)
z_hat, c_star, _ = ista(D, x, lmbd, max_iter=3000)
c_star = c_star[-1]
f, ax = plt.subplots(figsize=(4, 3))
ax.semilogy(cost_ista - c_star, label='ISTA')
ax.semilogy(cost_oista - c_star,
            label='oracle ISTA')
plt.legend()
ax2 = ax.twinx()
ax2.plot(steps, 'r--')
ax2.set_ylabel('Oracle steps')

ax.set_xlabel('Number of ISTA iterations')
ax.set_ylabel(r'$F - F^*$')
plt.show()
