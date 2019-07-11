import numpy as np

import matplotlib.pyplot as plt

from adopty.ista import ista
from setup import colors, rc


plt.rcParams.update(rc)

seed = np.random.randint(100000)
# seed = 93732
seed = 11506

rng = np.random.RandomState(seed)


def lasso_cost(x, D, z, lbda):
    r = D.dot(z) - x
    return .5 * np.sum(r ** 2) + lbda * np.sum(np.abs(z))


def surrogate(x, D, z, zt, lbda, L):
    r = D.dot(zt) - x
    g = np.sum(D.T.dot(r) * (z - zt))
    quad = .5 * np.sum(r ** 2) + .5 * L * np.sum((z - zt) ** 2) + g
    return quad + lbda * np.sum(np.abs(z))


def t_x(x, D, zt, lbda, alpha):
    z = zt - alpha * np.dot(D.T, D.dot(zt) - x)
    return np.sign(z) * np.maximum(np.abs(z) - alpha * lbda, 0)


n, m = 2, 3
n_points = 100
D = rng.randn(n, m)
x = rng.randn(n)
x /= np.max(np.abs(D.T.dot(x)))
lbda = .1
z = ista(D.T, x[None, :], lbda, max_iter=1000)[0]
zt = z.ravel() + .1 * rng.randn(m)
L = np.linalg.norm(D.T.dot(D), ord=2)

alphas = np.linspace(0, 10 / L, n_points)
z_ = [t_x(x, D, zt, lbda, alpha) for alpha in alphas]

F = [lasso_cost(x, D, z, lbda) for z in z_]
L = .4 * L
S1 = [surrogate(x, D, z, zt, lbda, L) for z in z_]
fac = .6
S2 = [surrogate(x, D, z, zt, lbda, fac * L) for z in z_]


f = plt.figure(figsize=(3, 2))
plt.plot(L * alphas, F, label=r'$F_x$', color='k')
plt.plot(L * alphas, S1, label=r'$Q_{x, L}(\cdot, z^{(t)})$',
         color=colors['ISTA'])
plt.plot(L * alphas, S2, label=r'$Q_{x, L_S}(\cdot, z^{(t)})$',
         color=colors['OISTA'])
vmin = min(F) * .99
plt.ylim(vmin, max(F) * 1.1)
plt.xlim(0, 4)

plt.vlines(1., vmin, np.min(S1), color=colors['ISTA'], linestyles='--')
plt.vlines(1 / fac, vmin, np.min(S2), color=colors['OISTA'], linestyles='--')
lgd = f.legend(ncol=3, loc='upper center', handletextpad=0.1, handlelength=0.8,
               columnspacing=.7)
x = plt.xlabel(r'Step size')
y = plt.ylabel(r'Cost function')
plt.xticks(np.array([0., 1., 1 / fac]),
           labels=[r'$0$',
                   r'$\frac{1}{L}$',
                   r'$\frac{1}{L_S}$'])
plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.show()
# plt.tight_layout()
plt.subplots_adjust(top=0.75)

plt.savefig('figures/surrogate.pdf',
            bbox_extra_artists=[x, y, lgd], bbox_inches='tight')
