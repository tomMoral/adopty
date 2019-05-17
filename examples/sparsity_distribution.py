import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc


rc = {"pdf.fonttype": 42, 'text.usetex': True}
plt.rcParams.update(rc)
rng = np.random.RandomState(0)


def lipschitz(D):
    return np.linalg.eigvalsh(D.T.dot(D))[-1]


def theoretical_law(zeta, gamma):
    return ((1 + np.sqrt(gamma)) / (1 + np.sqrt(gamma * zeta))) ** -2


def approximate_law(zeta):
    return zeta


n = 200
gamma = 3
zeta_list = np.linspace(0, 1, 10)

L_list = []

m = int(n * gamma)
D = rng.randn(n, m)
L = lipschitz(D)

for zeta in zeta_list:
    k = int(zeta * m)
    if k == 0:
        k = 1
    S = rng.choice(np.arange(m), size=k, replace=False)
    Ds = D[:, S]
    Ls = lipschitz(Ds)
    L_list.append(Ls / L)

plt.figure(figsize=(3.5, 2.5))
plt.plot(zeta_list, L_list, label=r'Empirical law', linewidth=3)
plt.plot(zeta_list, theoretical_law(zeta_list, gamma), linewidth=2,
         label=r'$\big(\frac{1 + \sqrt{\zeta\gamma}}{1 + \sqrt{\gamma}} \big)^2$')
plt.plot(zeta_list, zeta_list, linewidth=3, linestyle= '--', color='k',
         label=r'$\zeta$')
x_ = plt.xlabel(r'$\zeta$')
y_ = plt.ylabel(r'$\frac{L_S}{L}$')
plt.legend()
plt.savefig('lip_distrib.pdf', bbox_extra_artists=[x_, y_],
            bbox_inches='tight')
plt.show()
