import numpy as np
from joblib import Memory
from itertools import product
import matplotlib.pyplot as plt


from adopty.lista import Lista
from adopty.datasets import make_coding
from joblib import Parallel, delayed


############################################
# Configure matplotlib and joblib cache
############################################
from setup import colors, rc
plt.rcParams.update(rc)

mem = Memory(location='.', verbose=0)


###########################################
# Parameters of the simulation
###########################################
n_dim = 5
n_atoms = 10
power = 4
n = 3 * 10 ** power
n_samples = np.logspace(1, power, 10, dtype=int)
n_layers = 10
reg = 0.3
rng = 1
max_iter = 500
training = 'recursive'
n_jobs = 3
n_avg = 10


############################################
# Benchmark computation function
############################################
@mem.cache
def get_curve(n_dim, n_atoms, n, n_samples, n_layers, reg, rng, max_iter,
              training):
    x, D, z = make_coding(n_samples=n, n_atoms=n_atoms,
                          n_dim=n_dim, random_state=rng)
    x_test = x[n_samples[-1]:]
    c_star = Lista(D, 1000).score(x_test, reg)

    @delayed
    def compute_loss(n_sample, parametrization):
        x_train = x[:n_sample]
        lista = Lista(D, n_layers, parametrization=parametrization,
                      max_iter=max_iter, per_layer=training, verbose=1)
        lista.fit(x_train, reg)
        sc = lista.score(x_test, reg)
        print(n_sample, parametrization, sc)
        return sc
    params = ['coupled', 'step']
    op_list = Parallel(n_jobs=n_jobs)(
            compute_loss(n_sample, param)
            for (param, n_sample) in product(params, n_samples))
    loss_lista = op_list[:len(n_samples)]
    loss_slista = op_list[len(n_samples):]
    np.save('lista.npy', loss_lista)
    np.save('slista.npy', loss_slista)
    np.save('c_star.npy', c_star)
    return np.array(loss_lista), np.array(loss_slista), c_star


############################################
# Run the benchmark
############################################
get_curve(n_dim, n_atoms, n, n_samples, n_layers, reg, rng, max_iter, training)


############################################
# Plot the results
############################################
loss_lista = np.load('loss_lista.npy')
loss_slista = np.load('loss_slista.npy')
# loss_ista = Lista(D, n_layers).score(x_test, reg)
loss_lista = np.median(loss_lista, axis=1)
loss_slista = np.median(loss_slista, axis=1)
f = plt.figure(figsize=(3, 2))
idx = np.where(n_samples > 30)[0]
n_samples = n_samples[idx]
loss_lista = loss_lista[idx]
loss_slista = loss_slista[idx]

plt.plot(n_samples, loss_slista, label='SLISTA (proposed)',
         color=colors['SLISTA'], linewidth=3)
plt.plot(n_samples, loss_lista, label='LISTA', color=colors['LISTA'],
         linewidth=3)
# plt.hlines(loss_ista, n_samples[0], n_samples[-1], color=colors['ISTA'],
#            label='ISTA', linewidth=3)
x_ = plt.xlabel('Training samples')
y_ = plt.ylabel('Median test loss')
plt.yscale('log')
plt.xscale('log')
plt.grid()
lgd = f.legend(ncol=1, loc='upper right', handletextpad=0.1, handlelength=0.8,
               columnspacing=.7, bbox_to_anchor=(1.1, 1.1))
# plt.subplots_adjust(top=0.85)
plt.savefig('examples/figures/learning_curve.pdf',
            bbox_extra_artists=[lgd, x_, y_],
            bbox_inches='tight')
plt.show()
