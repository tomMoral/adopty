import numpy as np
import matplotlib.pyplot as plt


from adopty.lista import Lista
from adopty.datasets import make_coding


############################################
# Configure matplotlib and joblib cache
############################################
from joblib import Memory
from setup import colors, rc
plt.rcParams.update(rc)

mem = Memory(location='.', verbose=0)


###########################################
# Parameters of the simulation
###########################################
n_dim = 10
n_atoms = 20
n_samples = 1000
n_layers = 40
reg = 0.05
rng = 1
max_iter = 10000
training = 'recursive'


############################################
# Benchmark computation function
############################################
@mem.cache
def get_params(n_samples, n_atoms, n_dim, rng, max_iter,
               training, n_layers):
    x, D, z = make_coding(n_samples=n_samples, n_atoms=n_atoms,
                          n_dim=n_dim, random_state=rng)

    lista = Lista(D, n_layers, parametrization='coupled',
                  max_iter=max_iter, per_layer=training, verbose=1)
    lista.fit(x, reg)

    W_list = lista.get_parameters('W_coupled')
    thresholds = lista.get_parameters('threshold')
    return D, W_list, thresholds


############################################
# Run the benchmark
############################################
D, W_list, thresholds = get_params(n_samples, n_atoms, n_dim,
                                   rng, max_iter, training, n_layers)


############################################
# Plotting utilities
############################################
def fro_diff(W, thresholds, D):
    return np.linalg.norm(W - thresholds[:, None] * D)


def cosine_similarity(D, W):
    w = W.ravel()
    d = D.ravel()
    return np.abs(np.dot(w, d)) / np.sqrt(d.dot(d) * w.dot(w))


############################################
# Plot the results
############################################
cs_list = np.array([cosine_similarity(D, W.T) for W in W_list])
f_list = np.array([fro_diff(W.T, t, D) for W, t in zip(W_list, thresholds)])

f = plt.figure(figsize=(4, 2.5))
layers = np.arange(1, n_layers + 1)
plt.plot(layers, f_list, color=colors['LISTA'], label='LISTA',
         linewidth=3)
plt.hlines(0, 1, n_layers, color='k', linestyle='--', linewidth=3)
# plt.yscale('log')
lgd = plt.legend()
plt.xlabel('Layers')
plt.xticks([1, 10, 20, 30, 40])
plt.ylabel(r'$\|\alpha^{(t)}W^{(t)} - \beta^{(t)}D \|_F$')
plt.xlim([1, n_layers])
plt.grid()
plt.savefig('figures/fro_similarity.pdf',
            bbox_extra_artists=[lgd, ], bbox_inches='tight')
plt.show()
