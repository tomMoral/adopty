"""
Benchmark different network solvers of the same LASSO problem.

- Use example/run_comparison_networks.py to run the benchmark.
  The results are saved in figures/.
- Use example/plot_comparison_networks.py to plot the results.
  The figures are saved in figures/.
"""

from __future__ import print_function
import os
import time
from itertools import product

import torch
import pandas as pd
from joblib import Memory
from joblib import Parallel, delayed

from adopty.ista import ista
from adopty.lista import Lista

from adopty.datasets import make_coding, make_image_coding
from adopty.stopping_criterions import stop_on_no_decrease

# number of jobs and GPUs accessible for the parallel running of the methods
N_JOBS = 4
N_GPU = 0


# Constants for logging in console.
START = time.time()
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(30, 38)

# jobib cache to avoid loosing computations
mem = Memory(location='.', verbose=0)

##############################
# Parameters of the simulation
##############################

verbose = 1

# base string for the save names.
base_name = 'run_comparison_networks'


####################################################
# List of parameters to loop on in the simulation
####################################################

regs = [.1, .8]
datasets = ['images', 'simulations']
parametrizations = [
    'coupled',
    'alista',
    'step',
]


###################################
# Helper function for outputs
###################################

def colorify(message, color=BLUE):
    """Change color of the standard output"""
    return ("\033[1;%dm" % color) + message + "\033[0m"


################################################
# Calling function of the benchmark
################################################

@mem.cache
def run_one(parametrization, data, reg, n_layer, max_iter, n_samples, n_test,
            n_atoms, n_dim, random_state):

    # try to avoid having dangling memory
    torch.cuda.empty_cache()

    # Stread the computations on the different GPU. This strategy
    # might fail and some GPU might be overloaded if some workers are
    # re-spawned.
    if N_GPU == 0:
        device = None
    else:
        pid = os.getpid()
        device = f"cuda:{pid % N_GPU}"

    tag = f"[{parametrization} - {n_layer}]"
    current_time = time.time() - START
    msg = f"{tag} started at T={current_time:.0f} sec (device={device})"
    print(colorify(msg, BLUE))

    if data == "images":
        x, D, _ = make_image_coding(n_samples=n_samples + n_test,
                                    n_atoms=n_atoms, normalize=True,
                                    random_state=random_state)
    elif data == "simulations":
        x, D, _ = make_coding(n_samples=n_samples + n_test, n_atoms=n_atoms,
                              n_dim=n_dim, normalize=True,
                              random_state=random_state)

    x_test = x[n_samples:]
    x = x[:n_samples]

    network = Lista(D, n_layers=n_layer,
                    parametrization=parametrization,
                    max_iter=max_iter, per_layer='oneshot',
                    device=device, name=parametrization)
    network.fit(x, reg)
    loss = network.score(x_test, reg)
    training_loss = network.training_loss_

    duration = time.time() - START - current_time
    msg = (f"{tag} done in {duration:.0f} sec "
           f"at T={current_time:.0f} sec")
    print(colorify(msg, GREEN))

    return (parametrization, data, reg, n_layer, max_iter, n_samples, n_test,
            n_atoms, n_dim, random_state, loss, training_loss)


###################################
# Main script of the benchmark
###################################


if __name__ == '__main__':

    n_dim = 64
    n_atoms = 256
    n_test = 10000
    n_samples = 10000
    random_state = 42

    max_iter = 2000
    n_layers = 30

    save_name = os.path.join('figures', base_name)

    args = (max_iter, n_samples, n_test, n_atoms, n_dim,
            random_state)

    iterator = product(parametrizations, datasets, regs,
                       range(n_layers))

    if N_JOBS == 1:
        results = [run_one(parametrization, data, reg, n_layer + 1, *args)
                   for parametrization, data, reg, n_layer in iterator]
    else:
        delayed_run_one = delayed(run_one)
        results = Parallel(n_jobs=N_JOBS, batch_size=1)(
            delayed_run_one(parametrization, data, reg, n_layer + 1, *args)
            for parametrization, data, reg, n_layer in iterator)

    for data in datasets:
        if data == "images":
            x, D, _ = make_image_coding(n_samples=n_samples + n_test,
                                        n_atoms=n_atoms, normalize=True,
                                        random_state=random_state)
        elif data == "simulations":
            x, D, _ = make_coding(n_samples=n_samples + n_test,
                                  n_atoms=n_atoms, n_dim=n_dim,
                                  normalize=True, random_state=random_state)
        x_test = x[n_samples:]
        x = x[:n_samples]
        for reg in regs:

            def stopping_criterion(costs):
                return stop_on_no_decrease(1e-13, costs)

            _, cost_test, *_ = ista(D, x_test, reg, max_iter=int(1e7),
                                    stopping_criterion=stopping_criterion)
            c0 = cost_test[0]
            c_star = cost_test[-1]

            results = [(*r, c0, c_star) if r[2] == data and r[3] == reg else r
                       for r in results]
            for n_layer in range(n_layers):
                results.append(
                    ('ista', data, reg, n_layer + 1, *args,
                     cost_test[n_layer + 1], None, c0, c_star)
                )

    results_df = pd.DataFrame(
        results, columns='parametrization data reg n_layer max_iter n_samples '
        'n_test n_atoms n_dim random_state loss training_loss c0 c_star'
        .split(' '))
    results_df.to_pickle(save_name + '.pkl')

    print('-- End of the script --')
