"""
Benchmark different iterative solvers of the same LASSO problem.

- Use run_comparison_iterative.py to run the benchmark.
  The results are saved in adopty/figures.
- Use plot_comparison_iterative.py to plot the results.
  The figures are saved in adopty/figures.
"""

from __future__ import print_function
import os
import time
import pandas as pd
from itertools import product

from joblib import Memory
from joblib import Parallel, delayed

from adopty.ista import ista
from adopty.lista import Lista

from adopty.datasets import make_image_coding
from adopty.stopping_criterions import stop_on_no_decrease

N_GPU = 0
START = time.time()
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(30, 38)


mem = Memory(location='.', verbose=0)


##############################
# Parameters of the simulation
##############################

verbose = 1

# base string for the save names.
base_name = 'run_comparison_images'
# n_jobs for the parallel running of single core methods
n_jobs = 6


#########################################
# List of functions used in the benchmark
#########################################


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


###################################
# Calling function of the benchmark
###################################

@mem.cache
def run_one(parametrization, n_layer, max_iter, reg, n_samples, n_test,
            n_atoms, random_state):

    if N_GPU == 0:
        device = None
    else:
        device = f"cuda:{n_layer % N_GPU}"

    tag = f"[{parametrization} - {n_layer}]"
    current_time = time.time() - START
    msg = f"{tag} started at T={current_time:.0f} sec (device={device})"
    print(colorify(msg, BLUE))

    x, D, _ = make_image_coding(n_samples=n_samples + n_test, n_atoms=n_atoms,
                                normalize=True, random_state=random_state)

    x_test = x[n_samples:]
    x = x[:n_samples]

    network = Lista(D, n_layers=n_layer + 1,
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

    return (parametrization, n_layer, max_iter, reg, n_samples, n_test,
            n_atoms, random_state, loss, training_loss)


if __name__ == '__main__':

    reg = .8
    n_atoms = 256
    n_test = 2000
    n_samples = 2000
    random_state = 42

    max_iter = 500
    n_layers = 30

    save_name = os.path.join('figures', base_name)

    args = (max_iter, reg, n_samples, n_test, n_atoms,
            random_state)

    iterator = product(parametrizations, range(1, n_layers + 1))

    if n_jobs == 1:
        results = [run_one(parametrization, n_layer, *args)
                   for parametrization, n_layer in iterator]
    else:
        delayed_run_one = delayed(run_one)
        results = Parallel(n_jobs=n_jobs)(
            delayed_run_one(parametrization, n_layer, *args)
            for parametrization, n_layer in iterator)

    x, D, _ = make_image_coding(n_samples=n_samples + n_test, n_atoms=n_atoms,
                                normalize=True, random_state=random_state)

    x_test = x[n_samples:]
    x = x[:n_samples]

    def stopping_criterion(costs):
        return stop_on_no_decrease(1e-13, costs)

    _, cost_test, *_ = ista(D, x_test, reg, max_iter=int(1e7),
                            stopping_criterion=stopping_criterion)
    c0 = cost_test[0]
    c_star = cost_test[-1]
    loss_ista = cost_test[:n_layers + 1]

    for n_layer in range(n_layers):
        results.append(
            ('ista', n_layer + 1, *args, loss_ista[n_layer + 1], None)
        )
    results = [(*r, c0, c_star) for r in results]

    results_df = pd.DataFrame(
        results, columns='parametrization n_layer max_iter reg n_samples '
        'n_test n_atoms random_state loss training_loss c0 c_star'
        .split(' '))
    results_df.to_pickle(save_name + '.pkl')

    print('-- End of the script --')
