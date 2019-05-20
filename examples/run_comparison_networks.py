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
from joblib import Parallel, delayed


from adopty.ista import ista
from adopty.lista import Lista

from adopty.datasets import make_coding
from adopty.stopping_criterions import stop_on_no_decrease

START = time.time()
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(30, 38)

##############################
# Parameters of the simulation
##############################

verbose = 1

# base string for the save names.
base_name = 'run_comparison_network'
# n_jobs for the parallel running of single core methods
n_jobs = 4
# number of random states
n_states = 1
# loop over parameters
reg_list = [.1, .4, .8]


#########################################
# List of functions used in the benchmark
#########################################


methods = [
    ['coupled', 'LISTA'],
    ['alista', 'ALISTA'],
    ['step', 'SLISTA (proposed)'],
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


def run_one(method, max_iter, reg, n_samples, n_test, n_dims,
            n_atoms, n_layers, random_state):

    method, label = method

    current_time = time.time() - START
    msg = f"{label} - {reg}: started at T={current_time:.0f} sec"
    print(colorify(msg, BLUE))

    x, D, _ = make_coding(n_samples=n_samples + n_test, n_atoms=n_atoms,
                          n_dim=n_dims, normalize=True,
                          random_state=random_state)

    x_test = x[n_samples:]
    x = x[:n_samples]

    def stopping_criterion(costs):
        return stop_on_no_decrease(1e-13, costs)

    _, cost_test, *_ = ista(D, x_test, reg, max_iter=int(1e7),
                            stopping_criterion=stopping_criterion)
    c_star = cost_test[-1]
    loss_ista = cost_test[:n_layers + 1]

    loss = []
    training_losses = {}
    for n_layer in range(n_layers):
        network = Lista(D, n_layers=n_layer + 1, parametrization=method,
                        max_iter=max_iter, per_layer='oneshot',
                        name=label)
        network.fit(x, reg)
        loss += [network.score(x_test, reg)]
        training_losses[n_layer] = network.training_loss_

    duration = time.time() - START - current_time
    msg = (f"{label} - {reg}: done in {duration:.0f} sec "
           f"at T={current_time:.0f} sec")
    print(colorify(msg, GREEN))

    return (label, method, max_iter, reg, n_samples, n_test, n_dims,
            n_atoms, n_layers, random_state, loss, training_losses,
            loss_ista, c_star)


if __name__ == '__main__':

    reg = .8
    n_dims = 2
    n_atoms = 10
    n_test = 2000
    n_samples = 2000
    random_state = 42

    max_iter = 2000
    n_layers = 20

    save_name = os.path.join('figures', base_name)

    args = (max_iter, reg, n_samples, n_atoms, n_dims, n_atoms,
            n_layers, random_state)

    if n_jobs == 1:
        results = [run_one(method, *args) for method in methods]
    else:
        delayed_run_one = delayed(run_one)
        results = Parallel(n_jobs=n_jobs)(delayed_run_one(
            method, *args) for method in methods)

    # iterator = itertools.product(reg_list, range(n_states))

    results_df = pd.DataFrame(
        results, columns='label method max_iter reg n_samples '
        'n_test n_dims n_atoms n_layers random_state loss training_losses '
        'loss_ista c_star'.split(' '))
    results_df.to_pickle(save_name + '.pkl')

    print('-- End of the script --')
