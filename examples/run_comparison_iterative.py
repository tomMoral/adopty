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
import itertools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from adopty.ista import ista
from adopty.fista import fista
from adopty.oracle_ista import oracle_ista as oista

from adopty.stopping_criterions import stop_on_value, stop_on_no_decrease
from adopty.datasets import make_coding

START = time.time()
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(30, 38)

###########################################
# Parameters of the simulation
###########################################

verbose = 1

# base string for the save names.
base_name = 'run_comparison_iterative'
# n_jobs for the parallel running of single core methods
n_jobs = 4
# number of random states
n_states = 10
# loop over parameters
reg_list = [.1, .4, .8]
# Input dimension
n_dims = 100
n_atoms = 200


#########################################
# List of functions used in the benchmark
#########################################
methods = [
    [ista, 'ISTA'],
    [fista, 'FISTA'],
    [oista, 'Oracle ISTA (proposed)'],
]


############################################
# Benchmark computation function
############################################
def run_one(reg, n_dims, n_atoms, random_state):

    current_time = time.time() - START
    msg = f"{random_state} - {reg}: started at T={current_time:.0f} sec"
    print(colorify(msg, BLUE))

    x, D, _ = make_coding(n_samples=1, n_atoms=n_atoms, n_dim=n_dims,
                          normalize=True, random_state=random_state)

    def stopping_criterion(costs):
        return stop_on_no_decrease(1e-13, costs)

    _, cost, *_ = ista(D, x, reg, max_iter=int(1e6),
                       stopping_criterion=stopping_criterion)
    cost_stop = cost[-1] + 1e-8

    def stopping_criterion(costs):
        return stop_on_value(cost_stop, costs)

    results = {}
    for method, label in methods:
        if 'Oracle' in label:
            x_ = x[0]
        else:
            x_ = x
        _, cost, times, *_ = method(D, x_, reg, max_iter=int(1e6),
                                    stopping_criterion=stopping_criterion)
        results[label] = cost, np.cumsum(times)

    duration = time.time() - START - current_time
    msg = (f"{random_state} - {reg}: done in {duration:.0f} sec "
           f"at T={current_time:.0f} sec")
    print(colorify(msg, GREEN))

    return (reg, n_dims, n_atoms, random_state, label,
            cost_stop, results)


def colorify(message, color=BLUE):
    """Change color of the standard output"""
    return ("\033[1;%dm" % color) + message + "\033[0m"


############################################
# Run the benchmark
############################################
if __name__ == '__main__':

    save_name = os.path.join('figures', base_name)

    iterator = itertools.product(reg_list, range(n_states))

    delayed_run_one = delayed(run_one)
    results = Parallel(n_jobs=n_jobs)(delayed_run_one(
        reg, n_dims, n_atoms, random_state)
        for reg, random_state in iterator)

    results_df = pd.DataFrame(
        results, columns='reg n_dims n_atoms random_state label '
        'cost_stop results'.split(' '))
    results_df.to_pickle(save_name + '.pkl')

    print('-- End of the script --')
