import os
import numpy as np
import pandas as pd
from glob import glob
from time import time
from datetime import datetime
import matplotlib.pyplot as plt

from adopty.sinkhorn import Sinkhorn
from adopty.datasets.optimal_transport import make_ot


BENCH_NAME = "gradient_computations"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')


def run_benchmark(device):
    eps = .01
    n_rep = 50
    n_probe_layers = 20
    max_layers = 100
    n, m, p = 1000, 500, 2
    alpha, beta, C, *_ = make_ot(n, m, p, random_state=None)

    sinkhorn = Sinkhorn(n_layers=max_layers, log_domain=False, device=device)

    layers = np.unique(np.logspace(0, np.log(max_layers), n_probe_layers,
                                   dtype=int))
    n_probe_layers = len(layers)

    layers = np.minimum(max_layers, layers)
    results = []
    for i, nl in enumerate(layers):
        for j in range(n_rep):
            print(f"\r{(i*n_rep + j) / (n_rep * n_probe_layers):.1%}", end='',
                  flush=True)

            t_start = time()
            sinkhorn.gradient_beta_analytic(alpha, beta, C, eps,
                                            output_layer=nl)
            delta_t = time() - t_start

            results.append(dict(
                n_layers=nl, gradient='analytic', time=delta_t
            ))

            t_start = time()
            sinkhorn.gradient_beta(alpha, beta, C, eps, output_layer=nl)
            delta_t = time() - t_start

            results.append(dict(
                n_layers=nl, gradient='autodiff', time=delta_t
            ))

    df = pd.DataFrame(results)
    tag = datetime.now().strftime('%Y-%m-%d_%Hh%M')
    df.to_pickle(os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_{tag}.pkl"))


def plot_benchmark(file_name=None):

    if file_name is None:
        file_pattern = os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_*.pkl")
        file_list = glob(file_pattern)
        file_list.sort()
        file_name = file_list[-1]

    df = pd.read_pickle(file_name)

    fig, ax = plt.subplots()
    df[df.gradient == 'analytic'].groupby('n_layers').median(
     ).plot(y='time', ax=ax, label="analytic")
    df[df.gradient == 'autodiff'].groupby('n_layers').median(
     ).plot(y='time', ax=ax, label='autodiff')
    (2 * df[df.gradient == 'analytic'].groupby('n_layers').median()
     ).plot(y='time', ax=ax, label="2x analytic")
    (3 * df[df.gradient == 'analytic'].groupby('n_layers').median()
     ).plot(y='time', ax=ax, label="3x analytic")
    plt.legend(bbox_to_anchor=(-.02, 1.02, 1., .3), ncol=3,
               fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--gpu', action='store_true',
                        help='If present, use GPU computations')
    parser.add_argument('--plot', action='store_true',
                        help='Show the results from the benchmark')
    parser.add_argument('--file', type=str, default=None,
                        help='File to plot')
    args = parser.parse_args()

    device = 'gpu' if args.gpu else None

    if args.plot:
        plot_benchmark(file_name=args.file)
    else:
        run_benchmark(device=device)

    import IPython
    IPython.embed(colors='neutral')
