"""
Benchmark different network solvers of the same LASSO problem.

- Use example/run_comparison_networks.py to run the benchmark.
  The results are saved in figures/.
- Use example/plot_comparison_networks.py to plot the results.
  The figures are saved in figures/.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


##############################
# Configure matplotlib
##############################
from setup import colors, rc
plt.rcParams.update(rc)


# Can change input files using `--file FILE` option of the script.
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str,
                    default="figures/run_comparison_networks.pkl")
args = parser.parse_args()

#########################################
# Load data from the pickle file
#########################################
df = pd.read_pickle(args.file)

#########################################
# Define the styles for the plot
#########################################
base_style = {
    'linewidth': 3
}
method_styles = {
    'ista': dict(label='ISTA', color=colors['ISTA']),
    'coupled': dict(label="LISTA", color=colors['LISTA']),
    'alista': dict(label="ALISTA", color=colors['ALISTA']),
    'step': dict(label="SLISTA (proposed)", color=colors['SLISTA']),
}


#########################################
# Plot the results
#########################################
eps = 1e-8

regs = df.reg.unique()
datasets = df.data.unique()

for reg in regs:
    for data in datasets:
        this_expe = df[df.data == data]
        this_expe = this_expe[this_expe.reg == reg]
        fig = plt.figure(figsize=(6, 5))
        ax = fig.gca()
        c_star = this_expe.c_star.iloc[0] - eps
        n_layers = this_expe.n_layer.max()

        for method, style in method_styles.items():
            this_network = this_expe[this_expe.parametrization == method]
            loss = [this_network.c0.iloc[0]]
            layers = [0]
            for i in this_network.T:
                layers.append(this_network.n_layer[i])
                loss.append(this_network.loss[i] - this_network.c_star[i])
            loss = np.array(loss)

            style_ = base_style.copy()
            style_.update(style)
            plt.plot(layers, loss, **style_)

        ax.set_xticks([0, 10, 20, 30])
        ax.set_ylabel('$F_x - F_x^*$')
        ax.set_xlabel('Number of Layers/Iterations')
        # ax.set_ylim(1e-8, 1e-1)
        ax.set_xlim(0, n_layers)

        ncol = 2
        ax.set_yscale("log")
        ax.grid(True)
        fig.legend(loc='upper right', ncol=ncol, columnspacing=0.8)

        fig.tight_layout()
        fig.subplots_adjust(top=.8)

        suffix = f"_data_{data}_reg_{str(reg).replace('.', ',')}"
        fig.savefig(f"figures/comparison_images{suffix}")

plt.show()
