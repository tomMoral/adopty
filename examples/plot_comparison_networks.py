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
id_ax = {
    ('simulations', .1): (0, "Simulated data $\lambda=0.1$"),
    ('simulations', .8): (1, "Simulated data $\lambda=0.8$"),
    ('images', .1): (2, "Digits data $\lambda=0.1$"),
    ('images', .8): (3, "Digits data $\lambda=0.8$")
}
ncols = len(id_ax)


FILE = "figures/run_comparison_network_m=256_n=64_reg=0.8_maxiter=10000.pkl"

#########################################
# Plot the results
#########################################
eps = 1e-8

regs = df.reg.unique()
datasets = df.data.unique()

fig, axes = plt.subplots(ncols=ncols, figsize=(3 * ncols, 3))
for data in datasets:
    for reg in regs:
        idx = (data, reg)
        if idx not in id_ax:
            continue
        idx, title = id_ax[idx]
        ax = axes[idx]
        this_expe = df[df.data == data]
        this_expe = this_expe[this_expe.reg == reg]
        ista_expe = this_expe[this_expe.parametrization == 'ista']
        c_star = ista_expe.c_star.iloc[0] - eps
        c0 = ista_expe.c0.iloc[0] - c_star
        n_layers = this_expe.n_layer.max()

        if data == 'simulations' and reg == .8:
            df_hack = pd.read_pickle(FILE)
            for i in range(3):
                loss = np.array(df_hack.loss.iloc[i])
                loss_ista = np.array(df_hack.loss_ista.iloc[i])
                c_star = df_hack.c_star.iloc[i]
                method = df_hack.method.iloc[i]
                if method not in method_styles:
                    continue
                style_ = base_style.copy()
                style_.update(method_styles[method])
                ax.plot(np.r_[loss_ista[0], loss] - c_star, **style_)
            style_ = base_style.copy()
            style_.update(method_styles['ista'])
            ax.plot(loss_ista - c_star, **style_)

        else:
            lines = []
            for method, style in method_styles.items():
                this_network = this_expe[this_expe.parametrization == method]
                loss = [c0]
                layers = [0]
                for i in this_network.T:
                    layers.append(this_network.n_layer[i])
                    loss.append(this_network.loss[i] - c_star)
                loss = np.array(loss)

                style_ = base_style.copy()
                style_.update(style)
                lines.extend(ax.plot(layers, loss, **style_))

        ax.set_xticks([0, 10, 20, 30])
        labels = ax.get_yticklabels()
        ax.set_yticklabels(labels, fontsize=12)
        if idx == 0:
            ax.set_ylabel('$F_x - F_x^*$')
        ax.set_xlabel('Number of Layers')
        ax.set_yscale("log")
        ax.set_title(title)
        ylim = ax.get_ylim()
        ax.set_ylim(max(ylim[0], 5e-8), ylim[1])
        ax.set_xlim(0, n_layers)

        ax.grid(True)

ncol = 4
fig.legend(lines, [l.get_label() for l in lines],
           loc='upper center', ncol=ncol, columnspacing=0.8)
fig.tight_layout()
fig.subplots_adjust(top=.75, wspace=.2)
fig.savefig(f"figures/comparison_networks")

# plt.show()
