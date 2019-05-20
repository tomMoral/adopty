
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str,
                    default="figures/run_comparison_network.pkl")
args = parser.parse_args()


# Configure matplotlib
mpl.rc('font', size=18)
mpl.rc('mathtext', fontset='cm')


# Load data
data_frame = pd.read_pickle(args.file)

base_style = {
    'linewidth': 3
}
method_styles = {
    'ISTA': {'color': 'indigo'},
    'LISTA': {'color': 'mediumseagreen'},
    'ALISTA': {'color': 'cornflowerblue'},
    'SLISTA (proposed)': {'color': 'indianred'},
}


eps = 1e-8
loss_ista = np.array(data_frame.loss_ista[0])

fig = plt.figure(figsize=(6, 4))
ax = fig.gca()
c_star = data_frame.c_star[0] - eps
style = base_style.copy()
style.update(method_styles['ISTA'])
ax.plot(loss_ista - c_star, label='ISTA', **style)

for loss, name in zip(data_frame.loss, data_frame.label):
    loss = np.r_[loss_ista[0], loss]
    style = base_style.copy()
    style.update(method_styles[name])
    ax.plot(loss - c_star, label=name, **style)


ax.set_xticks([0, 10, 20, 30])
ax.set_ylabel('$F_x - F_x^*$')
ax.set_xlabel('Number of Layers/Iterations')
ax.set_ylim(1e-8, 1e-1)
ax.set_xlim(0, 30)

ncol = 2
ax.set_yscale("log")
ax.grid(True)
fig.legend(loc='upper right', ncol=ncol, columnspacing=0.8)

fig.tight_layout()
fig.subplots_adjust(top=.75)


fig.savefig("figures/comparison_network")

plt.show()
