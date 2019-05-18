
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from adopty.utils.viz import color_palette


# Configure matplotlib
mpl.rc('font', size=18)
mpl.rc('mathtext', fontset='cm')


# Load data
data_frame = pd.read_pickle("figures/run_comparison_network.pkl")

method_labels = [
    ('ISTA', {}),
    ('LISTA', {}),
    ('ALISTA', {}),
    ('cLISTA', {}),
    ('SLISTA', {}),
]

colors = color_palette(len(method_labels))


eps = 1e-10
loss_ista = np.array(data_frame.loss_ista[0])

fig = plt.figure(figsize=(11, 4))
ax = fig.gca()

for loss, c_star, name in zip(data_frame.loss, data_frame.c_star,
                              data_frame.label):
    loss = np.r_[loss_ista[0], loss]
    ax.plot(loss - c_star + eps, label=name)

ax.plot(loss_ista - c_star + eps, 'k--', label='ISTA')

ax.set_ylabel('Loss')
ax.set_xlabel('Layer/ Iteration')

ncol = 4
ax.set_yscale("log")
ax.grid(True)
fig.legend(loc='upper center', ncol=ncol, columnspacing=0.8)

fig.tight_layout()
fig.subplots_adjust(top=.85)


fig.savefig("figures/comparison_network")

plt.show()
