
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


##############################################
# Configure matplotlib
##############################################
from setup import colors, rc
plt.rcParams.update(rc)


###########################################
# Load data from the pickle file
###########################################
data_frame = pd.read_pickle("figures/run_comparison_iterative.pkl")


###########################################
# Define the styles for the plot
###########################################
width = 1
method_labels = [
    dict(label='ISTA', color=colors['ISTA']),
    dict(label='Oracle ISTA (proposed)', color=colors['OISTA']),
    dict(label='FISTA', color=colors['FISTA']),
]

##############################################
# Plot the results
##############################################
regs = data_frame['reg'].unique()

fig_it = plt.figure(figsize=(11, 4))
ax_it = fig_it.gca()
fig_time = plt.figure(figsize=(11, 4))
ax_time = fig_time.gca()
for i, reg in enumerate(regs):

    x_position = i * width * (len(method_labels) + 2)
    rect_it_list, rect_time_list = [], []
    for j, style in enumerate(method_labels):
        time, it = [], []
        name = style['label']
        for res in data_frame.query(f'reg == {reg}').results:
            cost, times = res[name]
            time += [times[-1]]
            it += [len(times)]

        rect = ax_it.bar(x=x_position + j * width, align='edge',
                         height=np.mean(it), width=1, **style)
        rect_it_list.append(rect)
        rect = ax_time.bar(x=x_position + j * width, align='edge',
                           height=np.mean(time), width=1, **style)
        rect_time_list.append(rect)

        # Add the actual numbers to show the dispersion
        ax_it.plot(
            np.ones_like(it) * x_position + (j + .5) * width,
            it, '_', color='k')
        ax_time.plot(
            np.ones_like(time) * x_position + (j + .5) * width,
            time, '_', color='k')

ax_it.set_ylabel('Iterations')
ax_time.set_ylabel('Time (s)')

ncol = 3
offset = width * len(method_labels) / 2.0
x_positions = np.arange(len(regs)) * width * (len(method_labels) + 2)
labels = [r'$\lambda=%s$' % r for r in regs]
for fig, ax in [(fig_it, ax_it), (fig_time, ax_time)]:
    ax.set_xticks(x_positions + offset)
    ax.set_xticklabels(labels, ha='center', fontsize=22)
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    ax.legend_.remove()
    fig.legend(rect_it_list, legend_labels, loc='upper center',
               ncol=ncol, columnspacing=0.8)

    fig.tight_layout()
    fig.subplots_adjust(top=.85)


fig_it.savefig("figures/comparison_iterative")
fig_time.savefig("figures/comparison_iterative_time")

plt.show()
