import time
import numpy as np
import pandas as pd
import matplotlib as mpl
from itertools import product
import matplotlib.pyplot as plt

from adopty.lista import Lista
from adopty.datasets import make_coding

from joblib import Parallel, delayed
from joblib import Memory

# Configure matplotlib
mpl.rc('font', size=18)
mpl.rc('mathtext', fontset='cm')


INIT_PRINT_FMT = 'Initial loss: {:2e}'
FINAL_PRINT_FMT = 'Final loss: {:2e}, delta= {:2e}'


mem = Memory(location='.', verbose=0)


START = time.time()
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(30, 38)


def colorify(message, color=BLUE):
    """Change color of the standard output"""
    return ("\033[1;%dm" % color) + message + "\033[0m"


####################################################################
#  Fitting function
#
@mem.cache
def run_one(parametrization, reg, n_samples, n_test, n_atoms, n_dim,
            n_layers, max_iter, random_state, reg_tests):

    current_time = time.time() - START
    msg = (f"{parametrization} - {reg}: "
           f"started at T={current_time:.0f} sec")
    print(colorify(msg, BLUE))

    # Generate samples
    x, D, z = make_coding(
        n_samples=n_samples + n_test, n_atoms=n_atoms, n_dim=n_dim,
        random_state=random_state)

    x_test = x[n_samples:]
    x = x[:n_samples]

    ista = Lista(D, 1000)
    c_star = ista.score(x, reg) - eps

    c_star_test = {}
    for reg_test in reg_tests:
        c_star_test[reg_test] = ista.score(x_test, reg_test) - eps

    loss_ista = np.array([ista.score(x, reg, output_layer=layer + 1)
                          for layer in range(n_layers)])
    losses_test_ista = {}
    for reg_test in reg_tests:
        losses_test_ista[reg_test] = np.array([
            ista.score(x_test, reg_test, output_layer=layer + 1)
            for layer in range(n_layers)])

    name = f"{parametrization} - {reg}"
    network = Lista(D, n_layers=n_layers, max_iter=max_iter, name=name,
                    parametrization=parametrization)
    init_score = network.score(x_test, reg_tests[0])
    print(INIT_PRINT_FMT.format(init_score))
    network.fit(x, reg)

    final_score = network.score(x_test, reg_tests[0])
    print(FINAL_PRINT_FMT.format(final_score, init_score - final_score))

    # Compute the losses
    loss = network.compute_loss(x, reg)
    losses_test = {}
    for reg_test in reg_tests:
        losses_test[reg_test] = network.compute_loss(x_test, reg_test)

    duration = time.time() - START - current_time
    msg = (f"{parametrization} - {reg}: done in {duration:.0f} sec "
           f"at T={current_time:.0f} sec")
    print(colorify(msg, GREEN))

    return (parametrization, reg, n_samples, n_test, n_atoms, n_dim,
            n_layers, random_state, max_iter, reg_tests, c_star, c_star_test,
            loss, losses_test, loss_ista, losses_test_ista)


if __name__ == "__main__":
    n_dim = 10
    n_atoms = 20
    n_samples = 1000
    n_test = 1000
    n_layers = 30
    eps = 1e-15
    n_jobs = 8
    max_iter = 10000
    random_state = 27

    reg_tests = np.logspace(-2, 0, 40)
    # reg_trains = [.9, .7, .5, .3, .1, .05]
    reg_trains = [.05]
    methods = ['step', 'coupled']

    iterator = product(methods, reg_trains)

    delayed_run_one = delayed(run_one)
    results = Parallel(n_jobs=n_jobs)(
        delayed_run_one(parametrization, reg, n_samples, n_test,
                        n_atoms, n_dim, n_layers, max_iter, random_state,
                        reg_tests)
        for parametrization, reg in iterator
    )

    df = pd.DataFrame(results, columns='parametrization reg n_samples n_test '
                      'n_atoms n_dim n_layers random_state max_iter reg_tests '
                      'c_star c_star_test loss loss_test loss_ista '
                      'loss_test_ista'.split(' '))
    df.to_pickle('figures/change_reg_test.pkl')

    base_style = dict(linewidth=3)
    method_styles = {
        'ista': dict(label='ISTA', color='indigo'),
        'coupled': dict(label='LISTA', color='mediumseagreen'),
        'step': dict(label='SLISTA', color='indianred'),
    }

    eps = 1e-8

    for reg_train in reg_trains:

        this_res = df[df.reg == reg_train]
        fig = plt.figure(f"$\lambda$ = {reg_train}", figsize=(6, 4))
        ax = fig.gca()

        patch = []
        labels = []
        losses = {}
        for i in this_res.T:
            c_star = this_res.c_star[i] - eps
            method = this_res.parametrization[i]
            loss = this_res.loss[i]
            reg_tests = this_res.reg_tests[i]
            loss = []
            loss_ista = []
            for reg_test in reg_tests:
                curve_ista = this_res.loss_test_ista[i][reg_test]
                c_test = this_res.loss_test[i][reg_test][-1]
                c_star_test = this_res.c_star_test[i][reg_test] - eps
                c_ista = curve_ista[-1] - c_star_test
                loss.append((c_test - c_star_test) / c_ista)
                loss_ista.append(1)
                losses[method] = loss
            losses['ista'] = loss_ista

        for method, style in method_styles.items():
            loss = losses[method]
            style_ = base_style.copy()
            style_.update(**style)
            ax.plot(reg_tests, loss, **style_)
        ax.set_ylabel('Relative gain')
        ax.set_xlabel('Regularization parameter $\lambda_{test}$')
        ax.set_yscale('log')
        ax.set_xscale('log')

        # Display the training lambda
        ylim = ax.get_ylim()
        ax.vlines(reg_train, *ylim, color='k', linestyle='--')
        ax.text(reg_train, ylim[1] / 10,
                "$\lambda_{train}$", ha='center', backgroundcolor='w')
        ax.set_ylim(ylim)
        ax.set_xlim(reg_tests.min(), reg_tests.max())

        fig.legend(loc='upper center', ncol=4)
        plt.tight_layout()
        plt.subplots_adjust(top=.85)
        fig.savefig(f"figures/change_reg_train_{reg_train}.pdf")

    plt.show()
