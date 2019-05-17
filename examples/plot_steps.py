import numpy as np

from copy import deepcopy

import matplotlib.pyplot as plt

from adopty.lista import Lista
from adopty.datasets import make_coding

n_dim = 5
n_atoms = 10
n_samples = 1000
n_test = 1000
n_layers = 30
reg = 0.5
rng = 0

lista_kwargs = dict(
    n_layers=n_layers,
    max_iter=3000)

init_print = 'Initial loss: {:2e}'
final_print = 'Final loss: {:2e}, delta= {:2e}'


#######################################################################
#  Generate samples
#
x, D, z = make_coding(n_samples=n_samples + n_test, n_atoms=n_atoms,
                      n_dim=n_dim, random_state=rng)
x_test = x[n_samples:]
x = x[:n_samples]


c_star = Lista(D, 1000).score(x_test, reg)
L = np.linalg.norm(D, ord=2) ** 2  # Lipschitz constant

networks = {}

for parametrization in ['first_step', 'step']:
    for per_layer in ['oneshot', 'recursive']:
        #######################################################################
        #  Training
        #
        name = "{} - {}".format(parametrization, per_layer)
        print(80*"=" + "\n" + name + "\n" + 80*"=")
        network = Lista(D, **lista_kwargs, name=name,
                        parametrization=parametrization, per_layer=per_layer)
        init_score = network.score(x_test, reg)
        print(init_print.format(init_score))
        network.fit(x, reg)
        losses = np.array([network.score(x_test, reg, output_layer=layer + 1)
                           for layer in range(n_layers)])
        final_score = network.score(x_test, reg)
        print(final_print.format(final_score, init_score - final_score))

        networks[name] = network

        #######################################################################
        # Compute maximal sparsity eigenvalue
        #
        L_s_list = []
        for layer in range(n_layers):
            z_ = network.transform(x, reg, output_layer=layer+1)
            supports = np.unique(z_ != 0, axis=0)
            # Compute the sparse pcas of D
            S_pca = []
            for support in supports:
                D_s = D[support]
                G_s = D_s.T.dot(D_s)
                S_pca.append(np.linalg.eigvalsh(G_s)[-1])
            L_s_list.append(np.max(S_pca))
        L_s_list = np.array(L_s_list)

        network_hack = deepcopy(network)
        network_hack.set_parameters('step_size', np.array(2 / L_s_list[-1]),
                                    offset=25)

        #######################################################################
        # Retrieve the step size values
        #
        steps = network.get_parameters(name='step_size')
        steps_hack = network_hack.get_parameters(name='step_size')
        if parametrization == "first_step":
            steps_hack[0] = steps[0] = network.get_parameters('threshold')[0]

        #######################################################################
        # Retrieve the losses
        #
        losses = np.array([network.score(x_test, reg, output_layer=layer + 1)
                           for layer in range(n_layers)])
        losses_hack = np.array([
            network_hack.score(x_test, reg, output_layer=layer + 1)
            for layer in range(n_layers)])

        #######################################################################
        # Retrieve the sparsity
        #
        spar = [np.mean(network.transform(x_test, reg, output_layer=i) != 0)
                for i in range(1, n_layers)]
        spar_hack = [
            np.mean(network_hack.transform(x_test, reg, output_layer=i) != 0)
            for i in range(1, n_layers)]

        #######################################################################
        #  Output plots
        #
        plt.figure("Steps")
        plt.plot(steps, label='{}. Test loss = {:2e}'
                 .format(name, final_score))
        plt.plot(steps_hack, label='{}. Test loss = {:2e}'
                 .format(name + "- hack", final_score))

        plt.figure("Loss")
        plt.semilogy(losses - c_star, label=name)
        plt.semilogy(losses_hack - c_star, label=name + '- hack')

        plt.figure("Sparsity")
        plt.plot(spar, label=name)
        plt.plot(spar_hack, label=name + "- hack")

#######################################################################
#  Format output plots
#
plt.figure("Steps")
plt.hlines(1 / L, 0, n_layers, linestyle='--', label='1 / L')
plt.hlines(2 / L_s_list[-1], 0, n_layers, color='r',
           linestyle='--', label='2 / L_S')
plt.ylabel('Step')
plt.xlabel('Layer')
plt.legend()
plt.tight_layout()

plt.figure("Loss")
plt.ylabel('Loss')
plt.xlabel('Layer')
plt.legend()
plt.tight_layout()

plt.figure("Sparsity")
plt.plot(spar, label=name)
plt.ylabel('Sparsity')
plt.xlabel('Layer')
plt.legend()
plt.tight_layout()

plt.show()
