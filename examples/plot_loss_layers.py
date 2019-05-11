import numpy as np
import matplotlib.pyplot as plt
from time import time

from adopty.lista import Lista
from adopty.datasets import make_coding
from adopty.loss_and_gradient import cost_lasso


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training')
    args = parser.parse_args()

    device = 'cuda' if args.gpu else 'cpu'
    n_dim = 5
    n_atoms = 8
    n_samples = 2000
    n_samples_train = 1000
    n_layers = 16
    reg = 0.5
    rng = 0
    x, D, z = make_coding(n_samples=n_samples, n_atoms=n_atoms, n_dim=n_dim,
                          random_state=rng)
    x_train = x[:n_samples_train]
    x_test = x[n_samples_train:]
    # Train
    saved_model = {}
    parametrizations = ['coupled', 'step', 'alista']
    for parametrization in parametrizations:
        lista = Lista(D, n_layers, parametrization=parametrization,
                      max_iter=100, device=device)
        t0 = time()
        lista.fit(x_train, reg)
        print('Fitting model "{}" took {:3.0f} sec'.format(parametrization,
                                                           time() - t0))
        saved_model[parametrization] = lista
    plt.figure()
    plt.xlabel('layers')
    plt.ylabel('Test loss')
    c_star = cost_lasso(Lista(D, 1000).transform(x_test, reg),
                        D, x_test, reg)
    for parametrization in parametrizations:
        lista = saved_model[parametrization]
        loss_list = []
        for layer in range(1, n_layers):
            z_hat = lista.transform(x_test, reg, output_layer=layer)
            loss_list.append(cost_lasso(z_hat, D, x_test, reg))
        loss_list = np.array(loss_list)
        plt.semilogy(loss_list - c_star, label=parametrization)
    plt.legend()
    plt.show()
