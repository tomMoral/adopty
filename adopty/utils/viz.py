import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt


def color_palette(n_colors=4, cmap='viridis', extrema=False):
    """Create a color palette from a matplotlib color map"""
    if extrema:
        bins = np.linspace(0, 1, n_colors)
    else:
        bins = np.linspace(0, 1, n_colors * 2 - 1 + 2)[1:-1:2]

    cmap = plt.get_cmap(cmap)
    palette = list(map(tuple, cmap(bins)[:, :3]))
    return palette


def plot_coding(x, D):
    plt.figure()

    # t = np.linspace(-3, 3, 1000)
    # xx, yy = np.meshgrid(t, t)
    # pts = np.concatenate((xx[None], yy[None]), axis=0)
    # zz = np.sum((pts - x.T[:, :, None]) ** 2, axis=0)
    # plt.contour(xx, yy, zz)
    # plt.contourf(xx, yy, zz)
    # plt.hlines(0, -3, 3)
    # plt.vlines(0, -3, 3)
    for dk in D:
        plt.arrow(0, 0, dk[0], dk[1])

    plt.scatter(x[:, 0], x[:, 1])


def basis_pursuit(x, D, tol=1e-6):

    n_samples, n_dim = x.shape
    n_atoms, n_dim = D.shape

    z = np.zeros((n_samples, n_atoms))
    r = x

    while np.abs(r).sum(axis=1).mean() > tol:
        g = r.dot(D.T)
        i0 = abs(g).argmax(axis=1)
        z[np.arange(n_samples), i0] += g[np.arange(n_samples), i0]
        r = x - z.dot(D)
    return z


def get_X_and_l1(D, n_points=1000):
    t = np.linspace(-2, 2, n_points)
    mesh = xx, yy = np.meshgrid(t, t)
    X = np.c_[xx.ravel(), yy.ravel()]
    l1 = np.abs(basis_pursuit(X, D, tol=1e-12)).sum(axis=1)

    return mesh, X.reshape(n_points, n_points, 2), l1.reshape(n_points,
                                                              n_points)


def plot_cost_func(x, D, reg):
    n_points = 500
    t = np.linspace(-2, 2, 500)
    xx, yy = np.meshgrid(t, t)
    X = np.c_[xx.ravel(), yy.ravel()]
    dft = np.sum((X - x)**2, axis=1) * .5
    l1 = np.abs(basis_pursuit(X, D, tol=1e-12)).sum(axis=1)
    loss = (dft + reg * l1).reshape(n_points, n_points)
    reg = .1
    loss = (dft + reg * l1).reshape(n_points, n_points)
    plt.contourf(xx, yy, loss)


def get_plot_level_lines(D, fig=None):
    if fig is None:
        fig = plt.figure()

    (xx, yy), X, l1 = get_X_and_l1(D)

    levels = np.r_[1e-10, np.logspace(-2, 1, 20)]
    norm = colors.LogNorm(vmin=.05, vmax=10, clip=True)

    def plot_level_lines(reg=.4, theta=.5):
        fig.clear('all')
        ax = fig.add_subplot(1, 1, 1)
        x = np.array([np.cos(theta), np.sin(theta)])
        ax = fig.add_subplot(1, 1, 1)
        dtf = 0.5 * np.sum((X - x[None, None]) ** 2, axis=-1)
        loss = 0 * dtf + reg * l1
        i0 = np.unravel_index(loss.argmin(), loss.shape)
        ax.contourf(xx, yy, loss, levels=levels, norm=norm)
        plt.scatter(x[0], x[1], c='k')
        plt.scatter(xx[0, i0[1]], yy[i0[0], 0], c='C1')
        U, s, V = np.linalg.svd(D, full_matrices=False)
        for dk, uk in zip(D, U):
            ax.arrow(-3 * dk[0], -3 * dk[1], 6 * dk[0], 6 * dk[1], alpha=.2)
            ax.arrow(0, 0, dk[0], dk[1])
            ax.arrow(0, 0, uk[0], uk[1], color='C1')
        for vk in V.T:
            ax.arrow(0, 0, vk[0], vk[1], color='C0')
        ax.axis('equal')

    return plot_level_lines


if __name__ == "__main__":
    from adopty.datasets import make_coding
    from adopty.lista import Lista
    n_dim = 2
    n_atoms = 3
    n_samples = 1
    n_display = 1

    reg = 0.6

    x, D, z = make_coding(n_samples=n_samples, n_atoms=n_atoms, n_dim=n_dim)
    x_train = np.random.randn(1000, n_dim)

    x = D[:1]

    n_layers = 5
    plot_coding(x[:n_display], D)
    lista = Lista(D, 40000, parametrization="hessian")
    path = []
    path2 = []
    for i_layer in range(1, n_layers + 1, 1):
        z_hat = lista.transform(x, reg, output_layer=i_layer)
        x_hat = z_hat.dot(D)
        path.append(z_hat.dot(D))
        z0 = np.zeros((1, n_atoms))
        z0[0] = 1
        z_hat = lista.transform(x, reg, z0=z0, output_layer=i_layer)
        x_hat2 = z_hat.dot(D)
        path2.append(z_hat.dot(D))
    path = np.array(path)
    path2 = np.array(path2)
    plt.plot(path[:, :n_display, 0], path[:, :n_display, 1], 'C1')
    plt.scatter(x_hat[:n_display, 0], x_hat[:n_display, 1], c='C1')
    plt.plot(path2[:, :n_display, 0], path2[:, :n_display, 1], 'C4')
    plt.scatter(x_hat2[:n_display, 0], x_hat2[:n_display, 1], c='C4')

    lista = Lista(D, n_layers, parametrization="hessian", max_iter=10000)
    lista.fit(x_train, reg)
    cmap = plt.get_cmap('viridis')
    path = []
    for i_layer in range(1, n_layers + 1, 1):
        z_hat = lista.transform(x, reg, output_layer=i_layer)
        x_hat = z_hat.dot(D)
        path.append(z_hat.dot(D))
        # plt.scatter(x_hat[:n_display, 0], x_hat[:n_display, 1],
        #             c=np.array([cmap(i_layer / n_layers)]))
    path = np.array(path)
    plt.plot(path[:, :n_display, 0], path[:, :n_display, 1], 'C2')
    plt.scatter(x_hat[:n_display, 0], x_hat[:n_display, 1], c='C2')

    for k, dk in enumerate(D):
        plt.arrow(0, 0, z_hat[0][k] * dk[0], z_hat[0][k] * dk[1], color="r")

    plt.axis('equal')
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))
    plt.show()
