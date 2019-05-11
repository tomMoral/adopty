# first line: 74
@mem.cache
def get_trained_lista(D, x, reg, n_layers, max_iter):

    lista = Lista(D, n_layers, max_iter=max_iter)
    lista.fit(x, reg)
    return lista
