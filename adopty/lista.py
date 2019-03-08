import torch
import numpy as np

from ._compat import AVAILABLE_CONTEXT


PARAMETRIZATIONS = ["lista", "hessian", "coupled"]


class Lista(torch.nn.Module):
    """L-ISTA network for the LASSO problem

    Parameters
    ----------

    """
    def __init__(self, D, n_layers, parametrization="lista", name="LISTA",
                 ctx=None):
        if ctx:
            msg = "Context {} is not available on this computer."
            assert ctx in AVAILABLE_CONTEXT, msg.format(ctx)
        else:
            ctx = AVAILABLE_CONTEXT[0]

        if parametrization not in PARAMETRIZATIONS:
            raise NotImplementedError("Could not find parametrization='{}'. "
                                      "Should be in {}".format(
                                          parametrization, PARAMETRIZATIONS
                                      ))

        self._ctx = ctx
        self.n_layers = n_layers
        self.parametrization = parametrization

        self.D = np.array(D)
        self.B = D.dot(D.T)
        self.L = np.linalg.norm(self.B, ord=2)

        self.params = []

        self.init_network_torch()

    def init_network_torch(self, params=[]):
        super().__init__()
        n_atoms = self.D.shape[0]
        I_k = np.eye(n_atoms)

        self.params = []
        for i in range(self.n_layers):
            if len(params) > i:
                param = params[i]
            else:
                if self.parametrization == "lista":
                    param = [I_k - self.B / self.L, self.D.T / self.L]
                elif self.parametrization == "coupled":
                    param = [self.D.T / self.L]
                elif self.parametrization == "hessian":
                    param = [I_k / self.L]
                else:
                    raise NotImplementedError()

            self.params += [[torch.nn.Parameter(torch.from_numpy(p))
                             for p in param]]

    def forward(self, x, lmbd, z0=None):
        # Compat numpy
        if isinstance(x, np.ndarray):
            x = torch.autograd.Variable(torch.Tensor(x).double())

        D = torch.Tensor(self.D).double()

        z_hat = z0
        # Compute the following layers
        for p in self.params:
            if self.parametrization == "lista":
                if z_hat is None:
                    z_hat = x.matmul(p[1])
                else:
                    z_hat = z_hat.matmul(p[0]) + x.matmul(p[1])
            elif self.parametrization == "coupled":
                if z_hat is None:
                    z_hat = x.matmul(p[0])
                else:
                    res = z_hat.matmul(D) - x
                    z_hat -= res.matmul(p[0])
            elif self.parametrization == "hessian":
                if z_hat is None:
                    z_hat = x.matmul(D.t()).matmul(p[0])
                else:
                    grad = (z_hat.matmul(D) - x).matmul(D.t())
                    z_hat -= grad.matmul(p[0])
            else:
                raise NotImplementedError()
            z_hat = torch.nn.functional.softshrink(z_hat, lmbd / self.L)

        return z_hat
