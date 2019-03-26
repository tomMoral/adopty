import torch
import numpy as np

from .utils import check_tensor
from ._compat import AVAILABLE_CONTEXT
from .loss_and_gradient import cost_lasso


PARAMETRIZATIONS = ["lista", "hessian", "coupled"]


class Lista(torch.nn.Module):
    """L-ISTA network for the LASSO problem

    Parameters
    ----------

    """
    def __init__(self, D, n_layers, parametrization="lista", solver="sgd",
                 max_iter=100, name="LISTA", ctx=None, verbose=1, device=None):
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

        self.max_iter = max_iter
        self.solver = solver
        self._ctx = ctx
        self.device = device
        self.verbose = verbose
        self.n_layers = n_layers
        self.parametrization = parametrization

        self.D = np.array(D)
        self.D_ = check_tensor(self.D, device=device)
        self.B = D.dot(D.T)
        self.L = np.linalg.norm(self.B, ord=2)

        self.params = []

        self.init_network_parameters()

    def init_network_parameters(self, parameters_init=[]):
        super().__init__()
        n_atoms = self.D.shape[0]
        I_k = np.eye(n_atoms)

        self.params = []
        for i in range(self.n_layers):
            if len(parameters_init) > i:
                parameters_layer = parameters_init[i]
            else:
                if self.parametrization == "lista":
                    parameters_layer = [I_k - self.B / self.L,
                                        self.D.T / self.L]
                elif self.parametrization == "coupled":
                    parameters_layer = [self.D.T / self.L]
                elif self.parametrization == "hessian":
                    parameters_layer = [I_k / self.L]
                else:
                    raise NotImplementedError()
            parameters_layer = [
                torch.nn.Parameter(check_tensor(p, device=self.device))
                for p in parameters_layer]
            for j, p in enumerate(parameters_layer):
                self.register_parameter("layer{}-p{}".format(i, j), p)

            self.params += [parameters_layer]

    def forward(self, x, lmbd, z0=None, output_layer=None):
        # Compat numpy
        x = check_tensor(x, device=self.device)
        z0 = check_tensor(z0, device=self.device)

        if output_layer is None:
            output_layer = self.n_layers
        elif output_layer > self.n_layers:
            raise ValueError("Requested output from out-of-bound layer "
                             "output_layer={} (n_layers={})"
                             .format(output_layer, self.n_layers))

        z_hat = z0
        # Compute the following layers
        for p in self.params[:output_layer]:
            if self.parametrization == "lista":
                if z_hat is None:
                    z_hat = x.matmul(p[1])
                else:
                    z_hat = z_hat.matmul(p[0]) + x.matmul(p[1])
            elif self.parametrization == "coupled":
                if z_hat is None:
                    z_hat = x.matmul(p[0])
                else:
                    res = z_hat.matmul(self.D_) - x
                    z_hat = z_hat - res.matmul(p[0])
            elif self.parametrization == "hessian":
                if z_hat is None:
                    z_hat = x.matmul(self.D_.t()).matmul(p[0])
                else:
                    grad = (z_hat.matmul(self.D_) - x).matmul(self.D_.t())
                    z_hat = z_hat - grad.matmul(p[0])
            else:
                raise NotImplementedError()
            z_hat = torch.nn.functional.softshrink(z_hat, lmbd / self.L)

        return z_hat

    def fit(self, x, lmbd):
        # Compat numpy
        x = check_tensor(x, device=self.device)

        if self.verbose > 1:
            # compute fix point
            z_hat = self.transform(x, lmbd)
            for i in range(100):
                z_hat = self.transform(x, lmbd, z0=z_hat)
            c_star = cost_lasso(z_hat.numpy(), self.D, x, lmbd)

        parameters = [p for params in self.params for p in params]

        training_loss = []

        max_iter_per_layer = [self.max_iter // self.n_layers] * self.n_layers
        # max_iter_per_layer = np.diff(
        #     np.logspace(0, np.log10(self.max_iter), self.n_layers + 1,
        #                 dtype=int)) + 1000

        for n_layer in range(1, self.n_layers + 1):
            lr = 1
            max_iter = max_iter_per_layer[n_layer - 1]
            for i in range(max_iter):

                # Compute the forward operator
                self.zero_grad()
                z_hat = self(x, lmbd, output_layer=n_layer)
                loss = self.loss_fn(x, lmbd, z_hat)

                # Verbosity of the output
                if self.verbose > 5 and i % 10 == 0:
                    loss_val = loss.detach().cpu().numpy()
                    print(i, loss_val - c_star)
                elif self.verbose > 0 and i % 50 == 0:
                    print("\rFitting model (layer {}/{}): {:7.2%}"
                          .format(n_layer, self.n_layers,
                                  (i+1) / max_iter),
                          end="", flush=True)

                # Back-tracking line search
                if len(training_loss) > 0 and torch.le(training_loss[-1],
                                                       loss):
                    if i + 1 == max_iter:
                        # In this case, do not perform the last step
                        lr *= 2
                    lr = self.backtrack_parameters(parameters, lr)
                    continue

                # Accepting the previous point
                training_loss.append(loss)

                # Next gradient iterate
                loss.backward()
                lr = self.update_parameters(parameters, lr=lr)

        self.training_loss_ = np.array([loss.detach().cpu().numpy()
                                        for loss in training_loss])
        print("\rFitting model: done".ljust(80))
        return self

    def loss_fn(self, x, lmbd, z_hat):
        n_samples = x.shape[0]
        x = check_tensor(x, device=self.device)

        res = z_hat.matmul(self.D_) - x
        return (0.5 * (res * res).sum() +
                lmbd * torch.abs(z_hat).sum()) / n_samples

    def update_parameters(self, parameters, lr):
        lr = min(4 * lr, 1e6)
        self._saved_gradient = []
        for p in parameters:
            if p.grad is None:
                self._saved_gradient.append(None)
                continue
            if self.parametrization == "hessian":
                p.data.add_(-lr, p.grad.data + p.grad.data.t())
                self._saved_gradient.append(
                    p.grad.data.clone() + p.grad.data.t().clone())
            else:
                p.data.add_(-lr, p.grad.data)
                self._saved_gradient.append(p.grad.data.clone())

        return lr

    def backtrack_parameters(self, parameters, lr):
        lr /= 2
        for p, g in zip(parameters, self._saved_gradient):
            if g is None:
                continue
            p.data.add_(lr, g)
        return lr

    def transform(self, x, lmbd, z0=None, output_layer=None):
        with torch.no_grad():
            return self(x, lmbd, z0=z0,
                        output_layer=output_layer).cpu().numpy()

    def score(self, x, lmbd, z0=None):
        x = check_tensor(x, device=self.device)
        with torch.no_grad():
            return self.loss_fn(x, lmbd, self(x, lmbd, z0=z0)).cpu().numpy()
