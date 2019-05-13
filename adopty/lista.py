import torch
import numpy as np
import torch.nn.functional as F

from .utils import check_tensor
from ._compat import AVAILABLE_CONTEXT
from .loss_and_gradient import cost_lasso
from .analytic_weights import get_alista_weights


PARAMETRIZATIONS = {
    "lista": [
        ('threshold', []),
        ('Wx', []),
        ('Wz', []),
    ],
    "hessian": [
        ('threshold', []),
        ('W_hessian', ["sym"]),
    ],
    "coupled": [
        ('threshold', []),
        ('W_coupled', []),
    ],
    "coupled_step": [
        ('step_size', []),
        ('W_coupled', []),
    ],
    "alista": [
        ('threshold', []),
        ('lr', []),
    ],
    "step": [
        ('step_size', []),
    ],
}


def symmetric_gradient(p):
    """Constrain the gradient to be symmetric."""
    p.grad.data.set_(p.grad.data + p.grad.data.t())


GRADIENT_HOOKS = {
    'sym': symmetric_gradient,
}


def soft_thresholding(z_hat, mu):
    return z_hat.sign() * F.relu(z_hat.abs() - mu)


class Lista(torch.nn.Module):
    """L-ISTA network for the LASSO problem

    Parameters
    ----------
    D : ndarray, shape (n_atoms, n_dim)
        Dictionary for the considered sparse coding problem.
    n_layer : int
        Number of layers in the network.
    parametrization : str, (default: "coupled")
        Parametrization for the weight of the network. Should be one of:
        - 'lista': original parametrization from Gregor and Le Cun (2010).
        - 'coupled': one weight parametrization from Chen et al (2018).
        - 'alista': analytic weights from Chen et al (2019).
        - 'hessian': one weight parametrization as a quasi newton technique.
        - 'step': only learn a step size
    learn_th : bool (default: True)
        Wether to learn the thresholds or not.
    solver : str, (default: 'gradient_decent')
        Not implemented for now.
    max_iter : int (default: 100)
        Maximal number of iteration for the training of each layer.
    name : str (default: LISTA)
        Name of the model.
    verbose : int (default: 1)
        Verbosity level.
    device : str or None (default: None)
        Device on which the model is implemented. This parameter should be set
        according to the pytorch API (_eg_ 'cpu', 'gpu', 'gpu/1',..).
    """
    def __init__(self, D, n_layers, parametrization="coupled", learn_th=True,
                 solver="gradient_decent", max_iter=100, per_layer='auto',
                 name="LISTA", ctx=None, verbose=1, device=None):
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
        if parametrization in ['step', 'coupled_step'] and not learn_th:
            raise ValueError("It is not possible to use parametrization "
                             "with step and learn_th=False")

        if per_layer == 'auto':
            per_layer = parametrization != "step"

        self.max_iter = max_iter
        self.solver = solver
        self._ctx = ctx
        self.device = device
        self.verbose = verbose
        self.n_layers = n_layers
        self.learn_th = learn_th
        self.per_layer = per_layer
        self.parametrization = parametrization
        self.pre_gradient_hooks = {"sym": []}

        self.D = np.array(D)
        self.D_ = check_tensor(self.D, device=device)
        self.B = D.dot(D.T)
        self.L = np.linalg.norm(self.B, ord=2)
        if self.parametrization == "alista":
            self.W = check_tensor(get_alista_weights(self.D).T, device=device)

        self.params = []

        self.init_network_parameters()

    def init_network_parameters(self, parameters_init=[]):
        super().__init__()
        n_atoms = self.D.shape[0]
        I_k = np.eye(n_atoms)

        self.params = []
        for layer in range(self.n_layers):
            if len(parameters_init) > layer:
                layer_params = parameters_init[layer]
            else:
                if self.parametrization == "step":
                    layer_params = [np.ones(1) / self.L]
                else:
                    layer_params = []
                    if self.learn_th:
                        layer_params += [np.ones(n_atoms) / self.L]
                    if self.parametrization == "lista":
                        layer_params += [I_k - self.B / self.L,
                                         self.D.T / self.L]
                    elif self.parametrization == "coupled":
                        layer_params += [self.D.T / self.L]
                    elif self.parametrization == "coupled_step":
                        layer_params = [np.ones(1) / self.L, self.D.T]
                    elif self.parametrization == "alista":
                        layer_params += [np.array(1 / self.L)]
                    elif self.parametrization == "hessian":
                        layer_params += [I_k / self.L]
                    else:
                        raise NotImplementedError()

            # transform all the parameters to learnable Tensor
            layer_params = [
                torch.nn.Parameter(check_tensor(p, device=self.device))
                for p in layer_params]

            # Retrieve parameters hooks and register them
            parameters_config = PARAMETRIZATIONS[self.parametrization]
            if not self.learn_th and self.parametrization != "step":
                parameters_config = parameters_config[1:]

            for p, (name, hooks) in zip(layer_params, parameters_config):
                self.register_parameter("layer{}-{}".format(layer, name), p)
                for h in hooks:
                    self.pre_gradient_hooks[h].append(p)

            self.params += [layer_params]

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
        for layer_params in self.params[:output_layer]:
            if self.learn_th and self.parametrization != "step":
                th = layer_params[0]
                layer_params = layer_params[1:]
            else:
                th = 1 / self.L
            if self.parametrization == "lista":
                if z_hat is None:
                    z_hat = x.matmul(layer_params[1])
                else:
                    z_hat = z_hat.matmul(layer_params[0]) \
                        + x.matmul(layer_params[1])
            elif "coupled" in self.parametrization:
                W = layer_params[0]
                if self.parametrization == "coupled_step":
                    W = th * W
                if z_hat is None:
                    z_hat = x.matmul(W)
                else:
                    res = z_hat.matmul(self.D_) - x
                    z_hat = z_hat - res.matmul(W)
            elif self.parametrization == "alista":
                if z_hat is None:
                    z_hat = x.matmul(self.W) * layer_params[0]
                else:
                    res = z_hat.matmul(self.D_) - x
                    z_hat = z_hat - res.matmul(self.W) * layer_params[0]
            elif self.parametrization == "hessian":
                if z_hat is None:
                    z_hat = x.matmul(self.D_.t()).matmul(layer_params[0])
                else:
                    grad = (z_hat.matmul(self.D_) - x).matmul(self.D_.t())
                    z_hat = z_hat - grad.matmul(layer_params[0])
            elif self.parametrization == "step":
                th = layer_params[0]
                if z_hat is None:
                    z_hat = layer_params[0] * x.matmul(self.D_.t())
                else:
                    res = z_hat.matmul(self.D_) - x
                    z_hat = z_hat - layer_params[0] * res.matmul(self.D_.t())
            else:
                raise NotImplementedError()

            z_hat = soft_thresholding(z_hat, lmbd * th)

        return z_hat

    def export_parameters(self):
        """Return a list with all the parameters of the network.

        This list can be used to init a new network which will have the same
        output. Usefull to save the parameters.
        """
        return [
            [p.detach().numpy() for p in layer_parameters]
            for layer_parameters in self.params
        ]

    def get_parameters(self, name):
        """Return a list with the parameter name of each layer in the network.
        """
        parametrization_config = PARAMETRIZATIONS[self.parametrization]
        parameter_names = [p_name for p_name, _ in parametrization_config]
        if name not in parameter_names:
            raise ValueError("'name' should be in {}".format(parameter_names))
        idx = parameter_names.index(name)
        return [
            layer_parameters[idx].detach().numpy()
            for layer_parameters in self.params
        ]

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
        if self.per_layer:
            layers = range(1, self.n_layers + 1)
            max_iters = np.diff(np.linspace(
                0, self.max_iter, self.n_layers + 1, dtype=int))
        else:
            layers = [self.n_layers]
            max_iters = [self.max_iter]
        for n_layer, max_iter in zip(layers, max_iters):
            lr = 1
            # max_iter = self.max_iter
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
                if len(training_loss) > 0 and training_loss[-1] <= float(loss):
                    if i + 1 == max_iter:
                        # In this case, do not perform the last step
                        lr *= 2
                    lr = self.backtrack_parameters(parameters, lr)
                    continue

                # Accepting the previous point
                training_loss.append(float(loss))

                # Next gradient iterate
                loss.backward()
                lr = self.update_parameters(parameters, lr=lr)

        self.training_loss_ = training_loss
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

        for hook, list_params in self.pre_gradient_hooks.items():
            for p in list_params:
                if p.grad is not None:
                    GRADIENT_HOOKS[hook](p)

        for p in parameters:
            if p.grad is None:
                self._saved_gradient.append(None)
                continue

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

    def score(self, x, lmbd, z0=None, output_layer=None):
        x = check_tensor(x, device=self.device)
        with torch.no_grad():
            return self.loss_fn(x, lmbd, self(x, lmbd, z0=z0,
                                output_layer=output_layer)).cpu().numpy()
