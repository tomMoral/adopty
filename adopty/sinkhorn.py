import torch
import numpy as np

from .utils import check_tensor
from ._compat import AVAILABLE_CONTEXT


def log_dot_exp(A, b, eps):
    """Compute the dot product between exp(A) and exp(b) in log domain.
    """
    return ((-A + b.view(1, -1)) / eps).logsumexp(axis=-1)


class Sinkhorn(torch.nn.Module):
    f"""Sinkhron network for the OT problem

    Parameters
    ----------
    n_layer : int
        Number of layers in the network.
    tol : float
        Stopping criterion. When the dual variable move less than the specified
        tolerance, return from the sinkhorn algorithm. If set to None, run for
        as many iteration as requested.
    log_domain: bool (default: True)
        If set to True, run the computation in the log-domain. This is useful
        for small values of eps but might slow down the computations.
    name : str (default: Sinkhorn)
        Name of the model.
    ctx : str or None
        Context to run the network. Can be in {{{AVAILABLE_CONTEXT}}}
    verbose : int (default: 1)
        Verbosity level.
    device : str or None (default: None)
        Device on which the model is implemented. This parameter should be set
        according to the pytorch API (_eg_ 'cpu', 'gpu', 'gpu/1',..).
    """

    def __init__(self, n_layers, tol=None, log_domain=True,
                 name="Sinkhorn", ctx=None, verbose=1, device=None):
        if ctx:
            msg = "Context {} is not available on this computer."
            assert ctx in AVAILABLE_CONTEXT, msg.format(ctx)
        else:
            ctx = AVAILABLE_CONTEXT[0]

        self.name = name
        self._ctx = ctx
        self.device = device
        self.verbose = verbose

        self.tol = tol
        self.n_layers = n_layers
        self.log_domain = log_domain

        super().__init__()

    def forward(self, alpha, beta, C, eps, output_layer=None):
        n, m = C.shape

        if output_layer is None:
            output_layer = self.n_layers
        elif output_layer > self.n_layers:
            raise ValueError("Requested output from out-of-bound layer "
                             "output_layer={} (n_layers={})"
                             .format(output_layer, self.n_layers))
        if self.log_domain:
            g = torch.zeros(m, dtype=alpha.dtype, device=self.device)
        else:
            v = torch.ones(m, dtype=alpha.dtype, device=self.device)
            K = torch.exp(- C / eps)

        # Compute the following layers
        for id_layer in range(output_layer):
            v_hat = v
            if self.log_domain:
                f = eps * (torch.log(alpha) - log_dot_exp(C, g, eps))
                g = eps * (torch.log(beta) - log_dot_exp(C.t(), f, eps))
            else:
                u = alpha / torch.matmul(K, v)
                v = beta / torch.matmul(u, K)

            if (self.tol is not None and id_layer % 10 == 0
                    and torch.norm(v - v_hat) < 1e-10):
                break

        if not self.log_domain:
            f = eps * torch.log(u)
            g = eps * torch.log(v)

        return f, g

    def _loss_fn(self, f, g, alpha, beta, C, eps, primal=False):
        if primal:
            K = torch.exp(- C / eps)
            if self.log_domain:
                P = torch.exp((-C + f.view(-1, 1) + g.view(1, -1)) / eps)
                output = torch.dot(P.view(-1), C.view(-1))
            else:
                u = torch.exp(f / eps)
                v = torch.exp(g / eps)
                K = torch.exp(- C / eps)
                P = u.view(-1, 1) * K * v.view(1, -1)
                output = torch.dot(P.view(-1), C.view(-1))
        else:
            if self.log_domain:
                reg = torch.exp(((-C + f.view(-1, 1) + g.view(1, -1)) / eps
                                 ).logsumexp((0, 1)))
            else:
                K = torch.exp(- C / eps)
                reg = torch.dot(torch.exp(f / eps),
                                torch.matmul(K, torch.exp(g / eps)))

            output = torch.dot(f, alpha) + torch.dot(g, beta)
            output -= eps * reg
        return output

    def score(self, alpha, beta, C, eps, output_layer=None, primal=False):
        """Compute the loss for the network's output

        Parameters
        ----------
        alpha : ndarray, shape (n_samples_1)
            First input distribution.
        beta: ndarray, shape (n_samples_2)
            Second input distribution.
        C : ndarray, shape (n_samples_1, n_samples_2)
            Cost matrix between the samples of each distribution.
        eps : float
            Entropic regularization parameter
        output_layer : int (default: None)
            Layer to output from. It should be smaller than the number of
            layers of the network. Ifs set to None, output the network's last
            layer.
        primal : boolean (default: False)
            If set to True, output the primal loss function. Else, output the
            dual loss.

        Return
        ------
        loss : float
            Optimal transport loss between alpha, beta, for the given C and eps
        """
        alpha = check_tensor(alpha, device=self.device)
        beta = check_tensor(beta, device=self.device)
        C = check_tensor(C, device=self.device)
        with torch.no_grad():
            f, g = self(alpha, beta, C, eps, output_layer=output_layer)
            return self._loss_fn(f, g, alpha, beta, C, eps, primal=primal
                                 ).cpu().numpy()

    def compute_loss(self, alpha, beta, C, eps, primal=False):
        """Compute the loss  along the network's layers

        Parameters
        ----------
        alpha : ndarray, shape (n_samples_1)
            First input distribution.
        beta: ndarray, shape (n_samples_2)
            Second input distribution.
        C : ndarray, shape (n_samples_1, n_samples_2)
            Cost matrix between the samples of each distribution.
        eps : float
            Entropic regularization parameter
        primal : boolean (default: False)
            If set to True, output the primal loss function. Else, output the
            dual loss.
        """
        alpha = check_tensor(alpha, device=self.device)
        beta = check_tensor(beta, device=self.device)
        C = check_tensor(C, device=self.device)
        loss = []
        with torch.no_grad():
            for output_layer in range(self.n_layers):
                f, g = self(alpha, beta, C, eps, output_layer=output_layer + 1)
                loss.append(self._loss_fn(f, g, alpha, beta, C, eps,
                                          primal=primal).cpu().numpy())
        return np.array(loss)

    def gradient_beta(self, alpha, beta, C, eps, output_layer=None):
        """Compute the gradient of Sinkhorn relative to beta with autodiff."""
        alpha = check_tensor(alpha, device=self.device)
        beta = check_tensor(beta, device=self.device, requires_grad=True)
        C = check_tensor(C, device=self.device)

        f, g = self(alpha, beta, C, eps, output_layer=output_layer)
        loss = self._loss_fn(f, g, alpha, beta, C, eps, primal=False)
        loss.backward()
        return beta.grad.cpu().numpy()

    def gradient_beta_analytic(self, alpha, beta, C, eps, output_layer=None):
        """Compute the analytic gradient of Sinkhorn relative to beta."""
        alpha = check_tensor(alpha, device=self.device)
        beta = check_tensor(beta, device=self.device)
        C = check_tensor(C, device=self.device)
        with torch.no_grad():
            f, g = self(alpha, beta, C, eps, output_layer=output_layer)
        return g.cpu().numpy()

    def get_jacobian_beta(self, alpha, beta, C, eps, output_layer=None):
        """Compute the Jacobian of the scale dual variable g relative to beta.
        """
        n_features = beta.shape

        alpha = check_tensor(alpha, device=self.device)
        beta = check_tensor(beta, device=self.device, require_grad=True)
        C = check_tensor(C, device=self.device)

        # Contruct the matrix to probe the jacobian
        beta = beta.squeeze()
        beta = beta.repeat(n_features, 1)
        f, g = self(alpha, beta, C, eps, output_layer=output_layer)
        return torch.autograd.grad(
            g, beta, grad_outputs=torch.eye(n_features))[0].cpu().numpy()

    def transform(self, alpha, beta, C, eps, output_layer=None):
        """Compute the dual variables associate to the transport plan.

        The transport plan can be recovered using the formula:
            P = exp(f / eps)[:, None] * exp(-C / eps) * exp (g / eps)[None]
        """
        # Compat numpy
        alpha = check_tensor(alpha, device=self.device)
        beta = check_tensor(beta, device=self.device)
        C = check_tensor(C, device=self.device)

        with torch.no_grad():
            f, g = self(alpha, beta, C, eps, output_layer=output_layer)

        return f.cpu().numpy(), g.cpu().numpy()
