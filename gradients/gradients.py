# Louis THIRY, 4.11.2019
# reference paper for Task Driven Dictionary Learning : https://www.di.ens.fr/~fbach/taskdriven_mairal2012.pdf
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def elastic_net_loss(inputs, dictionary, alpha, lambda_1, lambda_2):
    return (0.5 * ((torch.mm(alpha, dictionary.t()) - inputs)**2).sum(dim=1) +
            lambda_1 * torch.norm(alpha, p=1, dim=1) +
            0.5 * lambda_2 * (alpha**2).sum(dim=1))

def solve_elastic_net_with_ISTA(inputs, dictionary, lambda_1, lambda_2, maxiter=1000, plot_loss_and_support_size=False):
    """
        Solve elastic net problem :
            min_alpha 0.5 * || x - D alpha ||_2**2 + lambda_1 ||alpha||_1 + 0.5 * lambda_2 ||alpha||_2**2
        using the ISTA algorithm.

        Parameters:
            input: torch tensor, size (batch_size, input_dimension)

            dictionary: torch Tensor, size (input_dimension, n_atoms)
               dictionary matrix

            lambda_1: float
                regulariation parameter in front of the l1 norm

            lambda_2: float
                regulariation parameter in front of the l2 square norm

            maxiter: int, default 1000
                maxmum number of iterations of the ISTA algorithm

        Returns:
            alpha: torch Tensor, size (batch_size, dict_size)
               the sparse code of the input batch in the dictionary D
            n_iter: int
               number of iterations of the FISTA algorithm
            diff_mean: float
               mean diff in l1 norm between the two last iterates
            diff_max: float
               max diff in l1 norm between the two last iterates
    """

    n_atoms = dictionary.size(1)

    identity = torch.eye(n_atoms, out=dictionary.new(n_atoms, n_atoms))
    DtD = torch.mm(dictionary.t(), dictionary) + lambda_2 * identity

    with torch.no_grad():
        L = torch.symeig(DtD)[0].max().item()

    if plot_loss_and_support_size:
        mean_loss, max_loss, support_size = [], []

    alpha = nn.functional.softshrink(1 / L * torch.mm(inputs, dictionary), lambda_1 / L)
    for i_iter in range(1, maxiter):
        alpha = alpha + 1 / L * (torch.mm(inputs, dictionary) - torch.mm(alpha, DtD))
        alpha = nn.functional.softshrink(alpha, lambda_1 / L)

        if plot_loss_and_support_size:
            support_size.append((alpha > 0).sum(dim=1).float().mean())
            loss = elastic_net_loss(inputs, dictionary, alpha, lambda_1, lambda_2)
            mean_loss.append(loss.mean().item())
            max_loss.append(loss.max().item())

    if plot_loss_and_support_size:
        plt.figure()
        plt.yscale('log')
        plt.plot(range(maxiter), mean_loss, label='mean loss')
        plt.plot(range(maxiter), max_loss, label='max loss')
        plt.legend()
        plt.figure()
        plt.plot(range(maxiter), support_size, label='support_size')
        plt.legend()
        plt.show()

    return alpha


def solve_TDDL_regression_autograd(inputs, targets, dictionary, classifier, lambda_1, lambda_2, lr=0.1, maxiter=200, maxiter_EN=100):
    input_dim, n_atoms = dictionary.size()

    dictionary.requires_grad = True
    classifier.requires_grad = True

    loss = nn.MSELoss()

    optimizer = torch.optim.SGD([dictionary, classifier], lr=lr, momentum=0)

    for i_iter in range(maxiter):
        alpha = solve_elastic_net_with_ISTA(inputs, dictionary, lambda_1, lambda_2, maxiter=maxiter_EN)
        y = torch.mm(alpha, classifier)

        output = loss(y, targets)

        optimizer.zero_grad()
        output.backward()
        optimizer.step()


        if i_iter+1 % 10 == 0:
            print(" - iter {}, loss : {:.7f}".format(i_iter, output.item()))

        # return the gradients to compare them
        gradients = (classifier.grad.view(-1).clone(), dictionary.grad.view(-1).clone())
        return gradients


def solve_TDDL_regression_analytic(inputs, targets, dictionary, classifier, lambda_1, lambda_2, lr=0.1, maxiter=200, maxiter_EN=100):
    b_size, input_dim, n_atoms = inputs.size(0), dictionary.size(0), dictionary.size(1)
    lambda_2_identity = (lambda_2 * torch.eye(n_atoms, out=dictionary.new(n_atoms, n_atoms))).view(1, n_atoms, n_atoms).expand(b_size, n_atoms, n_atoms)
    with torch.no_grad():
        for i_iter in range(maxiter):
            alpha = solve_elastic_net_with_ISTA(inputs, dictionary, lambda_1, lambda_2, maxiter=maxiter_EN)

            grad_classifier = 2 / alpha.size(0) * torch.mm(alpha.t(), torch.mm(alpha, classifier) - targets)

            active_set = (alpha != 0).float()
            active_set_dictionary = dictionary.view(1, input_dim, n_atoms).expand(b_size, input_dim, n_atoms) * active_set.view(b_size, 1, n_atoms)
            active_set_DtD = torch.bmm(active_set_dictionary.transpose(1, 2), active_set_dictionary) + lambda_2_identity
            active_set_DtD_inverse = torch.inverse(active_set_DtD)
            grad_alpha = active_set.view(b_size, n_atoms, 1) * 2 * torch.mm(torch.mm(alpha, classifier) - targets, classifier.t()).view(b_size, n_atoms, 1)
            beta = torch.bmm(active_set_DtD_inverse, grad_alpha)
            expanded_dictionary = dictionary.view(1, input_dim, n_atoms).expand(b_size, input_dim, n_atoms)
            grad_dictionary = (
                    - torch.bmm(expanded_dictionary, torch.bmm(beta, alpha.view(b_size, 1, n_atoms)))
                    + torch.bmm((inputs - torch.mm(alpha, dictionary.t())).view(b_size, input_dim, 1), beta.transpose(1, 2))
                ).mean(dim=0)

            loss = torch.nn.functional.mse_loss(torch.mm(alpha, classifier), targets)
            if i_iter+1 % 10 == 0:
                print(" - iter {}, loss : {:.7f}".format(i_iter, loss.item()))

            classifier = classifier - lr * grad_classifier
            dictionary = dictionary - lr * grad_dictionary
            dictionary = dictionary / torch.norm(dictionary, dim=0, p=2, keepdim=True)

            # return the gradients to compare them
            gradients = (grad_classifier.view(-1).clone(), grad_dictionary.view(-1).clone())
            return gradients



if __name__ == '__main__':
    torch.manual_seed(7)

    input_dimension = 20
    n_atoms = 40
    print("Defining random dictionary with {} atoms in dimension {}".format(n_atoms, input_dimension))
    D = torch.cuda.FloatTensor(input_dimension, n_atoms).normal_(mean=0, std=1)
    D = D / torch.norm(D, dim=0, p=2, keepdim=True)

    signal_support_size = 3
    n_samples = 4 * n_atoms
    print("n samples {}".format(n_samples))

    samples_supports = torch.randint(0, n_atoms, (n_samples, signal_support_size))
    samples = D.t()[samples_supports].sum(dim=1)
    print("samples shape {}".format(samples.shape))

    lambda_1 = 0.1
    lambda_2 = 1e-2
    print("Elastic Net Parameters: lam_1 {}, lam_2 {}".format(lambda_1, lambda_2))

    classifier = torch.cuda.FloatTensor(n_atoms, 1).fill_(0)
    classifier[:n_atoms//2,:] = 1
    classifier[n_atoms//2:,:] = -1
    classifier /= classifier.view(-1).norm(p=2)

    random_classifier = torch.zeros_like(classifier).uniform_(-1, 1)

    random_dictionary  = torch.zeros_like(D).uniform_(-1, 1)
    random_dictionary = random_dictionary / torch.norm(random_dictionary, dim=0, p=2, keepdim=True)

    samples_indices = torch.randperm(n_samples)[:n_atoms]
    random_signal_dictionary = samples[samples_indices].t()
    random_signal_dictionary /= torch.norm(random_signal_dictionary, dim=0, p=2, keepdim=True)

    n_iter_EN_list = [50, 100, 1000, 10000]

    n_iter_EN_max = max(n_iter_EN_list)
    alpha_samples = solve_elastic_net_with_ISTA(samples, D, lambda_1, lambda_2, maxiter=n_iter_EN_max)
    y_samples = torch.mm(alpha_samples, classifier)
    print("target values computed with {} iterations Elastic Net".format(n_iter_EN_max))
    print("")

    # computing reference gradients
    _, true_grad_D_randW_trueD = solve_TDDL_regression_analytic(samples, y_samples, D.clone(), random_classifier.clone(), lambda_1, lambda_2, maxiter_EN=n_iter_EN_max)
    _, true_grad_D_trueW_randD = solve_TDDL_regression_analytic(samples, y_samples, random_dictionary.clone(), classifier.clone(), lambda_1, lambda_2, maxiter_EN=n_iter_EN_max)
    _, true_grad_D_trueW_randsignalD = solve_TDDL_regression_analytic(samples, y_samples, random_signal_dictionary.clone(), classifier.clone(), lambda_1, lambda_2, maxiter_EN=n_iter_EN_max)
    _, true_grad_D_randW_randD = solve_TDDL_regression_analytic(samples, y_samples, random_dictionary.clone(), random_classifier.clone(), lambda_1, lambda_2, maxiter_EN=n_iter_EN_max)

    for n_iter_EN in n_iter_EN_list:
        print("n iterations Elastic Net : {}".format(n_iter_EN))

        # compare gradients
        print("With W = W_true, D = D_true")
        grad_W_analytic, grad_D_analytic = solve_TDDL_regression_analytic(samples, y_samples, D.clone(), classifier.clone(), lambda_1, lambda_2, maxiter_EN=n_iter_EN)
        grad_W_autograd, grad_D_autograd = solve_TDDL_regression_autograd(samples, y_samples, D.clone(), classifier.clone(), lambda_1, lambda_2, maxiter_EN=n_iter_EN, lr=0.1)

        print(" - analytic gradients norms W {}, D {}".format(grad_W_analytic.norm().item(), grad_D_analytic.norm().item()))
        print(" - autograd gradients norms W {}, D {}".format(grad_W_autograd.norm().item(), grad_D_autograd.norm().item()))

        print("Random W, D = D_true")
        grad_W_analytic, grad_D_analytic = solve_TDDL_regression_analytic(samples, y_samples, D.clone(), random_classifier.clone(), lambda_1, lambda_2, maxiter_EN=n_iter_EN)
        grad_W_autograd, grad_D_autograd = solve_TDDL_regression_autograd(samples, y_samples, D.clone(), random_classifier.clone(), lambda_1, lambda_2, maxiter_EN=n_iter_EN, lr=0.1)

        grad_W_diff = (grad_W_analytic - grad_W_autograd).norm() * 2 / (grad_W_analytic.norm() + grad_W_autograd.norm())
        grad_D_diff = (grad_D_analytic - grad_D_autograd).norm() * 2 / (grad_D_analytic.norm() + grad_D_autograd.norm())
        grad_D_analytic_true_grad_diff, grad_D_autograd_true_grad_diff = (grad_D_analytic - true_grad_D_randW_trueD).norm(), (grad_D_autograd - true_grad_D_randW_trueD).norm()
        print(f" - grad_W diff {grad_W_diff}")
        print(f" - grad_D diff {grad_D_diff}, |grad_D_analytic - true_grad|={grad_D_analytic_true_grad_diff:.3f}, |grad_D_autograd - true_grad|={grad_D_autograd_true_grad_diff:.3f}")

        print("W = W_true, Random D")
        grad_W_analytic, grad_D_analytic = solve_TDDL_regression_analytic(samples, y_samples, random_dictionary.clone(), classifier.clone(), lambda_1, lambda_2, maxiter_EN=n_iter_EN)
        grad_W_autograd, grad_D_autograd = solve_TDDL_regression_autograd(samples, y_samples, random_dictionary.clone(), classifier.clone(), lambda_1, lambda_2, maxiter_EN=n_iter_EN, lr=0.1)

        grad_W_diff = (grad_W_analytic - grad_W_autograd).norm() * 2 / (grad_W_analytic.norm() + grad_W_autograd.norm())
        grad_D_diff = (grad_D_analytic - grad_D_autograd).norm() * 2 / (grad_D_analytic.norm() + grad_D_autograd.norm())
        grad_D_analytic_true_grad_diff, grad_D_autograd_true_grad_diff = (grad_D_analytic - true_grad_D_trueW_randD).norm(), (grad_D_autograd - true_grad_D_trueW_randD).norm()
        print(f" - grad_W diff {grad_W_diff}")
        print(f" - grad_D diff {grad_D_diff}, |grad_D_analytic - true_grad|={grad_D_analytic_true_grad_diff:.3f}, |grad_D_autograd - true_grad|={grad_D_autograd_true_grad_diff:.3f}")

        print("W = W_true, D set of randomly selected samples")
        grad_W_analytic, grad_D_analytic = solve_TDDL_regression_analytic(samples, y_samples, random_signal_dictionary.clone(), classifier.clone(), lambda_1, lambda_2, maxiter_EN=n_iter_EN)
        grad_W_autograd, grad_D_autograd = solve_TDDL_regression_autograd(samples, y_samples, random_signal_dictionary.clone(), classifier.clone(), lambda_1, lambda_2, maxiter_EN=n_iter_EN, lr=0.1)

        grad_W_diff = (grad_W_analytic - grad_W_autograd).norm() * 2 / (grad_W_analytic.norm() + grad_W_autograd.norm())
        grad_D_diff = (grad_D_analytic - grad_D_autograd).norm() * 2 / (grad_D_analytic.norm() + grad_D_autograd.norm())
        grad_D_analytic_true_grad_diff, grad_D_autograd_true_grad_diff = (grad_D_analytic - true_grad_D_trueW_randsignalD).norm(), (grad_D_autograd - true_grad_D_trueW_randsignalD).norm()
        print(f" - grad_W diff {grad_W_diff}")
        print(f" - grad_D diff {grad_D_diff}, |grad_D_analytic - true_grad|={grad_D_analytic_true_grad_diff:.3f}, |grad_D_autograd - true_grad|={grad_D_autograd_true_grad_diff:.3f}")

        print("Random W,  random D")
        grad_W_analytic, grad_D_analytic = solve_TDDL_regression_analytic(samples, y_samples, random_dictionary.clone(), random_classifier.clone(), lambda_1, lambda_2, maxiter_EN=n_iter_EN)
        grad_W_autograd, grad_D_autograd = solve_TDDL_regression_autograd(samples, y_samples, random_dictionary.clone(), random_classifier.clone(), lambda_1, lambda_2, maxiter_EN=n_iter_EN, lr=0.1)

        grad_W_diff = (grad_W_analytic - grad_W_autograd).norm() * 2 / (grad_W_analytic.norm() + grad_W_autograd.norm())
        grad_D_diff = (grad_D_analytic - grad_D_autograd).norm() * 2 / (grad_D_analytic.norm() + grad_D_autograd.norm())
        grad_D_analytic_true_grad_diff, grad_D_autograd_true_grad_diff = (grad_D_analytic - true_grad_D_randW_randD).norm(), (grad_D_autograd - true_grad_D_randW_randD).norm()
        print(f" - grad_W diff {grad_W_diff}")
        print(f" - grad_D diff {grad_D_diff}, |grad_D_analytic - true_grad|={grad_D_analytic_true_grad_diff:.3f}, |grad_D_autograd - true_grad|={grad_D_autograd_true_grad_diff:.3f}")

        print('------------')
        print('')