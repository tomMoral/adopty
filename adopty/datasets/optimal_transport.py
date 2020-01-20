from ..utils import check_random_state


def make_ot(n=100, m=30, p=2, random_state=None):

    rng = check_random_state(random_state)

    # Generate point cloud and probability distribution
    x = rng.randn(n, p)
    y = rng.randn(m, p)
    alpha = abs(rng.rand(n))
    beta = abs(rng.rand(m))

    # normalize the probability
    alpha /= alpha.sum()
    beta /= beta.sum()

    # Generate the cost matrix
    xmy = x[:, None] - y[None]
    C = (xmy ** 2).sum(2)

    return alpha, beta, C, x, y
