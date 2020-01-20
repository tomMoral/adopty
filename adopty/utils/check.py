import torch
import numpy as np


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def check_tensor(x, device=None, requires_grad=None):
    if isinstance(x, np.ndarray) or type(x) in [int, float]:
        x = torch.tensor(x)
    if isinstance(x, torch.Tensor):
        x = x.to(device=device, dtype=torch.float64)
    if requires_grad is not None:
        x.requires_grad_(requires_grad)
    return x
