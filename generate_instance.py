import numpy as np

from symmetric_matrix import SymmetricMatrix


def generate_random_instance(n, max_demand, sparsity, integer=False, seed=None):
    """
    Generates a random instance of ring loading
    :param sparsity: probability in [0, 1] that any demand is zero
    :param max_demand: maximum possible demand, not necessarily assumed
    :param n: instance size, i.e. number of nodes
    :param integer: whether the instance should consist of integer numbers only
    :return:
    """
    assert 0 <= sparsity <= 1
    np.random.seed(seed)
    a = np.random.rand(n, n) * max_demand
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0)
    if integer:
        a = a.astype(np.int)
    return SymmetricMatrix(n, initial_values=a)

