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
    sym_matrix = SymmetricMatrix(n, initial_values=a)
    zero_indices_1 = np.random.choice(np.arange(n, dtype=int), int(sparsity/2 * n**2), replace=True)
    zero_indices_2 = np.random.choice(np.arange(n, dtype=int), int(sparsity/2 * n**2), replace=True)
    sym_matrix[zero_indices_1, zero_indices_2] = 0
    return sym_matrix

