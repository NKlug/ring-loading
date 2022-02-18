import numpy as np

from symmetric_matrix import SymmetricMatrix


def generate_random_instance(n, max_demand, sparsity, integer=False, seed=None):
    """
    Generates a random instance of ring loading of the given size n.
    :param sparsity: probability in [0, 1] that any demand is zero
    :param max_demand: maximum possible demand, not necessarily assumed
    :param n: ring size, i.e. number of nodes
    :param integer: whether the instance should consist of integer numbers only
    :return: SymmetricMatrix containing generated demands
    """
    assert 0 <= sparsity <= 1
    np.random.seed(seed)
    a = np.random.rand(n, n) * max_demand
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0)
    if integer:
        a = a.astype(np.int)
    sym_matrix = SymmetricMatrix(n, initial_values=a)
    zero_indices_1 = np.random.choice(np.arange(n, dtype=int), int(sparsity / 2 * n ** 2), replace=True)
    zero_indices_2 = np.random.choice(np.arange(n, dtype=int), int(sparsity / 2 * n ** 2), replace=True)
    sym_matrix[zero_indices_1, zero_indices_2] = 0
    return sym_matrix


def convert_demands_to_list(n, demands):
    """
    Converts the demands in a SymmetricMatrix into a list of containing tuples of type (i, j, d_{ij}).
    The list only contains the non-zero demands.
    :param n: ring size
    :param demands: SymmetricMatrix containing demands
    :return: list of non-zero demands
    """
    demands_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if demands[i, j] != 0:
                demands_list.append((i, j, demands[i, j]))
    return demands_list
