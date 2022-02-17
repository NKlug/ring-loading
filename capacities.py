import numpy as np


def compute_capacities(n, demands_across_cuts):
    """
    Computes the capacities as given in the paper.
    :param n: ring size
    :param demands_across_cuts: SymmetricMatrix containing all demands across cuts.
    :return: np.array containing capacities
    """
    c = np.zeros((n,), dtype=np.float32)
    m = np.max(demands_across_cuts)

    for i in range(n):
        max_tight_capacity = np.max(demands_across_cuts[:i, i] - c[:i], initial=0)
        max_m_capacity = np.max(demands_across_cuts[i, i + 1:] - m / 2, initial=0)
        c[i] = np.maximum(max_tight_capacity, max_m_capacity)
    return c