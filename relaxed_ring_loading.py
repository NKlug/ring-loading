import numpy as np

from symmetric_matrix import SymmetricMatrix


def relaxed_ring_loading(n, demands):
    pass


def compute_capacities(n, demands_across_cuts):
    c = np.zeros((n,), dtype=demands_across_cuts.dtype)
    m = np.max(demands_across_cuts)

    for i in range(n):
        max_tight_capacity = np.max(demands_across_cuts[:i, i] - c[:i], initial=0)
        max_m_capacity = np.max(demands_across_cuts[i, i + 1:] - m / 2, initial=0)
        c[i] = np.maximum(max_tight_capacity, max_m_capacity)
    return c
