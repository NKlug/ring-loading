from symmetric_matrix import SymmetricMatrix
import numpy as np


def compute_demands_across_cuts(n, demands):
    """
    Computes the demands across all cuts using the recursion
    :param n: ring size
    :param demands: demands
    :return:
    """
    a = SymmetricMatrix(n, dtype=demands.dtype)
    D = SymmetricMatrix(n, dtype=demands.dtype)

    # k = 1
    for i in range(n - 1):
        a[i, i + 1] = demands[i, i + 1]
        D[i, i + 1] = np.sum(demands[:i + 1, i + 1]) + np.sum(demands[i + 1, i + 2:])

    # k = 2, ..., n
    for k in range(2, n):
        for i in range(0, n - k):
            j = i + k
            a[i, j] = a[i, j - 1] + demands[i, j]
            D[i, j] = D[i, i + 1] + D[i + 1, j] - 2 * a[i + 1, j]
    return D


def naive_compute_demands_across_cuts(n, d):
    D = SymmetricMatrix(n, dtype=d.dtype)

    # loop over cuts
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            x = 0
            # loop over demands
            for k in range(0, n - 1):
                for l in range(k + 1, n):
                    if crosses_cut((k, l), (i, j)):
                        x += d[k, l]
            D[i, j] = x

    return D


def crosses_cut(demand, cut):
    """

    :param demand:
    :param cut:
    :return:
    """
    k, l = demand
    i, j = cut
    return (k <= i < l <= j) or (i < k <= j < l)
