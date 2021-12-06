import numpy as np

from symmetric_matrix import SymmetricMatrix


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
                    if demand_crosses_cut((k, l), (i, j)):
                        x += d[k, l]
            D[i, j] = x

    return D


def demands_across_cuts_edge_fixed(n, S, demands):
    """
    Computes the demand across cuts for all cuts of the form {k, edge} with edge - n/2 <= k < edge.
    Furthermore assumes the special case that all demands are crossing and that the given instance is contracted.
    :param S: sorted list of indices of remaining demands
    :param edge:
    :param demands:
    :param n:

    :return:
    """
    # assert that size is even - for contracted crossing instances, it always is
    assert n % 2 == 0
    demands_across_cuts = np.zeros(n // 2 - 1)

    # the first cut {edge - n / 2 + 1, edge} is crossed by all remaining demands except the one we are trying to route,
    # which is the first element of S.
    if len(S) > 1:
        dim_1_indices, dim_2_indices = zip(*S)
        demands_across_cuts[0] = np.sum(demands[dim_1_indices, dim_2_indices])
    else:
        # if there is only one demand left, the demand across the first cut (and all other cuts) is 0.
        demands_across_cuts[0] = 0

    # assumes S to be sorted
    for k in range(1, len(S)):
        demands_across_cuts[k] = demands_across_cuts[k-1] - demands[S[k-1]]

    return demands_across_cuts


def demand_crosses_cut(demand, cut):
    """
    Determines whether a demand crosses a cut.
    :param demand:
    :param cut:
    :return:
    """
    i, j = min(demand), max(demand)
    g, h = min(cut), max(cut)
    return (i <= g < j <= h) or (g < i <= h < j)
