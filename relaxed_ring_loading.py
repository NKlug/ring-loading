import numpy as np

from demands_across_cuts import compute_demands_across_cuts
from symmetric_matrix import SymmetricMatrix
from utils import crange

UNROUTED = -1


def relaxed_ring_loading(n, demands):
    demands_across_cuts = compute_demands_across_cuts(n, demands)
    capacities = compute_capacities(n, demands_across_cuts)
    tight_cuts = find_tight_cuts(n, demands_across_cuts, capacities)

    # route all demands that are
    routing = route_parallel_demands(n, tight_cuts)

    # Find indices of unrouted demands
    S = find_unrouted_demands(n, routing)

    # Route remaining demands by splitting
    routing = route_crossing_demands(n, routing, S, demands)
    return routing, S


def route_crossing_demands(n, routing, S, demands):
    residual_capacities = []

    for (i, j) in S:
        c_max_front = np.min(residual_capacities[i:j])
        if demands[i, j] < c_max_front:
            routing[i, j] = 1
            residual_capacities[i:j] -= demands[i, j]
        else:
            routing[i, j] = c_max_front / demands[i, j]
            residual_capacities[i:j] -= c_max_front
            residual_capacities[:i] -= demands[i, j] - c_max_front
            residual_capacities[j:] -= demands[i, j] - c_max_front
    return routing


def find_unrouted_demands(n, routing):
    S = []
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            if routing[i, j] == UNROUTED:
                S.append((i, j))
    return S


def find_parallel_demands(S):
    parallel_demands = [[]] * len(S)
    for index, (i, j) in enumerate(S):
        for (k, l) in S:
            if (i != k or j != l) and demands_are_parallel((i, j), (k, l)):
                parallel_demands[index].append((k, l))
    return parallel_demands


def route_parallel_demands(n, tight_cuts):
    routing = SymmetricMatrix(n, initial_values=UNROUTED * np.ones((n, n)))
    # for each tight cut, route some parallel demands
    # TODO: can be further optimized: when computing tight cuts, try to get as few as possible
    # TODO: only iterate over distinct tight cuts
    for l in range(n):
        j = tight_cuts[l]
        i, j = min(l, j), max(l, j)
        # route demands in (i, j] through the front
        for k in crange(i + 2, j + 1, n):
            routing[i + 1, k] = 1

        # route demands in (j, i] through the back
        # we have to take care of some special cases, as (j+1) = n is possible
        # demands ending in (j, n-1]
        for k in crange(j + 2, n, n):
            routing[(j + 1) % n, k] = 0
        # demands ending in [0, i]
        for k in crange(max(0, (j + 1) % n), i + 1, n):
            routing[(j + 1) % n, k] = 0
    return routing


def compute_capacities(n, demands_across_cuts):
    c = np.zeros((n,), dtype=demands_across_cuts.dtype)
    m = np.max(demands_across_cuts)

    for i in range(n):
        max_tight_capacity = np.max(demands_across_cuts[:i, i] - c[:i], initial=0)
        max_m_capacity = np.max(demands_across_cuts[i, i + 1:] - m / 2, initial=0)
        c[i] = np.maximum(max_tight_capacity, max_m_capacity)
    return c


def find_tight_cuts(n, demands_across_cuts, c):
    tight_cuts = np.zeros((n,), dtype=np.int)
    for i in range(n):
        j = np.isclose(c + c[i], demands_across_cuts[i, :]).nonzero()[0]
        # there might be multiple tight cuts, choose any
        j = j[0]
        tight_cuts[i] = j
    return tight_cuts


def cut_is_tight(ci, cj, dij):
    return np.isclose(ci + cj, dij)


def demands_are_parallel(i1, i2):
    i, j = min(i1), max(i1)
    k, l = min(i2), max(i2)
    return i < j <= k < l or k < l <= i < j or i <= k < l <= j or k <= i < j <= l
