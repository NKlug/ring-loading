import numpy as np

from check_cut_condition import check_cut_condition
from constants import FORWARD, UNROUTED, BACKWARD
from crossing_demands import all_demands_crossing
from demands_across_cuts import compute_demands_across_cuts
from residual_capacities import naive_compute_residual_capacities
from symmetric_matrix import SymmetricMatrix
from tight_cuts import find_tight_cuts, find_all_tight_cuts
from utils import crange


def partial_integer_routing(n, demands):
    """

    :param n:
    :param demands:
    :return:
    """
    demands_across_cuts = compute_demands_across_cuts(n, demands)
    capacities = compute_capacities(n, demands_across_cuts)
    tight_cuts = find_tight_cuts(n, demands_across_cuts, capacities)

    # route all demands that are
    pi_routing = route_parallel_demands(n, tight_cuts)

    # Find indices of unrouted demands
    S = find_unrouted_demands(n, pi_routing)

    return pi_routing, S, capacities, demands_across_cuts, tight_cuts


def relaxed_ring_loading(n, demands):
    """

    :param n:
    :param demands:
    :return:
    """
    # determine partial integer routing, set of unrouted demands S and capacities
    pi_routing, S, capacities, _, _ = partial_integer_routing(n, demands)
    # compute residual capacities
    residual_capacities = naive_compute_residual_capacities(n, pi_routing, demands, capacities)
    # Route remaining demands by splitting
    routing = route_crossing_demands(n, pi_routing, S, demands, residual_capacities)

    return routing


def naive_relaxed_ring_loading(n, demands):
    pass


def route_crossing_demands(n, routing, S, demands, residual_capacities):
    """

    :param n:
    :param routing:
    :param S:
    :param demands:
    :return:
    """
    # TODO: Check conjecture: There are only crossing demands in S (and maybe adapt if untrue)
    for (i, j) in S:
        i, j = min(i, j), max(i, j)
        c_min_front = np.min(residual_capacities[i:j])
        if demands[i, j] <= c_min_front:
            routing[i, j] = FORWARD
            residual_capacities[i:j] -= demands[i, j]
        else:
            routing[i, j] = c_min_front / demands[i, j]
            residual_capacities[i:j] -= c_min_front
            residual_capacities[:i] -= demands[i, j] - c_min_front
            residual_capacities[j:] -= demands[i, j] - c_min_front
    return routing


def find_unrouted_demands(n, routing):
    S = []
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            if routing[i, j] == UNROUTED:
                S.append((i, j))
    return S


def route_parallel_demands_with_capacities(n, tight_cuts, capacities, demands):
    routing = SymmetricMatrix(n, initial_values=np.full((n, n), UNROUTED))
    # for each tight cut, route some parallel demands
    for l in range(n):
        j = tight_cuts[l]
        i, j = min(l, j), max(l, j)
        # route demands in (i, j] through the front (we have i < j)
        for k in crange(i + 2, j + 1, n):
            if routing[i + 1, k] == UNROUTED:
                capacities = decrease_capacities(n, capacities, demands[i + 1, k], i + 1, k, FORWARD)
            routing[i + 1, k] = FORWARD

        # route demands in (j, i] such that they miss the links [i, j)
        for k in crange((j + 2) % n, i + 1, n):
            start = (j + 1) % n
            route = determine_route(start, k)
            if routing[start, k] == UNROUTED:
                capacities = decrease_capacities(n, capacities, demands[start, k], start, k, route)
            routing[start, k] = route

    return routing, capacities


def determine_route(j, k):
    """
    Choose routing according to the relation of j and k. If j < k, route forward, if k < j, route backward.
    :param j:
    :param k:
    :return:
    """
    assert j != k
    if j < k:
        return FORWARD
    else:
        return BACKWARD


def decrease_capacities(n, capacities, demand, i, j, route):
    if route == FORWARD:
        i, j = min(i, j), max(i, j)
    elif route == BACKWARD:
        i, j = max(i, j), min(i, j)
    else:
        raise Exception("Bad route!")
    for k in crange(i, j, n):
        capacities[k] -= demand
        assert capacities[k] >= 0
    return capacities


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
            routing[i + 1, k] = FORWARD

        # route demands in (j, i] such that they miss the links [i, j)
        for k in crange((j + 2) % n, i + 1, n):
            start = (j + 1) % n
            routing[start, k] = determine_route(start, k)

    return routing


def compute_capacities(n, demands_across_cuts):
    c = np.zeros((n,), dtype=demands_across_cuts.dtype)
    m = np.max(demands_across_cuts)

    for i in range(n):
        max_tight_capacity = np.max(demands_across_cuts[:i, i] - c[:i], initial=0)
        max_m_capacity = np.max(demands_across_cuts[i, i + 1:] - m / 2, initial=0)
        c[i] = np.maximum(max_tight_capacity, max_m_capacity)
    return c

def cut_is_tight(ci, cj, dij):
    return np.isclose(ci + cj, dij)
