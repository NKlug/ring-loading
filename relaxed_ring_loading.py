import numpy as np

from constants import FORWARD, UNROUTED, BACKWARD
from demands_across_cuts import compute_demands_across_cuts, crosses_cut
from residual_capacities import compute_residual_capacities, compute_capacities
from symmetric_matrix import SymmetricMatrix
from tight_cuts import find_tight_cuts, find_new_tight_cuts
from utils import crange


def partial_integer_routing(n, demands):
    """
    A O(n^2) algorithm for finding a partial integer routing that leaves at most n/2 demands unrouted
    :param n: instance size
    :param demands: SymmetricMatrix containing demands
    :return:
    """
    demands_across_cuts = compute_demands_across_cuts(n, demands)
    capacities = compute_capacities(n, demands_across_cuts)
    tight_cuts = find_tight_cuts(n, demands_across_cuts, capacities)

    # route some parallel demands
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

    ci_routing = pi_routing
    # ci_routing, S = complete_integer_routing(n, pi_routing, S, demands, capacities)

    # Route remaining demands by splitting
    routing = split_route_crossing_demands(n, ci_routing, S, demands, capacities)

    return routing


def complete_integer_routing(n, routing, S, demands, capacities):
    """
    Determines a complete integer routing of the demands in S in O(|S|*n^2) time
    :param n:
    :param routing:
    :param S:
    :param demands:
    :param capacities:
    :return:
    """
    residual_capacities = compute_residual_capacities(n, routing, demands, capacities)
    remaining_demands = demands.copy()
    remaining_demands[np.where(routing != UNROUTED)] = 0

    remaining_demands_across_cuts = compute_demands_across_cuts(n, remaining_demands)

    tight_cuts = SymmetricMatrix(n, initial_values=np.zeros((n, n), dtype=bool))
    new_tight_cuts, tight_cuts = find_new_tight_cuts(n, remaining_demands_across_cuts, residual_capacities, tight_cuts)

    # O(n^3)
    for k in range(len(S)):
        for (g, h) in new_tight_cuts:
            for (i, j) in S:
                if not demand_parallel_to_cut((i, j), (g, h)):
                    # route demand and remove from S and decrease capacities
                    S.remove((i, j))
                    remaining_demands[i, j] = 0
                    pass

        residual_capacities = compute_residual_capacities(n, routing, demands, capacities)
        remaining_demands_across_cuts = compute_demands_across_cuts(n, remaining_demands)

        new_tight_cuts, tight_cuts = find_new_tight_cuts(n, remaining_demands_across_cuts, residual_capacities,
                                                         tight_cuts)

        # if there a no new tight cuts, stop the loop early
        if len(new_tight_cuts) == 0:
            break

    return routing, S


def split_route_crossing_demands(n, routing, S, demands, capacities):
    """

    :param capacities:
    :param n:
    :param routing:
    :param S:
    :param demands:
    :return:
    """
    residual_capacities = compute_residual_capacities(n, routing, demands, capacities)

    remaining_demands = demands.copy()
    remaining_demands[np.where(routing != UNROUTED)] = 0

    for (i, j) in S:
        remaining_demands_across_cuts = compute_demands_across_cuts(n, remaining_demands)
        # print(f"Cut condition satisfied: {check_cut_condition(n, remaining_demands_across_cuts, residual_capacities)}")

        i, j = min(i, j), max(i, j)

        pairwise_capacities_sum = residual_capacities[:, None] + residual_capacities[None, :]
        cut_slacks = pairwise_capacities_sum - remaining_demands_across_cuts

        m_front = np.min(cut_slacks[i:j - 1, i + 1:j])  # this takes O(n^2)
        if demands[i, j] <= m_front / 2:
            routing[i, j] = 1
            residual_capacities[i:j] -= demands[i, j]
        else:
            routing[i, j] = (m_front / 2) / demands[i, j]
            residual_capacities[i:j] -= m_front / 2
            residual_capacities[:i] -= demands[i, j] - m_front / 2
            residual_capacities[j:] -= demands[i, j] - m_front / 2

        # set remaining demand to zero
        remaining_demands[i, j] = 0

    return routing


def find_unrouted_demands(n, routing):
    S = []
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            if routing[i, j] == UNROUTED:
                S.append((i, j))
    return S


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


def route_adjacent_parallel_demands(n, tight_cuts):
    """
    Old way of routing demands that are parallel to the tight cuts in O(n^2).
    However, it could not be proven that this function is correct.
    :param n:
    :param tight_cuts:
    :return:
    """
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


def route_parallel_demands(n, tight_cuts):
    """

    :param n:
    :param tight_cuts:
    :return:
    """
    routing = SymmetricMatrix(n, initial_values=UNROUTED * np.ones((n, n)))

    next_unrouted = np.roll(np.arange(0, n), -1)  # = [2, 3, 4, ..., n, 1]
    for g in range(n):
        h = tight_cuts[g]
        for i in range(n):
            j = next_unrouted[i]
            while i != j and demand_parallel_to_cut((i, j), (g, h)):
                routing[i, j] = route_parallel_to_cut((i, j), (g, h))
                j = (j + 1) % n
            next_unrouted[i] = j

    return routing


def route_parallel_to_cut(demand, cut):
    """
    Determines the route of a demand given a cut it is parallel to.
    :param demand:
    :param cut:
    :return:
    """
    i, j = min(demand), max(demand)
    g, h = min(cut), max(cut)
    if j <= g or h < i or g < i < j <= h:
        return FORWARD
    elif i <= g and h < j:
        return BACKWARD
    else:
        raise Exception('Demand not parallel to cut!')


def cut_is_tight(ci, cj, dij):
    return np.isclose(ci + cj, dij)


def demand_parallel_to_cut(demand, cut):
    return not crosses_cut(demand, cut)
