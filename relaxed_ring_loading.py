import numpy as np
from demands_across_cuts import compute_demands_across_cuts

from constants import FORWARD, UNROUTED, BACKWARD
from proposed.tight_cuts import find_new_tight_cuts
from residual_capacities import compute_residual_capacities
from symmetric_matrix import SymmetricMatrix
from utils import crange


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

    # O(|S|*n^2) = O(n^3)
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


def naive_split_route_crossing_demands(n, routing, S, demands, capacities):
    """
    A general procedure to split-route unrouted demands.
    Runs in O(|S|*n^2) time.
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







def cut_is_tight(ci, cj, dij):
    return np.isclose(ci + cj, dij)


