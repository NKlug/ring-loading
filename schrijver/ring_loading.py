import numpy as np

from capacities import compute_capacities
from constants import FORWARD
from cut_utils import find_tight_cuts, determine_route_parallel_to_cut, demand_parallel_to_cut
from demand_utils import demands_are_parallel
from schrijver.demands_across_cuts import compute_demands_across_cuts
from symmetric_matrix import SymmetricMatrix


def ring_loading(n, demands):
    """
    Computes a minimal solution of ring loading using Schrijver et al.'s algorithm.
    Runs in O(k * n^2), where k is the number of non-zero demands.
    :param n: ring size
    :param demands: list containing non-zero demands in the form (i, j, d_{i, j})
    :return: SymmetricMatrix containing a minimal solution
    """
    demands_across_cuts = compute_demands_across_cuts(n, demands)
    capacities = compute_capacities(n, demands_across_cuts)

    pi_routing, capacities, demands = partial_integer_routing(n, demands, demands_across_cuts, capacities)
    routing = split_route_crossing_demands(n, pi_routing, demands, capacities)

    return routing


def partial_integer_routing(n, demands, demands_across_cuts, capacities):
    tight_cuts = find_tight_cuts(n, demands_across_cuts, capacities)
    routing = SymmetricMatrix(n)
    while True:
        demand1, demand2 = find_parallel_demands(demands)
        if demand1 is None:
            break
        # sort demands and indices
        i, j, d_ij = demand1
        k, l, d_kl = demand2
        i, j = min(i, j), max(i, j)
        k, l = min(k, l), max(k, l)
        if k < i or (k == i and l < j):
            i, j, k, l = k, l, i, j
        # now we have i < k

        g = find_edge_in_between((i, j), (k, l))
        h = tight_cuts[g]
        routing, capacities, demands = _route_demand_if_parallel(capacities, routing, demands, (i, j), d_ij, (g, h))
        routing, capacities, demands = _route_demand_if_parallel(capacities, routing, demands, (k, l), d_kl, (g, h))

    return routing, capacities, demands


def _route_demand_if_parallel(capacities, routing, demands, indices, value, cut):
    i, j = indices
    g, h = cut
    if demand_parallel_to_cut((i, j), (g, h)):
        demands.remove((*indices, value))
        routing[i, j] = determine_route_parallel_to_cut((i, j), (g, h))
        if routing[i, j] == FORWARD:
            capacities[i:j] -= value
        else:
            capacities[j:] -= value
            capacities[:i] -= value
    return routing, capacities, demands


def find_parallel_demands(demands):
    """
    Finds two parallel demands. Returns None if none exist.
    :param demands: list of demands
    :return: two parallel demands or None if none exist.
    """
    for i in range(len(demands)):
        for j in range(i + 1, len(demands)):
            if demands_are_parallel(demands[i][:2], demands[j][:2]):
                return demands[i], demands[j]
    return None, None


def find_edge_in_between(demand1, demand2):
    """
    Return the index of an edge in between demand1 and demand2. Assumes i <= k and (i == k implies j < l)
    :param demand1: indices of first demand
    :param demand2: indices of second demand
    :return: index of an edge in between
    """
    i, j = demand1
    k, l = demand2
    if j < k:
        return j
    else:
        return i


def split_route_crossing_demands(n, routing, demands, capacities):
    """
    A procedure to split-route unrouted demands. Runs in O(|S|*n^2) time.
    :param capacities: Remaining capacities
    :param n: ring size
    :param routing: SymmetricMatrix containing th partial integer routing
    :param S: List of unrouted demands
    :param demands: SymmetricMatrix containing demands
    :return:
    """
    for i, j, d_ij in demands:
        demands_across_cuts = compute_demands_across_cuts(n, demands)

        i, j = min(i, j), max(i, j)

        pairwise_capacities_sum = capacities[:, None] + capacities[None, :]
        cut_slacks = pairwise_capacities_sum - demands_across_cuts

        m_front = np.min(cut_slacks[i:j - 1, i + 1:j])  # this takes O(n^2)
        if d_ij <= m_front / 2:
            routing[i, j] = 1
            capacities[i:j] -= d_ij
        else:
            routing[i, j] = (m_front / 2) / d_ij
            capacities[i:j] -= m_front / 2
            capacities[:i] -= d_ij - m_front / 2
            capacities[j:] -= d_ij - m_front / 2
    return routing
