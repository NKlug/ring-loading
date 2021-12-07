import numpy as np

from constants import FORWARD, UNROUTED, BACKWARD
from contract_instance import contract_instance
from crossing_demands import all_demands_crossing
from demands_across_cuts import compute_demands_across_cuts, demand_crosses_cut, demands_across_cuts_edge_fixed
from residual_capacities import compute_residual_capacities, compute_capacities
from sanity_checks import check_cut_condition
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

    # remove zero demands from S and route w.l.o.g forward
    pruned_S = []
    for i, j in S:
        if demands[i, j] > 0:
            pruned_S.append((i, j))
        else:
            pi_routing[i, j] = FORWARD

    # Route remaining demands by splitting
    routing = split_route_crossing_demands(n, pi_routing, pruned_S, demands, capacities)

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


def naive_split_route_crossing_demands(n, routing, S, demands, capacities):
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


def split_route_crossing_demands(n, routing, S, demands, capacities):
    """
    Splits the <= n/2 non-zero demands that remain unrouted after a partial integer routing in O(n^2) time.
    :param n: instance size
    :param routing: partial integer routing
    :param S: list of indices of non-zero unrouted demands
    :param demands: SymmetricMatrix containing remaining demands
    :param capacities: residual capacities
    :return:
    """
    assert all_demands_crossing(S)  # check that all demands in S are mutually crossing
    # we know that all unrouted demands are crossing, i.e. |S| <= n/2
    m, T, contracted_demands, contracted_capacities = contract_instance(n, routing, S, demands, capacities)

    # compute demands across cuts and cut slacks in contracted instance in O(n^2) time
    demands_across_cuts = compute_demands_across_cuts(m, contracted_demands)
    pairwise_capacities_sum = contracted_capacities[:, None] + contracted_capacities[None, :]
    cut_slacks = pairwise_capacities_sum - demands_across_cuts
    print(f'Contracted instance feasible: {check_cut_condition(m, demands_across_cuts, contracted_capacities)}')

    # Intuition: min_slacks[i] corresponds to the minimal slack of the cut originating in edge i up to m/2 + i -1
    min_slacks = np.full(m, np.inf)
    # determine initial minimal slacks in O(n^2)
    for i in range(m // 2 - 1):
        min_slacks[i] = np.min(cut_slacks[i, i + 1:m // 2])

    # sorted(S)[i] corresponds bijectively to sorted(T)[i]:
    S = sorted(S)

    # O(n^2)
    for k in range(len(T)):
        i, j = T.pop(0)

        # skip in first step - we already calculated the appropriate minimal slacks during initialization
        if k > 0:
            # compute demands across cuts where one edge is {j, j+1}, i.e. of all demands of the form
            # {x, j}, i <= x < j, of which there are n / 2 - 1
            demands_across_cuts_j = demands_across_cuts_edge_fixed(m, T, contracted_demands)  # O(n)

            slacks_j = contracted_capacities[j-1] + contracted_capacities[i:j - 1] - demands_across_cuts_j  # O(n)

            # compute new minimal forward slacks
            for l in range(i, j - 1):
                min_slacks[l] = min(min_slacks[l], slacks_j[l - i])

        # find minimal slack along front route
        min_slack = np.min(min_slacks[i:j - 1])  # O(n)

        # route demand (i, j)
        M = min(contracted_demands[i, j], min_slack / 2)
        routing[S[k]] = M / contracted_demands[i, j]

        # decrease capacities accordingly
        contracted_capacities[i:j] -= M
        contracted_capacities[:i] -= contracted_demands[i, j] - M
        contracted_capacities[j:] -= contracted_demands[i, j] - M

        # set routed demand to zero in remaining contracted demands
        contracted_demands[i, j] = 0

        # the slacks of all cuts in [i+1, j) are decreased by 2 * M (M for each edge in the cut)
        min_slacks[i:j - 1] -= 2 * M

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
    Routes all demands that are parallel to the given tight cuts in O(n^2) time.
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
    In O(1) time.
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
    return not demand_crosses_cut(demand, cut)
