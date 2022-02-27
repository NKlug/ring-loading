import collections

import numpy as np

from capacities import compute_capacities
from constants import UNROUTED, FORWARD
from proposed.contract_instance import contract_instance
from proposed.demands_across_cuts import demands_across_cuts_edge_fixed, compute_demands_across_cuts
from symmetric_matrix import SymmetricMatrix
from utils.cut_utils import demand_parallel_to_cut, find_tight_cuts, determine_route_parallel_to_cut
from utils.demand_utils import find_unrouted_demands


def ring_loading(n, demands):
    """
    Computes a minimal soulution to ring loading in O(n^2) time.
    :param n: ring size
    :param demands: SymmetricMatrix containing demands
    :return: SymmetricMatrix containing a minimal solution
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

    if len(pruned_S) == 0:
        return pi_routing

    # Route remaining demands by splitting
    routing = split_route_crossing_demands(n, pi_routing, pruned_S, demands, capacities)

    return routing


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
                routing[i, j] = determine_route_parallel_to_cut((i, j), (g, h))
                j = (j + 1) % n
            next_unrouted[i] = j

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
    # we know that all unrouted demands are crossing, i.e. |S| <= n/2
    m, T, contracted_demands, contracted_capacities = contract_instance(n, routing, S, demands, capacities)

    # compute demands across cuts and cut slacks in contracted instance in O(n^2) time
    demands_across_cuts = compute_demands_across_cuts(m, contracted_demands)
    pairwise_capacities_sum = contracted_capacities[:, None] + contracted_capacities[None, :]
    cut_slacks = pairwise_capacities_sum - demands_across_cuts

    # Intuition: min_slacks[i] corresponds to the minimal slack of the cuts originating in edge i up to m/2 + i -1
    min_slacks = np.full(m, np.inf)
    # determine initial minimal slacks in O(n^2)
    for i in range(m // 2 - 1):
        min_slacks[i] = np.min(cut_slacks[i, i + 1:m // 2])

    # sorted(S)[i] corresponds bijectively to sorted(T)[i]:
    S = sorted(S)

    # turn T into a deque for fast popleft().
    T = collections.deque(T)

    # O(n^2)
    for k in range(len(T)):
        i, j = T.popleft()

        # skip in first step - we already calculated the appropriate minimal slacks during initialization
        if k > 0:
            # compute demands across cuts where one edge in the cut is {j, j+1}, i.e. the dacs of all demands of
            # the form {x, j}, i <= x < j, of which there are n / 2 - 1
            demands_across_cuts_j = demands_across_cuts_edge_fixed(m, T, contracted_demands)  # O(n)

            slacks_j = contracted_capacities[j - 1] + contracted_capacities[i:j - 1] - demands_across_cuts_j  # O(n)

            # compute new minimal forward slacks
            for l in range(i, j - 1):
                min_slacks[l] = min(min_slacks[l], slacks_j[l - i])

        # find minimal slack along front route
        min_slack = np.min(min_slacks[i:j - 1], initial=0)  # O(n)

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
