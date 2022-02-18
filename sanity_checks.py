import numpy as np

from constants import FORWARD, UNROUTED
from proposed.demands_across_cuts import compute_demands_across_cuts
from proposed.residual_capacities import compute_link_loads


def check_cut_condition(n, demands_across_cuts, capacities):
    """
    Checks whether the cut condition is satisfied for the given demands across cuts and capacities.
    :param n: ring size
    :param demands_across_cuts: SymmetricMatrix containing demands across cuts
    :param capacities: np.array containing capacities
    :return: whether the cut condition is satisfied
    """
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            if demands_across_cuts[i, j] > capacities[i] + capacities[j]:
                print(f"Cut condition violated for cut {(i, j)}!")
                return False
    return True


def is_complete_routing(routing):
    """
    Checks whether the given routing is complete, i.e. every demand has been routed.
    :param routing: SymmetricMatrix containing the routing
    :return: whether the routing is complete
    """
    copy = routing.copy()
    np.fill_diagonal(copy, FORWARD)
    return np.alltrue(copy != UNROUTED) and np.alltrue(copy >= 0)


def is_optimal_routing(n, demands, routing):
    """
    Checks whether the given routing is optimal, i.e. the maximal edge load is minimal.
    :param n: ring size
    :param demands: SymmetricMatrix of demands
    :param routing: SymmetricMatrix containing the routing
    :return: whether the given routing is optimal
    """
    link_loads = compute_link_loads(n, routing, demands)
    max_load = np.max(link_loads)
    max_demand_across_cut = np.max(compute_demands_across_cuts(n, demands))

    print(f'Difference to optimal: {max_load - max_demand_across_cut/2}')
    return np.isclose(max_load, max_demand_across_cut / 2)
