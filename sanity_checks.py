import numpy as np

from constants import FORWARD, UNROUTED
from demands_across_cuts import compute_demands_across_cuts
from residual_capacities import compute_link_loads


def check_cut_condition(n, demands_across_cuts, capacities):
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            if demands_across_cuts[i, j] > capacities[i] + capacities[j]:
                print(f"Cut condition violated for cut {(i, j)}!")
                return False
    return True


def is_complete_routing(routing):
    """

    :param n:
    :param routing:
    :return:
    """
    routing = np.fill_diagonal(routing.copy(), FORWARD)
    return np.alltrue(routing != UNROUTED)


def is_optimal_routing(n, demands, routing):
    """

    :param n:
    :param demands:
    :param routing:
    :return:
    """
    link_loads = compute_link_loads(n, routing, demands)
    max_load = np.max(link_loads)
    max_demand_across_cut = np.max(compute_demands_across_cuts(n, demands))

    return np.isclose(max_load, max_demand_across_cut / 2)
