import numpy as np

from constants import FORWARD, BACKWARD


def compute_residual_capacities(n, routing, demands, old_capacities):
    """
    Computes residual link capacities given a (partial) routing, previous capacities and demands in O(n^2) time.
    :param n: instance size
    :param routing: (partial) routing
    :param demands: symmetric demand matrix
    :param old_capacities: capacities prior to routing
    :return: residual capacities
    """
    link_loads = compute_link_loads(n, routing, demands)
    return old_capacities - link_loads


def compute_link_loads(n, routing, demands):
    """
    Computes link loads given a (partial) routing, previous capacities and demands in O(n^2) time.
    :param n: instance size
    :param routing: (partial) routing
    :param demands: symmetric demand matrix
    :return: residual capacities
    """
    forward_loads = np.zeros(n)
    backward_loads = np.zeros(n)

    # load on first link is the weighted sum of all demands starting at node 0 that are routed forward
    forward_loads[0] = directional_weighted_sum(demands[0, 1:], routing[0, 1:], FORWARD)
    # O(n^2)
    for k in range(1, n):
        forward_loads[k] = forward_loads[k - 1] \
                           - directional_weighted_sum(demands[:k, k], routing[:k, k], FORWARD) \
                           + directional_weighted_sum(demands[k, k + 1:], routing[k, k + 1:], FORWARD)

    # k = n-1; in O(n^2)
    for i in range(0, n - 1):
        backward_loads[-1] += directional_weighted_sum(demands[i, i + 1:], routing[i, i + 1:], BACKWARD)

    # k = 0, ..., n-2; in O(n^2)
    for k in range(0, n - 1):
        backward_loads[k] = backward_loads[k - 1] \
                            - directional_weighted_sum(demands[k, k + 1:], routing[k, k + 1:], BACKWARD) \
                            + directional_weighted_sum(demands[:k, k], routing[:k, k], BACKWARD)

    return forward_loads + backward_loads


def directional_weighted_sum(demand_array, routing_array, direction):
    """
    Computes the weighted sum of the given demands and routing in the given direction.
    :param demand_array:
    :param routing_array:
    :param direction: forward or backward
    :return:
    """
    condition = np.logical_and(0 <= routing_array, routing_array <= 1)
    if direction == FORWARD:
        splits = routing_array[condition]
    elif direction == BACKWARD:
        splits = 1 - routing_array[condition]
    else:
        raise Exception(f"Unknown routing direction {direction}!")
    return np.sum(demand_array[condition] * splits)
