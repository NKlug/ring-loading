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


# TODO: This might produce numerical errors. This has to be determined.
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

    # k = n-1, O(n^2)
    for i in range(0, n - 1):
        backward_loads[-1] += directional_weighted_sum(demands[i, i + 1:], routing[i, i + 1:], BACKWARD)

    # k = 0, ..., n-2, O(n^2)
    for k in range(0, n - 1):
        backward_loads[k] = backward_loads[k - 1] \
                            - directional_weighted_sum(demands[k, k + 1:], routing[k, k + 1:], BACKWARD) \
                            + directional_weighted_sum(demands[:k, k], routing[:k, k], BACKWARD)

    return forward_loads + backward_loads


def naive_compute_residual_capacities(n, routing, demands, prior_capacities):
    """
    A O(n^3) algorithm for computing the residual capacities
    :param n:
    :param routing:
    :param demands:
    :param prior_capacities:
    :return:
    """
    forward_loads, backward_loads = naive_compute_loads(n, routing, demands)
    return prior_capacities - (forward_loads + backward_loads)


def naive_compute_loads(n, routing, demands):
    """
    A O(n^3) algorithm for computing the residual capacities.
    :param n:
    :param routing:
    :param demands:
    :return:
    """
    forward_loads = np.zeros((n,))
    backward_loads = np.zeros((n,))

    for k in range(n):
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                if demand_routed_through_link((i, j), k, routing[i, j]):
                    if 0 <= routing[i, j] <= 1:
                        forward_loads[k] += demands[i, j] * routing[i, j]
                        backward_loads[k] += demands[i, j] * (1 - routing[i, j])

    return forward_loads, backward_loads


def demand_routed_through_link(ij, k, route):
    i, j = min(ij), max(ij)
    return (route == FORWARD and i <= k < j) or (route == BACKWARD and (k < i or j <= k))


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
