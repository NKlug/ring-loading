import numpy as np

from relaxed_ring_loading import FORWARD, BACKWARD


def compute_residual_capacities(n, routing, demands, old_capacities):
    forward_loads = np.zeros(n)
    backward_loads = np.zeros(n)

    # load on first link is the sum of all demands starting at node 0 that are routed forward
    forward_loads[0] = cond_sum(demands[0, 1:], routing[0, 1:], FORWARD)
    for k in range(1, n):
        forward_loads[k] = forward_loads[k - 1] - cond_sum(demands[:k, k], routing[:k, k], FORWARD) \
                           + cond_sum(demands[k, k + 1:], routing[k, k + 1:], FORWARD)

    # k = n, O(n^2)
    for i in range(0, n - 1):
        backward_loads[-1] += cond_sum(demands[i, i + 1:], routing[i, i + 1], BACKWARD)

    for k in range(0, n - 1):
        backward_loads[k] = backward_loads[k - 1] - cond_sum(demands[:k, k], routing[:k, k], BACKWARD) \
                            + cond_sum(demands[k, k + 1:], routing[k, k + 1:], BACKWARD)

    total_loads = forward_loads + backward_loads
    return old_capacities - total_loads


def naive_compute_residual_capacities(n, routing, demands, old_capacities):
    """
    A O(n^3) algorithm for computing the residual capacities
    :param n:
    :param routing:
    :param demands:
    :param old_capacities:
    :return:
    """
    loads = np.zeros((n, ))

    for k in range(n):
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                if demand_routed_through_link((i, j), k, routing[i, j]):
                    loads[k] += demands[i, j]
    return old_capacities - loads


def demand_routed_through_link(ij, k, route):
    i, j = min(ij), max(ij)
    return (route == FORWARD and i <= k < j) or (route == BACKWARD and (k < i or j <= k))


def cond_sum(demand_array, routing_array, c):
    assert demand_array.shape == routing_array.shape
    return np.sum(demand_array[np.where(routing_array == c)[0]])
