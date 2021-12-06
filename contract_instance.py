import numpy as np

from constants import UNROUTED
from residual_capacities import compute_residual_capacities
from symmetric_matrix import SymmetricMatrix


def contract_instance(n, routing, S, demands, capacities):
    """
    Creates a new, contracted instance of ring loading. Expects all demands in S to be mutually crossing.
    Takes O(n^2) time.
    :param n:
    :param routing:
    :param S: list of unrouted demands which are all mutually crossing
    :param demands:
    :param capacities:
    :return:
    """
    residual_capacities = compute_residual_capacities(n, routing, demands, capacities)
    remaining_demands = demands.copy()
    remaining_demands[np.where(routing != UNROUTED)] = 0

    m = len(S)
    T = [(i, i + m) for i in range(m)]

    new_capacities = np.zeros(2 * m, dtype=np.float32)
    ordered_S = sorted(S)
    # k = [1, 2, ..., m-2]
    for k in range(m - 1):
        i, j = ordered_S[k]
        u, v = ordered_S[(k + 1) % m]
        min_capacity_front = np.min(residual_capacities[i:u], initial=np.inf)
        min_capacity_back = np.min(residual_capacities[j:v], initial=np.inf)
        # take care of special cases when u < i  or j < v (then the intervals above are empty)
        if u < i:
            min_capacity_front = min(np.min(residual_capacities[i:n], initial=np.inf),
                                     np.min(residual_capacities[0:u], initial=np.inf))
        if j < v:
            min_capacity_back = min(np.min(residual_capacities[j:n], initial=np.inf),
                                    np.min(residual_capacities[0:v], initial=np.inf))
        new_capacities[k] = min_capacity_front
        new_capacities[k + m] = min_capacity_back

    # k = m - 1
    i, j = ordered_S[m - 1]
    v, u = ordered_S[0]  # u and v are purposely switched here!
    new_capacities[m - 1] = min(np.min(residual_capacities[i:n], initial=np.inf),
                                np.min(residual_capacities[0:u], initial=np.inf))
    new_capacities[2 * m - 1] = min(np.min(residual_capacities[j:n], initial=np.inf),
                                    np.min(residual_capacities[0:v], initial=np.inf))

    new_demands = SymmetricMatrix(2 * m, initial_values=np.zeros((2 * m, 2 * m)))
    for k in range(m):
        new_demands[k, k + m] = demands[S[k]]

    return 2 * m, T, new_demands, new_capacities
