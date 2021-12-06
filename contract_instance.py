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
    # because all demands are mutually crossing, all elements in S[:, 0] are <= than all in S[:, 1]
    flat_S = sorted([i for demand in S for i in demand])
    # k = [1, 2, ..., m-2]
    for k in range(2 * m - 1):
        i = flat_S[k]
        j = flat_S[(k + 1) % (2*m)]
        new_capacities[k] = np.min(residual_capacities[i:j], initial=np.inf)

    # k = m - 1
    i = flat_S[-1]
    j = flat_S[0]
    new_capacities[-1] = min(np.min(residual_capacities[i:n], initial=np.inf),
                             np.min(residual_capacities[0:j], initial=np.inf))

    new_demands = SymmetricMatrix(2 * m, initial_values=np.zeros((2 * m, 2 * m)))
    for k in range(m):
        new_demands[k, k + m] = demands[S[k]]

    return 2 * m, T, new_demands, new_capacities
