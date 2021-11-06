import numpy as np


def find_tight_cuts(n, demands_across_cuts, capacities):
    """
    A O(n^2) algorithm for finding one tight cut for each link.
    :param n:
    :param demands_across_cuts:
    :param capacities:
    :return:
    """
    tight_cuts = np.zeros((n,), dtype=np.int)
    for i in range(n):
        # TODO: check if comparing the whole row is desired/necessary
        j = np.isclose(capacities + capacities[i], demands_across_cuts[i, :]).nonzero()[0]
        # there might be multiple tight cuts, choose any
        j = j[0]
        tight_cuts[i] = j
    return tight_cuts


def find_all_tight_cuts(n, demands_across_cuts, capacities):
    tight_cuts = [[]] * n
    for i in range(n):
        # TODO: check if comparing the whole row is desired/necessary
        j = np.isclose(capacities + capacities[i], demands_across_cuts[i, :]).nonzero()[0]
        # there might be multiple tight cuts, choose any
        tight_cuts[i] = list(j)
    return tight_cuts
