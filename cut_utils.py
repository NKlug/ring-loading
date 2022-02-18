import numpy as np

from constants import FORWARD, BACKWARD


def demand_crosses_cut(demand, cut):
    """
    Determines whether a demand crosses a cut.
    :param demand: tuple of indices of the demand
    :param cut: tuple of indices of the cut
    :return: whether the given demand crosses the given cut
    """
    i, j = min(demand), max(demand)
    g, h = min(cut), max(cut)
    return (i <= g < j <= h) or (g < i <= h < j)


def demand_parallel_to_cut(demand, cut):
    return not demand_crosses_cut(demand, cut)


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
        j = np.equal(capacities + capacities[i], demands_across_cuts[i, :]).nonzero()[0]
        # there might be multiple tight cuts, choose any
        j = j[np.random.randint(0, len(j))]
        tight_cuts[i] = j
    return tight_cuts

def determine_route_parallel_to_cut(demand, cut):
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
