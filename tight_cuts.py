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
        j = np.equal(capacities + capacities[i], demands_across_cuts[i, :]).nonzero()[0]
        # there might be multiple tight cuts, choose any
        j = j[0]
        tight_cuts[i] = j
    return tight_cuts


def find_all_tight_cuts(n, demands_across_cuts, capacities):
    tight_cuts = []
    for i in range(n):
        cuts = np.isclose(capacities + capacities[i], demands_across_cuts[i, :]).nonzero()[0]
        for j in cuts:
            if i < j:
                tight_cuts.append((i, j))
    return tight_cuts


def find_new_tight_cuts(n, demands_across_cuts, capacities, previous_tight_cuts=None):
    """

    :param n:
    :param demands_across_cuts:
    :param capacities:
    :param previous_tight_cuts: boolean SymmetricMatrix of cuts that were previously tight
    :return:
    """
    tight_cuts = []
    for i in range(n):
        # determine tight cuts
        cuts = np.isclose(capacities[i+1:] + capacities[i], demands_across_cuts[i, i+1:]).nonzero()[0]
        for j in cuts:
            if previous_tight_cuts is not None and not previous_tight_cuts[i, j]:
                previous_tight_cuts[i, j] = True
                tight_cuts.append((i, j))
            elif previous_tight_cuts is None:
                tight_cuts.append((i, j))
    return tight_cuts, previous_tight_cuts
