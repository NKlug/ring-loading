from cut_utils import demand_crosses_cut
from symmetric_matrix import SymmetricMatrix


def compute_demands_across_cuts(n, demands):
    """

    :param n:
    :param demands:
    :return:
    """
    D = SymmetricMatrix(n, dtype=demands.dtype)

    # loop over cuts
    for g in range(0, n - 1):
        for h in range(g + 1, n):
            demand_across_cut = 0
            # loop over demands
            for i, j, d_ij in demands:
                if demand_crosses_cut((i, j), (g, h)):
                    demand_across_cut += d_ij
            D[g, h] = demand_across_cut
    return D
