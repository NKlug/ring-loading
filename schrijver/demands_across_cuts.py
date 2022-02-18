from symmetric_matrix import SymmetricMatrix
from utils.cut_utils import demand_crosses_cut


def compute_demands_across_cuts(n, demands):
    """

    :param n:
    :param demands:
    :return:
    """
    D = SymmetricMatrix(n, dtype=type(demands[0][2]))

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
