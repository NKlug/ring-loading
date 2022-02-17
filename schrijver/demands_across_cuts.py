from cut_utils import demand_crosses_cut
from symmetric_matrix import SymmetricMatrix


def naive_compute_demands_across_cuts(n, d):
    D = SymmetricMatrix(n, dtype=d.dtype)

    # loop over cuts
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            x = 0
            # loop over demands
            for k in range(0, n - 1):
                for l in range(k + 1, n):
                    if demand_crosses_cut((k, l), (i, j)):
                        x += d[k, l]
            D[i, j] = x

    return D
