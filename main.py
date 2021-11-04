import numpy as np

from demands_across_cuts import compute_demands_across_cuts, naive_compute_demands_across_cuts
from generate_instance import generate_random_instance
from relaxed_ring_loading import compute_capacities
from symmetric_matrix import SymmetricMatrix

if __name__ == '__main__':
    n = 100
    demands = generate_random_instance(n=n, max_demand=100, sparsity=0.3, integer=True, seed=None)
    # demands = SymmetricMatrix(n, np.ones((n, n), dtype=np.float32) - np.diag(np.zeros(n)))
    demands_across_cuts = compute_demands_across_cuts(n, demands)
    capacities = compute_capacities(n, demands_across_cuts)
    pass