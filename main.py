import numpy as np

from check_cut_condition import check_cut_condition
from generate_instance import generate_random_instance
from relaxed_ring_loading import relaxed_ring_loading
from residual_capacities import naive_compute_residual_capacities, compute_residual_capacities

if __name__ == '__main__':
    n = 100
    demands = generate_random_instance(n=n, max_demand=100, sparsity=0.3, integer=True, seed=0)
    # demands = SymmetricMatrix(n, np.ones((n, n), dtype=np.float32) - np.diag(np.zeros(n)))
    routing, S, capacities, demands_across_cuts, tight_cuts = relaxed_ring_loading(n, demands)
    print(f"Unrouted demands: {S}")
    print(f"Capacities: {capacities}")
    print(f"Tight cuts: {tight_cuts}")
    cc = check_cut_condition(n, demands_across_cuts, capacities)
    print(f"Cut condition fulfilled: {cc}")
    naive_res_capacities = naive_compute_residual_capacities(n, routing, demands, capacities)
    print(f"Naive residual capacities: {naive_res_capacities}")
    print(f"Naive residual capacities feasible: {np.alltrue(naive_res_capacities >= 0)}")
    res_capacities = compute_residual_capacities(n, routing, demands, capacities)
    print(f"Residual capacities: {res_capacities}")
    print(f"Residual capacities feasible: {np.alltrue(res_capacities >= 0)}")
    print(
        f"Naive residual capacities equal to residual capacities: {np.allclose(naive_res_capacities, res_capacities)}")

    pass
