import numpy as np

from check_cut_condition import check_cut_condition
from constants import UNROUTED
from demands_across_cuts import compute_demands_across_cuts
from generate_instance import generate_random_instance
from relaxed_ring_loading import partial_integer_routing, relaxed_ring_loading
from residual_capacities import naive_compute_residual_capacities, compute_residual_capacities, naive_compute_loads
from symmetric_matrix import SymmetricMatrix

if __name__ == '__main__':
    n = 10
    demands = generate_random_instance(n=n, max_demand=100, sparsity=0.3, integer=True, seed=0)
    # demands = SymmetricMatrix(n, np.ones((n, n), dtype=np.float32) - np.diag(np.zeros(n)))
    pi_routing, S, capacities, demands_across_cuts, tight_cuts = partial_integer_routing(n, demands)
    print(f"Unrouted demands: {S}")
    print(f"Capacities: {capacities}")
    print(f"Tight cuts: {tight_cuts}")
    cc = check_cut_condition(n, demands_across_cuts, capacities)
    print(f"Cut condition fulfilled: {cc}")
    # naive_res_capacities = naive_compute_residual_capacities(n, pi_routing, demands, capacities)
    # print(f"Naive residual capacities: {naive_res_capacities}")
    # print(f"Naive residual capacities feasible: {np.alltrue(naive_res_capacities >= 0)}")
    res_capacities = compute_residual_capacities(n, pi_routing, demands, capacities)
    print(f"Residual capacities: {res_capacities}")
    print(f"Residual capacities feasible: {np.alltrue(res_capacities >= 0)}")
    # print(
    #     f"Naive residual capacities equal to residual capacities: {np.allclose(naive_res_capacities, res_capacities)}")

    remaining_demands = demands.copy()
    remaining_demands[np.where(pi_routing != UNROUTED)] = 0
    remaining_demands_across_cuts = compute_demands_across_cuts(n, remaining_demands)
    print(
        f"Cut condition fulfilled for remaining demands: "
        f"{check_cut_condition(n, remaining_demands_across_cuts, res_capacities)}")

    routing = relaxed_ring_loading(n, demands)
    res_capacities = naive_compute_residual_capacities(n, routing, demands, capacities)
    print(f"Total residual capacities: {res_capacities}")
    pass
