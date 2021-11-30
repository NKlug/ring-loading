import numpy as np

from constants import UNROUTED
from demands_across_cuts import compute_demands_across_cuts
from generate_instance import generate_random_instance
from relaxed_ring_loading import partial_integer_routing, relaxed_ring_loading
from residual_capacities import compute_residual_capacities, compute_link_loads
from sanity_checks import check_cut_condition, is_complete_routing, is_optimal_routing

if __name__ == '__main__':
    n = 10
    seed = np.random.randint(0, 100000)
    # seed = 6017  # n = 10
    # seed = 5264  # n = 100
    # seed = 89608  # n = 100
    print(f'Seed: {seed}')
    demands = generate_random_instance(n=n, max_demand=100, sparsity=0.3, integer=True, seed=seed)
    # demands = SymmetricMatrix(n, np.ones((n, n), dtype=np.float32) - np.diag(np.zeros(n)))
    pi_routing, S, capacities, demands_across_cuts, tight_cuts = partial_integer_routing(n, demands)
    # print(f"Unrouted demands: {S}")
    print(f"Capacities: {capacities}")
    # print(f"Tight cuts: {tight_cuts}")
    # cc = check_cut_condition(n, demands_across_cuts, capacities)
    # print(f"Cut condition fulfilled: {cc}")
    res_capacities = compute_residual_capacities(n, pi_routing, demands, capacities)

    remaining_demands = demands.copy()
    remaining_demands[np.where(pi_routing != UNROUTED)] = 0
    remaining_demands_across_cuts = compute_demands_across_cuts(n, remaining_demands)
    print(
        f"Cut condition fulfilled for remaining demands: "
        f"{check_cut_condition(n, remaining_demands_across_cuts, res_capacities)}")

    routing = relaxed_ring_loading(n, demands)
    # print(routing)
    print(f'Link loads: {compute_link_loads(n, routing, demands)}')
    print(f'Complete Routing: {is_complete_routing(routing)}')
    print(f'Optimal Routing: {is_optimal_routing(n, demands, routing)}')

    pass
