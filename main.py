import time

import numpy as np

import proposed.ring_loading as proposed
import schrijver.ring_loading as schrijver
from generate_instance import generate_random_instance, convert_demands_to_list
from proposed.residual_capacities import compute_link_loads
from sanity_checks import is_complete_routing, is_optimal_routing

if __name__ == '__main__':
    n = 100
    seed = np.random.randint(0, 100000)
    print(f'Seed: {seed}')
    demands = generate_random_instance(n=n, max_demand=100, sparsity=0.1, integer=True, seed=seed)
    demands_list = convert_demands_to_list(n, demands)
    print()

    print('Schrijver algorithm:')
    time_start = time.time_ns()
    routing = schrijver.ring_loading(n, demands_list)
    print('Computation took {:.2f}ms'.format((time.time_ns() - time_start) / 1e6))
    print(f'Maximal edge load: {np.max(compute_link_loads(n, routing, demands))}')
    print(f'Routing is complete: {is_complete_routing(routing)}')
    print(f'Routing is optimal: {is_optimal_routing(n, demands, routing)}')
    print()

    print('Proposed:')
    time_start = time.time_ns()
    routing = proposed.ring_loading(n, demands)
    print('Computation took {:.2f}ms'.format((time.time_ns() - time_start) / 1e6))
    print(f'Maximal edge load: {np.max(compute_link_loads(n, routing, demands))}')
    print(f'Routing is complete: {is_complete_routing(routing)}')
    print(f'Routing is optimal: {is_optimal_routing(n, demands, routing)}')
