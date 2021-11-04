from generate_instance import generate_random_instance
from relaxed_ring_loading import relaxed_ring_loading
from residual_capacities import naive_compute_residual_capacities

if __name__ == '__main__':
    n = 100
    demands = generate_random_instance(n=n, max_demand=100, sparsity=0.3, integer=True, seed=0)
    # demands = SymmetricMatrix(n, np.ones((n, n), dtype=np.float32) - np.diag(np.zeros(n)))
    routing, S, capacities = relaxed_ring_loading(n, demands)
    naive_res_capacities = naive_compute_residual_capacities(n, routing, demands, capacities)
    print(naive_res_capacities)
    pass