from generate_instance import generate_random_instance
from relaxed_ring_loading import relaxed_ring_loading

if __name__ == '__main__':
    n = 100
    demands = generate_random_instance(n=n, max_demand=100, sparsity=0.3, integer=True, seed=0)
    # demands = SymmetricMatrix(n, np.ones((n, n), dtype=np.float32) - np.diag(np.zeros(n)))
    routing = relaxed_ring_loading(n, demands)
    pass