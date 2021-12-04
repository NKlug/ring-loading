import numpy as np
from tqdm import tqdm

from generate_instance import generate_random_instance
from relaxed_ring_loading import partial_integer_routing


def find_many_unrouted_demands_instance(n, max_tries, max_demand, sparsity):
    """

    :param n:
    :param max_tries:
    :return:
    """

    for i in tqdm(range(max_tries)):
        seed = np.random.randint(0, 1000 * max_tries)
        demands = generate_random_instance(n=n, max_demand=max_demand, sparsity=sparsity, integer=True, seed=seed)

        pi_routing, S, _, _, _ = partial_integer_routing(n, demands)
        if len(S) > n/2:
            print(seed)
            break


if __name__ == '__main__':
    find_many_unrouted_demands_instance(10, 1000000, 100, 0)

