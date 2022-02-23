import itertools
import json
import time

import numpy as np

import proposed.ring_loading as proposed
import schrijver.ring_loading as schrijver
from generate_instance import generate_random_instance, demands_to_list


def runtime_test_proposed(n, sparsity, seeds):
    results = []
    for seed in seeds:
        demands = generate_random_instance(n=n, max_demand=100, sparsity=sparsity, integer=True, seed=seed)
        time_start = time.time_ns()
        _ = proposed.ring_loading(n, demands)
        time_end = time.time_ns()
        results.append((seed, (time_end - time_start) / 1e6))

    return results


def runtime_test_schrijver(n, sparsity, seeds):
    results = []
    for seed in seeds:
        demands = generate_random_instance(n=n, max_demand=100, sparsity=sparsity, integer=True, seed=seed)
        demands = demands_to_list(n, demands)
        time_start = time.time_ns()
        _ = schrijver.ring_loading(n, demands)
        time_end = time.time_ns()
        results.append((seed, (time_end - time_start) / 1e6))

    return results


if __name__ == '__main__':

    instance_sizes = [10, 20, 50, 100, 200]
    sparsities = [0, 0.2, 0.5, 0.9]
    num_instances_proposed = [100, 100, 50, 30, 20]
    num_instances_schrijver = [50, 50, 30, 5, 2]

    results = {
        'proposed': {},
        'schrijver': {}
    }

    for i, j in itertools.product(range(len(instance_sizes)), range(len(sparsities))):
        n = instance_sizes[i]
        sparsity = sparsities[j]
        num_schrijver = num_instances_schrijver[i]
        num_proposed = num_instances_proposed[i]
        print(f'Size {n}, sparsity {sparsity}')

        seeds = np.random.default_rng().integers(0, 1e6, max(num_schrijver, num_proposed))

        proposed_results = runtime_test_proposed(n, sparsity, seeds[:num_proposed])
        print(proposed_results)
        results['proposed'][f'{n}_{sparsity}'] = proposed_results
        schrijver_results = runtime_test_schrijver(n, sparsity, seeds[:num_schrijver])
        print(schrijver_results)
        results['schrijver'][f'{n}_{sparsity}'] = schrijver_results

    with open('results.json', 'w') as f:
        json.dump(results, f)



