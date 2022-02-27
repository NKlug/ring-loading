import json
import numpy as np
from pprint import pprint

if __name__ == '__main__':
    with open('results.json', 'r') as f:
        results = json.load(f)

    runtime_averages = {'schrijver': {}, 'proposed': {}}

    for algo in results:
        for key, values in results[algo].items():
            result_list = np.asarray(values)
            mean_runtime = np.mean(result_list[:, 1])
            std_runtime = np.std(result_list[:, 1])
            runtime_averages[algo][key] = (mean_runtime, std_runtime)

    pprint(runtime_averages)

    for i in [0, 0.2, 0.5, 0.9]:
        for algo in ['proposed', 'schrijver']:
            for j in [10, 20, 50, 100, 200]:
                print('{:.2f}'.format(runtime_averages[algo][f'{j}_{i}'][0]), " & ", end='')
            print()
        print()
