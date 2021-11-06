def find_parallel_demands(S):
    parallel_demands = [[]] * len(S)
    for index, (i, j) in enumerate(S):
        for (k, l) in S:
            if (i != k or j != l) and demands_are_parallel((i, j), (k, l)):
                parallel_demands[index].append((k, l))
    return parallel_demands


def all_demands_crossing(S):
    parallel_demands = find_parallel_demands(S)
    for p_demands in parallel_demands:
        if len(p_demands) > 0:
            return False
    return True


def demands_are_parallel(i1, i2):
    i, j = min(i1), max(i1)
    k, l = min(i2), max(i2)
    return i < j <= k < l or k < l <= i < j or i <= k < l <= j or k <= i < j <= l
