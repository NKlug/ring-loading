def check_cut_condition(n, demands_across_cuts, capacities):
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            if demands_across_cuts[i, j] > capacities[i] + capacities[j]:
                print(f"Cut condition violated for cut {(i, j)}!")
                return False
    return True
