def demand_crosses_cut(demand, cut):
    """
    Determines whether a demand crosses a cut.
    :param demand:
    :param cut:
    :return:
    """
    i, j = min(demand), max(demand)
    g, h = min(cut), max(cut)
    return (i <= g < j <= h) or (g < i <= h < j)


def demand_parallel_to_cut(demand, cut):
    return not demand_crosses_cut(demand, cut)