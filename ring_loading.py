from relaxed_ring_loading import partial_integer_routing


def ring_loading(n, demands):
    """

    :param n:
    :param demands:
    :return:
    """
    routing, S = partial_integer_routing(n, demands)

