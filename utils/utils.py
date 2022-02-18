def crange(start, stop, modulo):
    """
    Generator for circular range
    :param start:
    :param stop:
    :param modulo:
    :return:
    """
    # return nothing if start >= modulo
    if start >= modulo:
        return
    # get stop in bounds if necessary
    stop = stop % modulo
    index = start
    while index != stop:
        yield index
        index = (index + 1) % modulo
