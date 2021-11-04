def crange(start, stop, modulo):
    """
    Generator for circular range
    :param start:
    :param stop:
    :param modulo:
    :return:
    """
    assert stop <= modulo
    index = start % modulo
    while index != stop:
        yield index
        index = (index + 1) % modulo
