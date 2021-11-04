import numpy as np


class SymmetricMatrix:
    """
    Datatype for memory-efficient storing of symmetric matrices.
    """
    _n: int
    _values: np.array
    dtype: type or np.dtype

    def __init__(self, n, initial_values=None, dtype=np.float):
        """

        :type dtype: datatype
        """
        self._n = n
        self._values = np.zeros((n, n), dtype=dtype)
        self.dtype = dtype
        if initial_values is not None:
            if len(initial_values.shape) == 2:
                assert initial_values.shape == (n, n)
                self._values[:] = initial_values
            else:
                raise Exception("Initial values has improper shape!")

    def __setitem__(self, key, value):
        self._values[key] = value
        self._values[key[::-1]] = value

    def __getitem__(self, key):
        return self._values[key]

    def __str__(self):
        return str(self._values)

    def __eq__(self, other):
        if isinstance(other, SymmetricMatrix):
            return self._values == other._values
        return False
