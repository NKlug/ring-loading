import numpy as np


class SymmetricMatrix:
    _n: int
    _values: np.array

    def __init__(self, n, initial_values_array=None, dtype=np.float):
        """

        :type dtype: datatype
        """
        self._n = n
        self._values = np.zeros((n*(n-1)//2,), dtype=dtype)
        if initial_values_array:
            self._values[:] = initial_values_array

    def _get_index(self, index_tuple):
        row, column = min(index_tuple), max(index_tuple)
        return row * (row + 1) // 2 + column

    def __setitem__(self, key, value):
        index = self._get_index(key)
        self._values[index] = value

    def __getitem__(self, key):
        index = self._get_index(key)
        return self._values[index]