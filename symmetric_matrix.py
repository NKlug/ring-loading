import numpy as np


class SymmetricMatrix(np.ndarray):
    def __new__(cls, n, input_array=None, dtype=np.float32, *args, **kwargs):
        if input_array is None:
            obj = np.zeros((n, n), dtype=dtype)
        elif input_array.shape == 2 and np.allclose(input_array, input_array.T, rtol=1e-05, atol=1e-08):
            obj = np.asarray(input_array)
        else:
            raise Exception("Class SymmetricMatrix can only be initialized with a symmetric matrix!")
        return obj.view(cls)

    def __setitem__(self, key, value):
        i, j = key
        super(SymmetricMatrix, self).__setitem__((i, j), value)
        super(SymmetricMatrix, self).__setitem__((j, i), value)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        elif len(obj.shape) == 2 and obj.shape[0] == obj.shape[1]:
            return
        else:
            # Ideally cast to np.ndarray since obj is no square matrix anymore. But how?
            return
