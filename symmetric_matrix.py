import numpy as np


class SymmetricMatrix(np.ndarray):
    """
    Custom implementation of a Symmetric Matrix that allows all numpy operations.
    Warning: Some operations are somewhat unsafe, like writing slices crossing the main diagonal.
    """
    def __new__(cls, n, initial_values=None, dtype=np.float32, *args, **kwargs):
        if initial_values is None:
            obj = np.zeros((n, n), dtype=dtype)
        elif len(initial_values.shape) == 2 and np.allclose(initial_values, initial_values.T):
            obj = np.asarray(initial_values)
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
