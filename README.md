# Minimal solutions to Ring Loading
This repository contains a Python implementation of the RLPW algorithm proposed in [1]
and also an implementation of the algorithm proposed by Schrijver et al. [2].
Our algorithm improves upon the runtime of the latter by a factor of K, where K is the number of non-zero demands.
The runtime of the proposed algorithm is thus O(n^2).

## Contents of this repository
The proposed algorithm is located in ``proposed/ring_loading.py``.
In can be invoked using the function ``ring_loading(n, demands)``, where `demands` is a `SymmetricMatrix` containing the demands.

The implementation of Schrijver et al.'s algorithm can be found in ``schrijver/ring_loading.py``, where it, too, can be invoked
using the `ring_loading(n, demands)` function. Note that here, `demands` is a list containing all non-zero demands.

## Experimental results

Runtime experiments can be conducted using ``runtime_test.py`` in the ``experiments`` directory.
The results given in [1] were derived from the computational results in ``result.json``, which contains tuples
of type `(seed, time_ms)`.

## References
[1] Nikolas Klug, 2022. _Computing Minimal Solutions to the Ring Loading
Problem_. Term Paper.

[2] Schrijver, A., Seymour, P. and Winkler, P., 1999. _The Ring Loading Problem_. SIAM review, 41(4), pp.777-791.