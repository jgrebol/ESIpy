import numpy as np
from esipy.tools import mapping
from esipy.indicators import sequential_mci, multiprocessing_mci
from math import factorial
from multiprocessing import Pool
from functools import partial
from time import time

def aproxmci(arr, Smo, partition=None, mcialg=0, d=1, ncores=1):
    """
    Truncates and approximates the Multicenter Index (MCI) of a ring.
    Args:
        arr: Indices of the atoms in ring connectivity.
        Smo: The Atomic Overlap Matrices (AOMs) in the MO basis.
        partition: String with the name of the partition.
        mcialg: Integer with the number of the algorithm to use.
        d: Integer with the maximum distance between vertices.
        ncores: Integer with the number of cores to use in the multiprocessing.

    Returns:
        Tuple with the value of the MCI, the number of permutations and the time taken to compute the MCI.
    """

    start = time()

    if mcialg == 0:
        if partition == "mulliken":
            if ncores == 1:
                val = sequential_mci(arr, Smo, partition)
            else:
                val = multiprocessing_mci(arr, Smo, ncores, partition)
            nperms = factorial(len(arr)-1)
            t = time() - start
            return val, nperms, t
        if ncores == 1:
            val = sequential_mci(arr, Smo, partition)
        else:
            val = multiprocessing_mci(arr, Smo, ncores, partition)
        nperms = factorial(len(arr)-1) * 0.5
        t = time() - start
        return val, nperms, t

    perms = HamiltonMCI(arr, d, mcialg)
    nperms = perms.countperms()

    if ncores == 1:
        val = sum(compute_iring(mapping(arr, p), Smo) for p in perms)
        return val, nperms, time() - start
    else:
        pool = Pool(processes=ncores)
        dumb = partial(compute_iring, Smo=Smo)
        chunk_size = 50000

        if partition == 'mulliken' or partition == "non-symmetric":
            iter =(mapping(arr, p) for p in perms)
            # We account for twice the value for symmetric AOMs
            result = 0.5 * sum(pool.imap(dumb, iter, chunk_size))
        else:  # Remove reversed permutations
            iter = (mapping(arr, x) for x in perms if x[1] < x[-1])
            result = sum(pool.imap(dumb, iter, chunk_size))

        pool.close()
        pool.join()
        return result, nperms, time() - start

def compute_iring(arr, Smo):
    product = np.identity(Smo[0].shape[0])

    for i in arr:
        product = np.dot(product, Smo[i - 1])

    return 2 ** (len(arr) - 1) * np.trace(product)


class HamiltonMCI:
    """
    HamiltonMCI class with four different algorithms for generating permutations.
    """
    def __init__(self, arr, d, alg):
        self.arr = arr
        self.d = d
        self.n = len(arr)
        self.alg = alg
        self.generator = self._select_algorithm()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)

    def _select_algorithm(self):
        if self.alg == 1:
            return self._alg1([0])
        elif self.alg == 2:
            return self._alg2([0])
        elif self.alg == 3:
            return self._alg3([0])
        elif self.alg == 4:
            return self._alg4([0])
        else:
            raise ValueError("Invalid algorithm number. Choose between 1 and 4.")

    def _alg1(self, path):
        if len(path) == self.n:
            val = min(abs(path[0] - path[-1]), self.n - abs(path[0] - path[-1]))
            if val <= self.d and path[1] < path[-1]:
                yield path

        for v in range(self.n):
            if v not in path:
                val = min(abs(v - path[-1]), self.n - abs(v - path[-1]))
                if val <= self.d:
                    yield from self._alg1(path + [v])

    def _alg2(self, path):
        if len(path) == self.n:
            maxval = 0
            for i in range(self.n - 1):
                val = min(abs(path[i] - path[i + 1]), self.n - abs(path[i] - path[i + 1]))
                if val > maxval:
                    maxval = val
            val = min(abs(path[0] - path[-1]), self.n - abs(path[0] - path[-1]))
            if val > maxval:
                maxval = val
            if maxval == self.d and path[1] < path[-1]:
                yield path

        for v in range(self.n):
            if v not in path:
                val = min(abs(v - path[-1]), self.n - abs(v - path[-1]))
                if val <= self.d:
                    yield from self._alg2(path + [v])

    def _alg3(self, path):
        if len(path) == self.n:
            val = min(abs(path[0] - path[-1]), self.n - abs(path[0] - path[-1]))
            if val <= self.d and val % 2 != 0 and path[1] < path[-1]:
                yield path

        for v in range(self.n):
            if v not in path:
                val = min(abs(v - path[-1]), self.n - abs(v - path[-1]))
                if val <= self.d and val % 2 != 0:
                    yield from self._alg3(path + [v])

    def _alg4(self, path):
        if len(path) == self.n:
            val = min(abs(path[0] - path[-1]), self.n - abs(path[0] - path[-1]))
            if val <= self.d and val % 2 == 0 and path[1] < path[-1]:
                yield path

        for v in range(self.n):
            if v not in path:
                val = min(abs(v - path[-1]), self.n - abs(v - path[-1]))
                if val <= self.d and val % 2 == 0:
                    yield from self._alg4(path + [v])

    def countperms(self):
        count = 0
        for _ in self._select_algorithm():
            count += 1
        return count
