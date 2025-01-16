import numpy as np
from esipy.tools import mapping, build_connec, build_connec_no, filter_connec, wf_type, find_middle_nodes, is_onering
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

    onering = is_onering(Smo)
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

    if wf_type(Smo) == 'rest' or wf_type(Smo) == 'unrest':
        connec = build_connec(Smo)
    else:
        connec = build_connec_no(Smo)
    print(connec)
    perms = HamiltonMCI(arr, d, mcialg, connec, onering)
    nperms = perms.countperms()

    if ncores == 1:
        if partition == 'mulliken' or partition == "non-symmetric":
            val = 0.5 * sum(compute_iring(p, Smo) for p in perms)
            return val, nperms, time() - start
        else:
            perms = (x for x in perms if x[1] < x[-1])
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
    def __init__(self, arr, d, alg, connec = False, onering=False):
        self.arr = arr
        self.d = d
        self.n = len(arr)
        self.alg = alg
        self.generator = self._select_algorithm()
        self.onering = onering
        self.connec = connec
        print("pollas", self.onering)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)

    def _select_algorithm(self):
        if self.onering:
            if self.alg == 1:
                return self._alg1([0])
            elif self.alg == 2:
                return self._alg2([0])
            elif self.alg == 3:
                return self._alg3([0])
            elif self.alg == 4:
                return self._alg4([0])
            else:
                raise ValueError(" | Invalid algorithm number. Choose between 1 and 4.")
        else:
            if not self.connec:
                raise ValueError(" | Missing adjacency matrix.")
            if self.alg == 1:
                print('doing for more than one rings')
                return self._anilat_alg1([0], self.connec)

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



    def _anilat_alg1(self, path, connec):
        if len(path) == self.n:
            # Check if the last node connects back to the first node
            if connec[path[-1]][path[0]] == 1:
                yield path

        for v in range(self.n):
            if v not in path and connec[path[-1]][v] == 1:  # Check for connection
                yield from self.dfs_anilat(path + [v], connec)

    def countperms(self):
        count = 0
        for _ in self._select_algorithm():
            count += 1
        return count