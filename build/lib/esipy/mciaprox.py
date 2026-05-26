import numpy as np
from esipy.tools import mapping, build_connec_rest, build_connec_unrest, build_connec_no, filter_connec, wf_type, find_middle_nodes, find_node_distances
from esipy.indicators import sequential_mci, multiprocessing_mci
from math import factorial
from multiprocessing import Pool
from functools import partial
from time import time

def aproxmci(arr, aom, partition=None, mcialg=0, d=1, ncores=1, minlen=6, maxlen=6, rings_thres = 0.3):
    start = time()

    if mcialg == 0:
        if partition == "mulliken" or partition == "non-symmetric":
            if ncores == 1:
                val = sequential_mci(arr, aom, partition)
            else:
                val = multiprocessing_mci(arr, aom, ncores, partition)
            nperms = factorial(len(arr)-1)
            t = time() - start
            return val, nperms, t
        if ncores == 1:
            val = sequential_mci(arr, aom, partition)
        else:
            val = multiprocessing_mci(arr, aom, ncores, partition)
        nperms = factorial(len(arr)-1) * 0.5
        t = time() - start
        return val, nperms, t

    if wf_type(aom) == 'rest':
        connec = build_connec_rest(aom, rings_thres)
    elif wf_type(aom) == 'unrest':
        connec = build_connec_unrest(aom, rings_thres)
    else:
        connec = build_connec_no(aom, rings_thres)
    connec = filter_connec(connec)

    perms = HamiltonMCI(arr, d, mcialg, connec, minlen=minlen, maxlen=maxlen, partition=partition)
    nperms = perms.countperms()

    if ncores == 1:
        if partition == 'mulliken' or partition == "non-symmetric":
            perms = (mapping(arr, p) for p in perms)
            val = 0.5 * sum(compute_iring(p, aom) for p in perms)
            return val, nperms, time() - start
        else:
            perms = (mapping(arr, p) for p in perms)
            val = sum(compute_iring(p, aom) for p in perms)
            return val, nperms, time() - start
    else:
        pool = Pool(processes=ncores)
        dumb = partial(compute_iring, aom=aom)
        chunk_size = 50000

        if partition == 'mulliken' or partition == "non-symmetric":
            perms = (mapping(arr, p) for p in perms)
            result = 0.5 * sum(pool.imap(dumb, perms, chunk_size))
        else:
            perms = (mapping(arr, p) for p in perms)
            result = sum(pool.imap(dumb, perms, chunk_size))

        pool.close()
        pool.join()
        return result, nperms, time() - start

def compute_iring(arr, aom):
    product = np.identity(aom[0].shape[0])

    for i in arr:
        product = np.dot(product, aom[i - 1])

    return 2 ** (len(arr) - 1) * np.trace(product)

class HamiltonMCI:
    def __init__(self, arr, d, mcialg, connec=False, maxlen=None, minlen=None, partition=None):
        # User input features
        self.arr = arr
        self.partition = partition
        self.d = d
        self.n = len(arr)
        self.maxlen = maxlen
        self.minlen = minlen
        self.mcialg = mcialg
        # Path finding features
        self.connec = filter_connec(connec)
        self.start = find_middle_nodes(self.connec)
        self.distances = find_node_distances(self.connec)
        # Algorithm selection
        self.generator = self._select_algorithm()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)

    def _select_algorithm(self):
        if self.partition in ["mulliken", "non-symmetric"]:
            if self.mcialg == 1:
                return self._alg1_sym()
            elif self.mcialg == 2:
                return self._alg2_sym()
            elif self.mcialg == 3:
                return self._alg3_sym()
            elif self.mcialg == 4:
                return self._alg4_sym()
            else:
                raise ValueError(" | Invalid algorithm number. Choose between 1 and 4.")
        else:
            if self.mcialg == 1:
                return self._alg1()
            elif self.mcialg == 2:
                return self._alg2()
            elif self.mcialg == 3:
                return self._alg3()
            elif self.mcialg == 4:
                return self._alg4()
            else:
                raise ValueError(" | Invalid algorithm number. Choose between 1 and 4.")

    def _alg1(self):
        r = self.arr
        def dfs(path):
            if len(path) == self.n:
                val = self.distances[r[path[0]]][r[path[-1]]]
                if val <= self.d and path[1] < path[-1]:
                    yield path
            for v in range(self.n):
                if v not in path:
                    val = self.distances[r[v]][r[path[-1]]]
                    if val <= self.d:
                        yield from dfs(path + [v])

        return dfs([0])

    def _alg2(self):
        r = self.arr
        def dfs(path):
            if len(path) == self.n:
                maxval = 0
                for i in range(self.n - 1):
                    val = self.distances[r[path[i]]][r[path[i + 1]]]
                    if val > maxval:
                        maxval = val
                val = self.distances[r[path[0]]][r[path[-1]]]
                if val > maxval:
                    maxval = val
                if maxval == self.d and path[1] < path[-1]:
                    yield path
            for v in range(self.n):
                if v not in path:
                    val = self.distances[r[v]][r[path[-1]]]
                    if val <= self.d:
                        yield from dfs(path + [v])

        return dfs([0])

    def _alg3(self):
        r = self.arr
        def dfs(path):
            if len(path) == self.n:
                val = self.distances[r[path[0]]][r[path[-1]]]
                if val <= self.d and path[1] < path[-1] and val % 2 != 0:
                    yield path
            for v in range(self.n):
                if v not in path:
                    val = self.distances[r[v]][r[path[-1]]]
                    if val <= self.d and val % 2 != 0:
                        yield from dfs(path + [v])

        return dfs([0])

    def _alg4(self):
        r = self.arr
        def dfs(path):
            if len(path) == self.n:
                val = self.distances[r[path[0]]][r[path[-1]]]
                if val <= self.d and path[1] < path[-1] and val % 2 == 0:
                    yield path
            for v in range(self.n):
                if v not in path:
                    val = self.distances[r[v]][r[path[-1]]]
                    if val <= self.d and val % 2 == 0:
                        yield from dfs(path + [v])

        return dfs([0])

    def _alg1_sym(self):
        r = self.arr
        def dfs(path):
            if len(path) == self.n:
                val = self.distances[r[path[0]]][r[path[-1]]]
                if val <= self.d:
                    yield path
            for v in range(self.n):
                if v not in path:
                    val = self.distances[r[v]][r[path[-1]]]
                    if val <= self.d:
                        yield from dfs(path + [v])

        return dfs([0])

    def _alg2_sym(self):
        r = self.arr
        def dfs(path):
            if len(path) == self.n:
                maxval = 0
                for i in range(self.n - 1):
                    val = self.distances[r[path[i]]][r[path[i + 1]]]
                    if val > maxval:
                        maxval = val
                val = self.distances[r[path[0]]][r[path[-1]]]
                if val > maxval:
                    maxval = val
                if maxval == self.d:
                    yield path
            for v in range(self.n):
                if v not in path:
                    val = self.distances[r[v]][r[path[-1]]]
                    if val <= self.d:
                        yield from dfs(path + [v])

        return dfs([0])

    def _alg3_sym(self):
        r = self.arr
        def dfs(path):
            if len(path) == self.n:
                val = self.distances[r[path[0]]][r[path[-1]]]
                if val <= self.d and val % 2 != 0:
                    yield path
            for v in range(self.n):
                if v not in path:
                    val = self.distances[r[v]][r[path[-1]]]
                    if val <= self.d and val % 2 != 0:
                        yield from dfs(path + [v])

        return dfs([0])

    def _alg4_sym(self):
        r = self.arr
        def dfs(path):
            if len(path) == self.n:
                val = self.distances[r[path[0]]][r[path[-1]]]
                if val <= self.d and val % 2 == 0:
                    yield path
            for v in range(self.n):
                if v not in path:
                    val = self.distances[r[v]][r[path[-1]]]
                    if val <= self.d and val % 2 == 0:
                        yield from dfs(path + [v])

        return dfs([0])

    def countperms(self):
        return sum(1 for _ in self._select_algorithm())
