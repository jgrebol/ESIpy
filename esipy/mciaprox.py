import numpy as np
from esipy.tools import mapping, build_connec_rest, build_connec_unrest, build_connec_no, filter_connec, wf_type, find_middle_nodes, is_closed, find_node_distances
from esipy.indicators import sequential_mci, multiprocessing_mci
from math import factorial
from multiprocessing import Pool
from functools import partial
from time import time

def aproxmci(arr, Smo, partition=None, mcialg=0, d=1, ncores=1, minlen=6, maxlen=6, rings_thres = 0.3, closed = None):
    start = time()

    if wf_type(Smo) == "rest":
        connec = build_connec_rest(Smo, rings_thres)
    elif wf_type(Smo) == "unrest":
        connec = build_connec_unrest(Smo, rings_thres)
    else:
        connec = build_connec_no(Smo, rings_thres)

    connec = filter_connec(connec)
    if not closed:
        closed = is_closed(arr, connec)

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

    if wf_type(Smo) == 'rest':
        connec = build_connec_rest(Smo, rings_thres)
    elif wf_type(Smo) == 'unrest':
        connec = build_connec_unrest(Smo, rings_thres)
    else:
        connec = build_connec_no(Smo, rings_thres)

    perms = HamiltonMCI(arr, d, mcialg, connec, closed, minlen=minlen, maxlen=maxlen)
    nperms = perms.countperms()

    if ncores == 1:
        if partition == 'mulliken' or partition == "non-symmetric":
            perms =(mapping(arr, p) for p in perms)
            val = 0.5 * sum(compute_iring(p, Smo) for p in perms)
            return val, nperms, time() - start
        else:
            perms =(mapping(arr, p) for p in perms if p[1] < p[-1])
            val = sum(compute_iring(p, Smo) for p in perms)
            return val, nperms, time() - start
    else:
        pool = Pool(processes=ncores)
        dumb = partial(compute_iring, Smo=Smo)
        chunk_size = 50000

        if partition == 'mulliken' or partition == "non-symmetric":
            iter =(mapping(arr, p) for p in perms)
            result = 0.5 * sum(pool.imap(dumb, iter, chunk_size))
        else:
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
    def __init__(self, arr, d, alg, connec=False, closed=False, maxlen=None, minlen=None):
        # User input features
        self.arr = arr
        self.d = d
        self.n = len(arr)
        self.maxlen = maxlen
        self.minlen = minlen
        self.alg = alg
        # Path finding features
        self.closed = closed
        self.connec = filter_connec(connec)
        print("Connectivity dict", self.connec)
        self.start = find_middle_nodes(self.connec)
        self.distances = find_node_distances(self.connec)
        print("Distances", self.distances)
        # Algorithm selection
        self.generator = self._select_algorithm()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)

    def _select_algorithm(self):
        if not self.closed:
            if self.alg == 1:
                return self._alg1()
            elif self.alg == 2:
                return self._alg2()
            elif self.alg == 3:
                return self._alg3()
            elif self.alg == 4:
                return self._alg4()
            else:
                raise ValueError(" | Invalid algorithm number. Choose between 1 and 4.")
        else:
            if not hasattr(self, "connec"):
                raise ValueError(" | Missing adjacency matrix.")
            if self.alg == 1:
                return self._anilat_alg1()
            elif self.alg == 2:
                return self._anilat_alg2()
            elif self.alg == 3:
                return self._anilat_alg3()
            elif self.alg == 4:
                return self._anilat_alg4()
            else:
                raise ValueError(" | Invalid algorithm number. Choose between 1 and 4.")

    def _alg1(self):
        def dfs(path):
            if len(path) == self.n:
                val = min(abs(path[0] - path[-1]), self.n - abs(path[0] - path[-1]))
                if val <= self.d and path[1] < path[-1]:
                    yield path
            for v in range(self.n):
                if v not in path:
                    val = min(abs(v - path[-1]), self.n - abs(v - path[-1]))
                    if val <= self.d:
                        yield from dfs(path + [v])

        return dfs([0])

    def _alg2(self):
        def dfs(path):
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
                        yield from dfs(path + [v])

        return dfs([0])

    def _alg3(self):
        def dfs(path):
            if len(path) == self.n:
                val = min(abs(path[0] - path[-1]), self.n - abs(path[0] - path[-1]))
                if val <= self.d and path[1] < path[-1] and val % 2 != 0:
                    yield path
            for v in range(self.n):
                if v not in path:
                    val = min(abs(v - path[-1]), self.n - abs(v - path[-1]))
                    if val <= self.d and val % 2 != 0:
                        yield from dfs(path + [v])

        return dfs([0])

    def _alg4(self):
        def dfs(path):
            if len(path) == self.n:
                val = min(abs(path[0] - path[-1]), self.n - abs(path[0] - path[-1]))
                if val <= self.d and path[1] < path[-1] and (val % 2 == 0 or val == 1):
                    yield path
            for v in range(self.n):
                if v not in path:
                    val = min(abs(v - path[-1]), self.n - abs(v - path[-1]))
                    if val <= self.d and (val % 2 == 0 or val == 1):
                        yield from dfs(path + [v])

        return dfs([0])

    def _anilat_alg1(self):
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

    def _anilat_alg2(self):
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

    def _anilat_alg3(self):
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

    def _anilat_alg4(self):
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

    def countperms(self):
        return sum(1 for _ in self._select_algorithm())
