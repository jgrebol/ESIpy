import numpy as np
from esipy.tools import mapping, build_connec, build_connec_no, filter_connec, wf_type, find_middle_nodes, is_closed
from esipy.indicators import sequential_mci, multiprocessing_mci
from math import factorial
from multiprocessing import Pool
from functools import partial
from time import time

def aproxmci(arr, Smo, partition=None, mcialg=0, d=1, ncores=1):
    start = time()

    closed = is_closed(Smo)
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

    perms = HamiltonMCI(arr, d, mcialg, connec, closed)
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
    def __init__(self, arr, d, alg, connec=False, closed=False):
        self.arr = arr
        self.d = d
        self.n = len(arr)
        self.alg = alg
        self.closed = closed
        self.connec = filter_connec(connec)
        print(self.connec)
        self.start = find_middle_nodes(self.connec)
        self.generator = self._select_algorithm()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)

    def _select_algorithm(self):
        if not self.closed:
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
            if not hasattr(self, "connec"):
                raise ValueError(" | Missing adjacency matrix.")
            if self.alg == 1:
                print(' | Doing for more than one rings')
                return self._anilat_alg1()

    def _alg1(self, path):
        if len(path) == self.n:
            val = min(abs(path[0] - path[-1]), self.n - abs(path[0] - path[-1]))
            if val <= self.d and path[1] < path[-1]:
                yield path

        for v in range(self.n):
            if v not in set(path):
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
            if v not in set(path):
                val = min(abs(v - path[-1]), self.n - abs(v - path[-1]))
                if val <= self.d:
                    yield from self._alg2(path + [v])

    def _alg3(self, path):
        if len(path) == self.n:
            val = min(abs(path[0] - path[-1]), self.n - abs(path[0] - path[-1]))
            if val <= self.d and val % 2 != 0 and path[1] < path[-1]:
                yield path

        for v in range(self.n):
            if v not in set(path):
                val = min(abs(v - path[-1]), self.n - abs(v - path[-1]))
                if val <= self.d and val % 2 != 0:
                    yield from self._alg3(path + [v])

    def _alg4(self, path):
        if len(path) == self.n:
            val = min(abs(path[0] - path[-[-1]]), self.n - abs(path[0] - path[-1]))
            if val <= self.d and val % 2 == 0 and path[1] < path[-1]:
                yield path

        for v in range(self.n):
            if v not in set(path):
                val = min(abs(v - path[-1]), self.n - abs(v - path[-1]))
                if val <= self.d and val % 2 == 0:
                    yield from self._alg4(path + [v])

    def _anilat_alg1(self):
        print(self.connec)
        def dfs(path, visited):
            print(f"Current Path: {path}")
            print(f"Visited Nodes: {visited}")
            if len(path) == len(self.connec) and path[0] in self.connec.get(path[-1], []):
                yield path

            for v in self.connec.get(path[-1], []):
                if not visited[v]:
                    print(f"Exploring Neighbor: {v}")
                    visited[v] = True
                    new_path = path + [v]
                    yield from dfs(new_path, visited)
                    visited[v] = False

        for start in range(1, self.n + 1):
            visited = {node: False for node in self.connec}
            print(f"############Starting DFS from: {start} ############")
            visited[start] = True
            yield from dfs([start], visited)

    def countperms(self):
        count = 0
        for _ in self._select_algorithm():
            count += 1
        return count
