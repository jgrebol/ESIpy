"""Exact counters for the permutations kept by ESIpy approximate-MCI algorithms.
All counts are DIRECTED Hamiltonian cycles through atom 0 (= Mulliken count; halve for symmetric partitions)."""
import numpy as np
from collections import defaultdict, deque
from math import factorial

# ---- verbatim logic of esipy/mci.py distance functions -------------------------------------------
def find_node_distances(connec):
    distances = defaultdict(dict)
    for start in connec:
        queue = deque([(start, 0)]); visited = set()
        while queue:
            cur, dd = queue.popleft()
            if cur not in visited:
                visited.add(cur); distances[start][cur] = dd
                for nb in connec[cur]:
                    if nb not in visited: queue.append((nb, dd + 1))
    return distances

def find_node_distances_onlyodd(connec):
    distances = defaultdict(dict)
    for start in connec:
        queue = deque([(start, 0)]); visited = {(start, 0)}
        so, se = {}, {}
        while queue:
            cur, dd = queue.popleft()
            if dd % 2 == 1: so.setdefault(cur, dd)
            else: se.setdefault(cur, dd)
            for nb in connec.get(cur, []):
                st = (nb, (dd + 1) % 2)
                if st not in visited:
                    visited.add(st); queue.append((nb, dd + 1))
        for node in set(so) | set(se):
            distances[start][node] = so[node] if node in so else se[node]
        distances[start][start] = 0
    return distances

def ring_connec(n):
    return {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}

def dist_matrix(connec, onlyodd=False):
    arr = sorted(connec)
    fd = find_node_distances_onlyodd(connec) if onlyodd else find_node_distances(connec)
    n = len(arr)
    D = np.array([[fd[arr[i]].get(arr[j], 999999) for j in range(n)] for i in range(n)])
    np.fill_diagonal(D, 999999)
    return D

def adjs_for(connec, alg, d):
    """list of (sign, adjacency) exactly as mci_approx_glynn builds them"""
    D = dist_matrix(connec, onlyodd=alg in (4, 5))
    if alg in (3, 5): return [(1, (D <= d) & (D % 2 == 1))]
    if alg in (2, 4): return [(1, D <= d), (-1, D <= d - 1)]
    return [(1, D <= d)]

# ---- brute-force DFS mirroring _kernel_approx (directed count, no sym prune) ---------------------
def dfs_count(connec, alg, d):
    D = dist_matrix(connec, onlyodd=alg in (4, 5))
    n = len(D)
    np.fill_diagonal(D, 0)
    odd = alg in (3, 5); mx = alg in (2, 4)
    nb = [[v for v in range(n) if v != u and D[u, v] <= d and (not odd or D[u, v] % 2)] for u in range(n)]
    vis = [False] * n; vis[0] = True
    cnt = 0
    def rec(prev, depth, cm):
        nonlocal cnt
        if depth == n:
            if 0 in nb[prev] or (n == 1):
                if (not mx) or max(cm, D[prev, 0]) == d: cnt += 1
            return
        for v in nb[prev]:
            if not vis[v]:
                vis[v] = True
                rec(v, depth + 1, max(cm, D[prev, v]) if mx else 0)
                vis[v] = False
    rec(0, 1, 0)
    return cnt

# ---- Held-Karp bitmask DP: directed Hamiltonian cycles through vertex 0 --------------------------
_pc_cache = {}
def _layers(m):
    if m not in _pc_cache:
        masks = np.arange(1 << m, dtype=np.int64)
        pc = np.zeros(1 << m, dtype=np.int8)
        for b in range(m): pc += ((masks >> b) & 1).astype(np.int8)
        order = np.argsort(pc, kind='stable')
        bounds = np.searchsorted(pc[order], np.arange(m + 2))
        _pc_cache.clear()
        _pc_cache[m] = (order, bounds)
    return _pc_cache[m]

def hk_directed(adj):
    """adj: n x n boolean (may be non-symmetric). Returns # directed Hamiltonian cycles (each cycle from vertex 0)."""
    adj = np.asarray(adj, dtype=bool)
    n = len(adj)
    assert 3 <= n <= 21
    if not (adj.any(0).all() and adj.any(1).all()): return 0
    m = n - 1
    A = adj[1:, 1:].astype(np.int64)
    order, bounds = _layers(m)
    dp = np.zeros((1 << m, m), dtype=np.int64)
    for v in range(m):
        if adj[0, v + 1]: dp[1 << v, v] = 1
    for k in range(1, m):
        M = order[bounds[k]:bounds[k + 1]]
        W = dp[M] @ A
        for v in range(m):
            if not A[:, v].any(): continue
            sel = ((M >> v) & 1) == 0
            dp[M[sel] | (1 << v), v] = W[sel, v]
    return int(dp[(1 << m) - 1] @ adj[1:, 0].astype(np.int64))

def count(connec, alg, d):
    n = len(connec)
    if alg == 0: return factorial(n - 1)
    return sum(s * hk_directed(a) for s, a in adjs_for(connec, alg, d))

def circ_adj(n, offs):
    A = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for o in offs:
            if o % n: A[i, (i + o) % n] = A[i, (i - o) % n] = True
    return A
