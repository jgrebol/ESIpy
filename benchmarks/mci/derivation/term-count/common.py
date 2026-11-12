import numpy as np
from itertools import permutations

def rand_mats(n, q, r, sym, rng):
    """n low-rank q x q matrices: symmetric PSD (L L^T) or non-symmetric (L R^T)."""
    out = []
    for _ in range(n):
        L = rng.standard_normal((q, r))
        R = L if sym else rng.standard_normal((q, r))
        out.append(L @ R.T / q)
    return out

def brute(mats, adj=None):
    """sum over permutations sigma of 1..m of Tr(S_0 S_s1 ... S_sm), steps restricted to adj (incl. closing)."""
    n = len(mats); tot = 0.0; cnt = 0
    for p in permutations(range(1, n)):
        path = (0,) + p + (0,)
        if adj is not None and not all(adj[path[i], path[i + 1]] for i in range(n)):
            continue
        P = mats[0]
        for v in p:
            P = P @ mats[v]
        tot += np.trace(P); cnt += 1
    return tot, cnt

def rand_adj(n, p, rng):
    A = np.triu(rng.random((n, n)) < p, 1)
    return (A | A.T)
