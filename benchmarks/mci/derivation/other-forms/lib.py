import numpy as np, itertools
from math import factorial

def brute(mats):
    """R = sum_sigma Tr(S_0 S_s1..S_sm) by explicit enumeration (prefix DFS)."""
    m = len(mats) - 1
    tot = 0.0
    def rec(P, used, k):
        nonlocal tot
        if k == m:
            tot += np.trace(P); return
        for v in range(1, m + 1):
            if not used >> v & 1:
                rec(P @ mats[v], used | 1 << v, k + 1)
    rec(mats[0], 0, 0)
    return tot

def deltas(s0, s1, m, dtype=float):
    bits = (np.arange(s0, s1)[:, None] >> np.arange(m - 1)) & 1
    d = np.ones((s1 - s0, m), dtype=dtype); d[:, 1:] -= 2 * bits
    return d, d.prod(axis=1)

def factor(M, sym):
    """M = L R^T with numerical rank (R = L if sym PSD)."""
    if sym:
        w, V = np.linalg.eigh((M + M.T) / 2)
        k = w > w[-1] * len(w) * np.finfo(float).eps
        L = V[:, k] * np.sqrt(w[k]); return L, L
    U, s, Vt = np.linalg.svd(M)
    k = max(1, int((s > s[0] * len(s) * np.finfo(float).eps).sum()))
    return U[:, :k] * s[:k], Vt[:k].T

def glynn_krylov(mats, sym=False, batch=256, s0=0, s1=None, terms=False):
    """Exact R by Glynn with block-Krylov bilinear form:
       Tr(S_0 X^m) = Tr(R0^T X^m L0); m block matvecs (nonsym) or ceil(m/2) (sym: (X^a L0)^T (X^b L0))."""
    m, q = len(mats) - 1, mats[0].shape[0]
    L0, R0 = factor(mats[0], sym)
    rest = np.array(mats[1:]).reshape(m, -1)
    n1 = 1 << (m - 1) if s1 is None else s1
    tot, out = 0.0, []
    for a in range(s0, n1, batch):
        d, sg = deltas(a, min(a + batch, n1), m)
        X = (d @ rest).reshape(-1, q, q)
        V = np.broadcast_to(L0, (len(sg),) + L0.shape)
        if sym:
            for _ in range(m // 2): V = X @ V
            W = X @ V if m & 1 else V
            t = np.einsum('bij,bij->b', V, W)
        else:
            for _ in range(m): V = X @ V
            t = np.einsum('ij,bij->b', R0, V)
        if terms: out.append(sg * t)
        tot += sg @ t
    if terms: return np.concatenate(out)
    return tot / (1 << (m - 1))

def glynn_gram(mats, sym=False, batch=256):
    """Same in the Gram representation (cheaper when total rank m*r < q):
       Tr(S_0 X^m) = Tr( (R0^T L) D (G D)^(m-1) (R^T L0) ), G = R^T L, X = L D R^T."""
    m = len(mats) - 1
    L0, R0 = factor(mats[0], sym)
    fs = [factor(M, sym) for M in mats[1:]]
    L = np.hstack([f[0] for f in fs]); R = np.hstack([f[1] for f in fs])
    rk = np.array([f[0].shape[1] for f in fs])
    G = R.T @ L; A = R0.T @ L; E = R.T @ L0
    tot = 0.0
    for a in range(0, 1 << (m - 1), batch):
        d, sg = deltas(a, min(a + batch, 1 << (m - 1)), m)
        D = np.repeat(d, rk, axis=1)[:, None, :]
        if sym:   # (GD)^k symmetric trick: Y_a = A D (G D)^(a-1)... use Tr(A D (GD)^(m-1) A^T)
            Y = D * A
            h = (m - 1) // 2
            for _ in range(h): Y = D * (Y @ G)
            # Y = A D (G D)^h ; need A D (GD)^(m-1) A^T = [A D (GD)^h] G^? ...
            Z = Y @ G if (m - 1) & 1 else Y            # A D (GD)^h G  (odd) or itself
            # even m-1=2h: A (DG)^h D (GD)^h A^T = sum Y * D * Y ; odd m-1=2h+1: A D (GD)^h G D (DG... ) 
            if (m - 1) & 1: t = np.einsum('bij,bij->b', Z * D, Z @ np.eye(1)) if False else np.einsum('bij,bij->b', Y @ G, Y)
            else: t = np.einsum('bij,bij->b', Y * D, Y)
        else:
            Y = D * A
            for _ in range(m - 1): Y = D * (Y @ G)
            t = np.einsum('bij,ji->b', Y, E)
        tot += sg @ t
    return tot / (1 << (m - 1))

def rand_mats(n, q, r, sym, rng, scale=None):
    out = []
    for _ in range(n):
        if sym:
            U = rng.standard_normal((q, r)); M = U @ U.T
        else:
            M = rng.standard_normal((q, r)) @ rng.standard_normal((r, q))
        out.append(M / np.linalg.norm(M, 2) * 0.6)
    return out

def partition_of_identity(N, q, r, rng):
    """N 'atoms' with symmetric PSD rank-r AOMs summing exactly to I_q (needs N*r >= q)."""
    Qm, _ = np.linalg.qr(rng.standard_normal((N * r, N * r)))
    W = Qm[:q]            # q x Nr, W W^T = I
    return [W[:, i * r:(i + 1) * r] @ W[:, i * r:(i + 1) * r].T for i in range(N)]

def huckel(n):
    """Hueckel annulene C_n: rank-1 AOMs S_i = c_i c_i^T over occupied MOs."""
    H = np.zeros((n, n))
    for i in range(n): H[i, (i + 1) % n] = H[(i + 1) % n, i] = -1
    w, C = np.linalg.eigh(H)
    C = C[:, :n // 2]
    return [np.outer(C[i], C[i]) for i in range(n)]
