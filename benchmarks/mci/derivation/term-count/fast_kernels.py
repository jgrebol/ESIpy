"""Reference kernels (numpy only, batched over the signs, constant memory):
  exact_krylov     : exact sum, Tr(S_0 X^m) from a block Krylov sequence X^t L0 (no q x q powers);
                     symmetric AOMs: reversal => only ceil(m/2) steps.
  restricted_fast  : restricted sum, low-rank transfer with
                       (i) signs fixed on the pivots of the parity space W  -> 2^(m-dimW) terms,
                      (ii) bipartite block structure of T                    -> each step on half-size blocks,
                     (iii) symmetric AOMs + symmetric adjacency: meet in the middle, ceil((m-1)/2) steps.
"""
import numpy as np

def factor(M, sym):
    if sym:
        w, V = np.linalg.eigh((M + M.T) / 2)
        k = w > w.max() * len(w) * np.finfo(float).eps
        L = V[:, k] * np.sqrt(w[k])
        return L, L
    U, s, Vt = np.linalg.svd(M)
    k = max(1, int((s > s[0] * len(s) * np.finfo(float).eps).sum()))
    return U[:, :k] * s[:k], Vt[:k].T

def deltas(s0, s1, m, pivots=(0,)):
    free = [i for i in range(m) if i not in pivots]
    bits = (np.arange(s0, s1)[:, None] >> np.arange(len(free))) & 1
    d = np.ones((s1 - s0, m)); d[:, free] -= 2 * bits
    return d, d.prod(axis=1)

def exact_krylov(mats, sym, batch=256):
    """sum_sigma Tr(S_0 S_s1..S_sm) = 2^-(m-1) sum_delta prod(delta) Tr(R0^T X^m L0)."""
    rest = np.array(mats[1:]); m, k = rest.shape[0], rest.shape[1]
    L0, R0 = factor(mats[0], sym)
    nterms = 1 << (m - 1); total = 0.0
    for a in range(0, nterms, batch):
        d, sign = deltas(a, min(a + batch, nterms), m)
        X = np.dot(d, rest.reshape(m, -1)).reshape(-1, k, k)
        V = np.broadcast_to(L0, (len(sign),) + L0.shape)
        if sym:   # Tr(L0^T X^m L0) = <X^j L0, X^(m-j) L0>, j = ceil(m/2)
            j = (m + 1) // 2; prev = V
            for _ in range(j):
                prev, V = V, np.matmul(X, V)
            other = V if m % 2 == 0 else prev
            total += np.dot(sign, np.einsum('bij,bij->b', V, other))
        else:
            for _ in range(m):
                V = np.matmul(X, V)
            total += np.dot(sign, np.einsum('ij,bij->b', R0, V))
    return total / nterms

def restricted_fast(mats, adj, sym, pivots=(0,), sides=None, batch=256):
    """adj: (n,n) bool. pivots: sign positions (0-based among vertices 1..m) fixed to +1 (pivots of a basis of W).
    sides: None, or bool array over vertices 1..m, True = side visited at odd steps (bipartite graphs)."""
    m = len(mats) - 1
    fs = [factor(M, sym) for M in mats]
    (L0, R0) = fs[0]; r0 = L0.shape[1]
    rk = [f[0].shape[1] for f in fs[1:]]; off = np.concatenate([[0], np.cumsum(rk)]); N = off[-1]
    L = np.hstack([f[0] for f in fs[1:]]); R = np.hstack([f[1] for f in fs[1:]])
    vert = np.repeat(np.arange(m), rk)                         # vertex of each column
    T = (R.T @ L) * adj[1:, 1:][vert][:, vert]                 # T_uv = adj[u,v] R_u^T L_v
    Y0 = (R0.T @ L) * adj[0, 1:][vert]                         # first step 0 -> v
    E = (R.T @ L0) * adj[1:, 0][vert][:, None]                 # closing v -> 0
    if sides is None:
        cols = [np.arange(N)] * (m + 1)
    else:
        odd, even = np.flatnonzero(sides[vert]), np.flatnonzero(~sides[vert])
        cols = [None] + [odd if t % 2 else even for t in range(1, m + 1)]
    nterms = 1 << (m - len(pivots)); total = 0.0
    for a in range(0, nterms, batch):
        d, sign = deltas(a, min(a + batch, nterms), m, pivots)
        dd = d[:, vert][:, None, :]                            # (b, 1, N)
        F = dd[:, :, cols[1]] * Y0[:, cols[1]]                 # F_1
        if sym:   # value = Tr(F_j D F_(m+1-j)^T), j = ceil((m+1)/2): both halves end on the same vertex
            j = (m + 2) // 2; prev = F
            for t in range(1, j):
                prev, F = F, dd[:, :, cols[t + 1]] * np.matmul(F, T[np.ix_(cols[t], cols[t + 1])])
            other = F if m % 2 == 1 else prev
            total += np.dot(sign, np.einsum('bij,bij->b', F, other * dd[:, :, cols[j]]))
        else:
            for t in range(1, m):
                F = dd[:, :, cols[t + 1]] * np.matmul(F, T[np.ix_(cols[t], cols[t + 1])])
            total += np.dot(sign, np.einsum('bij,ji->b', F, E[cols[m]]))
    return total / nterms
