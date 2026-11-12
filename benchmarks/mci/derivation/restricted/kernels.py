"""Batched kernels (same calling convention as esipy.mci kernels: sum over terms s0 <= s < s1, not normalised)."""
import numpy as np
from lib import factor_sym, bfs_dist


def _deltas(s0, s1, m, nfix=1):
    """signs with the first nfix entries held at +1 (caller orders the vertices so that those are the fixed ones)."""
    bits = (np.arange(s0, s1)[:, None] >> np.arange(m - nfix)) & 1
    delta = np.ones((s1 - s0, m)); delta[:, nfix:] -= 2 * bits
    return delta, delta.prod(axis=1)


def _sym_setup(mats, adj):
    m = len(mats) - 1
    U0, lam0 = factor_sym(mats[0]); fs = [factor_sym(M) for M in mats[1:]]
    r = max(u.shape[1] for u, _ in fs)
    U = np.zeros((m, len(mats[0]), r)); lam = np.zeros((m, r))
    for v, (u, l) in enumerate(fs): U[v, :, :u.shape[1]] = u; lam[v, :len(l)] = l
    G = np.einsum('uka,vkb->uavb', U, U)                       # blocks U_u^T U_v
    Y0 = (adj[0, 1:, None, None] * np.einsum('ki,vkb->vib', U0, U)).transpose(1, 0, 2)   # r0, m, r
    return m, r, U0, lam0, lam, G, Y0


def kernel_mitm_dense(mats, adj, s0, s1, batch=256, nfix=1):
    """P1. symmetric AOMs + symmetric adj. floor(m/2) dense products per term."""
    m, r, U0, lam0, lam, G, Y0 = _sym_setup(mats, adj)
    r0 = U0.shape[1]
    T = (adj[1:, 1:, None, None] * G.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3).reshape(m * r, m * r)
    Y0 = Y0.reshape(r0, m * r)
    b_ = m // 2; a_ = m - 1 - b_
    total = 0.0
    for a in range(s0, s1, batch):
        delta, sign = _deltas(a, min(a + batch, s1), m, nfix)
        Dt = (delta[:, :, None] * lam).reshape(-1, 1, m * r)
        Gk = np.broadcast_to(Y0, (len(sign), r0, m * r)); Ga = Gk
        for k in range(1, b_ + 1):
            Gk = ((Gk * Dt).reshape(-1, m * r) @ T).reshape(-1, r0, m * r)
            if k == a_: Ga = Gk
        total += np.dot(sign, np.einsum('i,bij,bij->b', lam0, Ga * Dt, Gk))
    return total


def kernel_mitm_sparse(mats, adj, s0, s1, batch=256, nfix=1):
    """P1+P2. as above with block-sparse T (one gemm per live vertex and step) and cone pruning."""
    m, r, U0, lam0, lam, G, Y0 = _sym_setup(mats, adj)
    r0 = U0.shape[1]
    dist = bfs_dist(adj, 0)[1:]
    b_ = m // 2; a_ = m - 1 - b_
    # plan[k] = list of (v, in-neighbours live at step k, stacked blocks (g r) x r); G_k lives on dist <= k+1
    plan = []
    for k in range(1, b_ + 1):
        st = []
        for v in range(m):
            if dist[v] > k + 1: continue
            nb = [u for u in range(m) if adj[u + 1, v + 1] and dist[u] <= k]
            if nb: st.append((v, nb, G[nb, :, v, :].reshape(len(nb) * r, r)))
        plan.append(st)
    total = 0.0
    for a in range(s0, s1, batch):
        delta, sign = _deltas(a, min(a + batch, s1), m, nfix)
        B = len(sign)
        Dt = (delta[:, None, :, None] * lam)                      # B,1,m,r
        Gk = np.broadcast_to(Y0, (B, r0, m, r)); Ga = Gk
        for k in range(1, b_ + 1):
            X = (Gk * Dt).reshape(B * r0, m, r)
            Gn = np.zeros((B * r0, m, r))
            for v, nb, Tv in plan[k - 1]:
                Gn[:, v, :] = X[:, nb, :].reshape(B * r0, -1) @ Tv
            Gk = Gn.reshape(B, r0, m, r)
            if k == a_: Ga = Gk
        total += np.dot(sign, np.einsum('i,bivr,bivr->b', lam0, Ga * Dt, Gk))
    return total


def kernel_exact_mitm(mats, s0, s1, batch=256):
    """Complete graph (one class, self loops free): Tr(S_0 X^m) = sum lam0 * (X^a U0) * (X^b U0), a+b = m,
    ceil(m/2) products (q x q)(q x r0) per term instead of ~log2(m) products (q x q)(q x q). Symmetric AOMs."""
    m = len(mats) - 1; q = len(mats[0])
    U0, lam0 = factor_sym(mats[0]); r0 = U0.shape[1]
    rest = np.array(mats[1:]).reshape(m, -1)
    b_ = (m + 1) // 2; a_ = m - b_
    total = 0.0
    for a in range(s0, s1, batch):
        delta, sign = _deltas(a, min(a + batch, s1), m)
        X = (delta @ rest).reshape(-1, q, q)
        V = np.broadcast_to(U0, (len(sign), q, r0)); Va = V
        for k in range(1, b_ + 1):
            V = X @ V
            if k == a_: Va = V
        total += np.dot(sign, np.einsum('i,bki,bki->b', lam0, Va, V))
    return total
