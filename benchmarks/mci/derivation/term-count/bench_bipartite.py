"""Bipartite adjacency: (a) halving with the EXISTING esipy.mci kernels (reorder + first half of the terms),
(b) alternating-side low-rank kernel. Checked against enumeration (n<=8) and timed against the current kernel."""
import itertools, time, numpy as np
from esipy import mci
from identities import brute
rng = np.random.default_rng(11)

def ring_odd(n, d):
    dist = np.array([[min(abs(i - j), n - abs(i - j)) for j in range(n)] for i in range(n)])
    return (dist <= d) & (dist % 2 == 1)

def lowrank_mats(n, q, r, sym):
    out = []
    for _ in range(n):
        L = rng.standard_normal((q, r)); Rr = L if sym else rng.standard_normal((q, r))
        out.append(L @ Rr.T / q)
    return out

def bip_order(adj):
    """Order 0, a, ..., b with a on the side opposite to 0 ... and b on the other side than a. Returns order, sides."""
    n = len(adj); side = -np.ones(n, int); side[0] = 0; stack = [0]
    while stack:
        u = stack.pop()
        for v in np.nonzero(adj[u])[0]:
            if side[v] < 0: side[v] = 1 - side[u]; stack.append(v)
            elif side[v] == side[u]: return None, None
    Q = [v for v in range(1, n) if side[v] == 1]; P = [v for v in range(1, n) if side[v] == 0]
    return [0] + Q + P, side          # first rest vertex in Q, last in P (needs P non-empty, i.e. n >= 4)

def halved_existing(mats, adj, kernel):
    order, _ = bip_order(adj); m = len(mats) - 1
    pm = [mats[i] for i in order]; pa = adj[np.ix_(order, order)]
    tot, _ = kernel((pm, pa, 0, 1 << (m - 2), 256))          # top bit = 0  <=>  delta of the last vertex = +1
    return tot / (1 << (m - 2))

def kernel_bip_lowrank(mats, adj, batch=256):
    """Alternating-side low-rank transfer: state r0 x (h r) on ONE side per step, 2^(m-2) terms."""
    order, _ = bip_order(adj); n = len(mats); m = n - 1; h = n // 2
    pm = [mats[i] for i in order]; pa = adj[np.ix_(order, order)]
    (L0, R0), L, R = mci._factors(pm)
    r0, r = L0.shape[1], L.shape[2]
    q_, p_ = slice(0, h), slice(h, m)                         # rest indices of side Q (h) and side P (h-1)
    blk = lambda U, V, A: (A[:, :, None, None] * np.einsum('uka,vkb->uvab', R[U], L[V])).transpose(0, 2, 1, 3).reshape(A.shape[0] * r, A.shape[1] * r)
    Tqp = blk(q_, p_, pa[1:h + 1, h + 1:]); Tpq = blk(p_, q_, pa[h + 1:, 1:h + 1])
    Y0 = (pa[0, 1:h + 1, None, None] * np.matmul(R0.T, L[q_])).transpose(1, 0, 2).reshape(r0, h * r)
    E = (pa[1:h + 1, 0, None, None] * np.matmul(R[q_].transpose(0, 2, 1), L0)).reshape(h * r, r0)
    total = 0.0; nterms = 1 << (m - 2)
    for a in range(0, nterms, batch):
        delta, sign = mci._glynn_deltas(a, min(a + batch, nterms), m)     # delta_1 = +1 and (top bit 0) delta_m = +1
        dq = np.repeat(delta[:, q_], r, axis=1)[:, None, :]; dp = np.repeat(delta[:, p_], r, axis=1)[:, None, :]
        Y = dq * Y0
        for _ in range(h - 1):
            Y = dp * (Y.reshape(-1, h * r) @ Tqp).reshape(-1, r0, (h - 1) * r)
            Y = dq * (Y.reshape(-1, (h - 1) * r) @ Tpq).reshape(-1, r0, h * r)
        total += np.dot(sign, np.einsum('bij,ji->b', Y, E))
    return total / nterms

if __name__ == '__main__':
    worst = [0, 0, 0]
    for sym in (False, True):
        for n in (4, 6, 8):
            for d in (3, 5):
                for trial in range(3):
                    adj = ring_odd(n, d)
                    if trial:                      # random bipartite subgraph, random labelling
                        keep = np.triu(rng.random((n, n)) < 0.85, 1); adj = adj & (keep | keep.T)
                        p = np.r_[0, 1 + rng.permutation(n - 1)]; adj = adj[np.ix_(p, p)]
                        if bip_order(adj)[0] is None or (bip_order(adj)[1] < 0).any(): continue
                    mats = lowrank_mats(n, 12, 2, sym)
                    ref, cnt = brute(mats, adj)
                    sc = max(abs(ref), 1e-9) if cnt else 1e-9
                    for i, val in enumerate((halved_existing(mats, adj, mci._kernel_glynn_walks),
                                             halved_existing(mats, adj, mci._kernel_glynn_lowrank),
                                             kernel_bip_lowrank(mats, adj))):
                        if cnt: worst[i] = max(worst[i], abs(val - ref) / sc)
                        else: assert abs(val) < 1e-9
    print("max rel err vs enumeration (n=4,6,8; nonsym + sym low rank): halved walks kernel %.1e, halved lowrank kernel %.1e, alternating kernel %.1e" % tuple(worst))
    for n, q, r in ((14, 49, 5), (16, 60, 5), (18, 60, 5)):
        mats = lowrank_mats(n, q, r, True); adj = ring_odd(n, n); m = n - 1
        t = time.perf_counter(); full, _ = mci._kernel_glynn_lowrank((mats, adj, 0, 1 << (m - 1), 256)); t0 = time.perf_counter() - t
        full /= 1 << (m - 1)
        t = time.perf_counter(); a = halved_existing(mats, adj, mci._kernel_glynn_lowrank); t1 = time.perf_counter() - t
        t = time.perf_counter(); b = kernel_bip_lowrank(mats, adj); t2 = time.perf_counter() - t
        print(f"n={n} q={q} r={r} odd-distance ring: current {t0:.2f}s | halved terms {t1:.2f}s (x{t0/t1:.1f}) | halved+alternating {t2:.2f}s (x{t0/t2:.1f}) | rel diff {abs(a-full)/abs(full):.1e} {abs(b-full)/abs(full):.1e}")
