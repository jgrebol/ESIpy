"""Cost model in units of one block product (r0 x r)(r x r) = 2 r0 r^2 flops. No heavy work."""
import numpy as np
from lib import *

def dfs_nodes(A):
    """number of self-avoiding paths 0 -> v1 -> ... (all lengths >= 1) = block products of a low-rank DFS;
    and number of Hamiltonian cycles. Layered subset count, a few MB."""
    n = len(A); m = n - 1
    cnt = np.zeros((1 << m, m), dtype=np.int64)
    for v in range(m):
        if A[0, v + 1]: cnt[1 << v, v] = 1
    masks = np.arange(1 << m); pc = np.zeros(1 << m, int)
    for b in range(m): pc += (masks >> b) & 1
    for k in range(1, m):
        layer = masks[pc == k]
        for u in range(m):
            lu = layer[(layer >> u) & 1 == 1]
            for v in range(m):
                if A[u + 1, v + 1]:
                    sel = lu[(lu >> v) & 1 == 0]
                    cnt[sel | (1 << v), v] += cnt[sel, u]
    ham = sum(int(cnt[-1, v]) for v in range(m) if A[v + 1, 0])
    return int(cnt.sum()), ham

def blocks(A, mitm=False, prune=True):
    n = len(A); m = n - 1
    dout = bfs_dist(A, 0); din = bfs_dist(A.T, 0)
    live = lambda k: [v for v in range(1, n) if (not prune) or (dout[v] <= k and din[v] <= m + 1 - k)]
    last = m // 2 + 1 if mitm else m      # mitm: floor(m/2) products = steps 2 .. floor(m/2)+1
    tot = 0
    for k in range(2, last + 1):
        a, b = live(k - 1), live(k)
        tot += int(A[np.ix_(a, b)].sum())
    return tot

print('ring graph C_n^d (ring distance <= d). units: block products per Glynn term; q=49, r=r0=5 for the through-q column')
print('%3s %2s | %8s %8s %8s %8s %8s | %12s %12s %14s | %s' % ('n', 'd', 'dense', 'sparse', '+prune', '+mitm', 'thru-q', 'DFS nodes', 'ham.cycles', 'glynn best tot', 'DFS/Glynn'))
q, r = 49, 5
for n in (10, 12, 14, 16):
    m = n - 1
    for d in range(1, n // 2 + 1):
        A = ring_adj(n, d)
        dense = (m - 1) * m * m
        sp = blocks(A, prune=False); pr = blocks(A); mi = blocks(A, mitm=True)
        tq = (m // 2) * 2 * m * q / r           # mitm through-q: floor(m/2) steps of 4 m r0 r q flops
        nodes, ham = dfs_nodes(A)
        best = min(mi, tq) * 2 ** (m - 1)
        print('%3d %2d | %8d %8d %8d %8d %8d | %12d %12d %14.3g | %.3g' % (n, d, dense, sp, pr, mi, tq, nodes, ham, best, nodes / best))

print()
print('inclusion-exclusion over missing edges: sign terms of exact Glynn, total, vs 2^(n-2) of the transfer')
rng = np.random.default_rng(5)
for n in (6, 7):
    for d in (n // 2 - 1, ):
        A = ring_adj(n, d)
        mats = rand_mats(n, 5, 2, 'psd', rng)
        v, nt, nf = glynn_complement(mats, A)
        ref = brute(mats, A)[0]
        e = int((~A).sum() - n) // 2
        print('n=%d d=%d missing edges=%d forests=%d IE terms=%d  2^(n-2)=%d ratio=%.3f  (1.5^e=%.3f)  rel err %.1e'
              % (n, d, e, nf, nt, 2 ** (n - 2), nt / 2 ** (n - 2), 1.5 ** e, abs(v - ref) / abs(ref)))
