import numpy as np, time
from perterm import *
from esipy import mci as M
from bench import mk, cur_exact, cur_walks, T
rng = np.random.default_rng(5)

def walks_sym_dense(mats, adj, batch=256):
    """Symmetric matrices, dense block matrix T (one gemm per step as in esipy), palindrome halving."""
    mats = project(mats)
    n = len(mats); m = n - 1; k = mats[0].shape[0]
    fixed = int(np.argmin([np.linalg.matrix_rank(x) for x in mats]))
    order = [fixed] + [i for i in range(n) if i != fixed]
    mats, adj = [mats[i] for i in order], np.asarray(adj)[np.ix_(order, order)]
    L0, sig0, _ = factor(mats[0], True); r0 = L0.shape[1]
    fs = [factor(x, True) for x in mats[1:]]
    r = max(f[0].shape[1] for f in fs)
    L, Sg = np.zeros((m, k, r)), np.zeros((m, r))
    for v, (l, sg, _) in enumerate(fs):
        L[v, :, :l.shape[1]], Sg[v, :l.shape[1]] = l, sg
    Tm = (adj[1:, 1:, None, None] * np.einsum('uka,vkb->uvab', L, L)).transpose(0, 2, 1, 3).reshape(m * r, m * r)
    Y0 = (adj[0, 1:, None, None] * np.matmul(L0.T, L)).transpose(1, 0, 2).reshape(r0, m * r)
    nt = 1 << (m - 1); total = 0.0; h = (m - 1) // 2; odd = (m - 1) % 2
    for a in range(0, nt, batch):
        delta, sign = M._glynn_deltas(a, min(a + batch, nt), m)
        d = (delta[:, :, None] * Sg).reshape(len(sign), 1, m * r)
        Y = np.broadcast_to(Y0, (len(sign), r0, m * r))
        for _ in range(h):
            Y = ((Y * d).reshape(-1, m * r) @ Tm).reshape(-1, r0, m * r)
        W = Y * d
        if odd:
            W = (W.reshape(-1, m * r) @ Tm).reshape(-1, r0, m * r) * d
        total += np.dot(sign, np.einsum('bax,a,bax->b', W, sig0, Y))
    return total / nt

# correctness of walks_sym_dense against enumeration
e = 0
for n in (4, 5, 6, 7, 8, 9):
    for trial in range(3):
        mats = []
        for _ in range(n):
            B = rng.standard_normal((8, 2)); mats.append(B @ np.diag(rng.standard_normal(2)) @ B.T)
        adj = rng.random((n, n)) < 0.75; adj = np.triu(adj, 1); adj = adj | adj.T
        ref = brute(mats, adj)
        if abs(ref) > 1e-12: e = max(e, abs(walks_sym_dense(mats, adj) - ref) / abs(ref))
print('walks_sym_dense vs enumeration, max rel err', e)

print('RESTRICTED sym (n,q,r,d): current  dense+halving  speedup')
for n, q, r, d in ((13, 49, 5, 1), (13, 49, 5, 2), (14, 49, 5, 3), (12, 49, 14, 2)):
    mats = mk(n, q, r, True)
    dist = np.abs(np.arange(n)[:, None] - np.arange(n)); dist = np.minimum(dist, n - dist)
    adj = (dist > 0) & (dist <= d)
    (a, name), ta = T(cur_walks, mats, adj)
    b, tb = T(walks_sym_dense, mats, adj)
    print((n, q, r, d), '%.3f (%s) %.3f x%.1f  %.1e' % (ta, name, tb, ta / tb, abs(a - b) / abs(a)))

print('EXACT tbits scan (13,49,5,sym) and (13,49,5,nonsym)')
for sym in (True, False):
    mats = mk(13, 49, 5, sym)
    a, ta = T(cur_exact, mats)
    for tb in (4, 6, 8, 10):
        b, t = T(glynn_fast, mats, tbits=tb)
        print(sym, 'tbits', tb, 'cur %.3f new %.3f x%.1f' % (ta, t, ta / t))

# accuracy: long double reference for a symmetric PSD case (heavy cancellation in Glynn)
n, q, r = 11, 20, 3
mats = mk(n, q, r, True)
m = n - 1
ld = [x.astype(np.longdouble) for x in mats]
tot = np.longdouble(0)
L0 = np.linalg.cholesky(mats[0] + 0 * np.eye(q) + 1e-300 * np.eye(q)) if False else None
for sidx in range(1 << (m - 1)):
    dl = [1] + [1 - 2 * ((sidx >> i) & 1) for i in range(m - 1)]
    X = sum(d * x for d, x in zip(dl, ld[1:]))
    V = ld[0]
    for _ in range(m): V = V @ X
    tot += np.prod(dl) * np.trace(V)
ref = float(tot / 2 ** (m - 1))
print('accuracy n=11 q=20 r=3 sym PSD: current rel err %.1e, new rel err %.1e, new(nonsym path) %.1e' % (
    abs(cur_exact(mats) - ref) / abs(ref), abs(glynn_fast(mats) - ref) / abs(ref), abs(glynn_fast(mats, sym=False) - ref) / abs(ref)))
