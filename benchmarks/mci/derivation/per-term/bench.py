import numpy as np, time, sys
from perterm import *
from esipy import mci as M
rng = np.random.default_rng(11)

def mk(n, q, r, sym):
    out = []
    for _ in range(n):
        if sym:
            B = rng.standard_normal((q, r)); out.append(B @ B.T / q)
        else:
            out.append(rng.standard_normal((q, r)) @ rng.standard_normal((r, q)) / q)
    return out

def cur_exact(mats):
    mats = M._project(mats)
    m, k = len(mats) - 1, mats[0].shape[0]
    nt = 1 << (m - 1)
    batch = int(min(nt, max(1, 64e6 // (8 * 4 * k * k))))
    return M._kernel_glynn_exact((mats, None, 0, nt, batch))[0] / nt

def cur_walks(mats, adj):
    mats = M._project(mats)
    m, k = len(mats) - 1, mats[0].shape[0]
    nt = 1 << (m - 1)
    r0, r = np.linalg.matrix_rank(mats[0]), max(np.linalg.matrix_rank(x) for x in mats[1:])
    if r0 * (m * r) ** 2 < m * k ** 3 + m * m * k * k:
        kernel, tm, name = M._kernel_glynn_lowrank, 3 * r0 * m * r, 'lowrank'
    else:
        kernel, tm, name = M._kernel_glynn_walks, 3 * m * k * k, 'full'
    batch = int(min(nt, max(1, 64e6 // (8 * tm))))
    return kernel((mats, adj, 0, nt, batch))[0] / nt, name

def T(f, *a, **kw):
    t = time.perf_counter(); v = f(*a, **kw); return v, time.perf_counter() - t

print('EXACT  (n, q, r, sym): t_current  t_new(s)  speedup  model  rel.diff')
for n, q, r, sym in ((13, 49, 5, True), (13, 49, 5, False), (13, 49, 14, True), (13, 49, 14, False),
                     (14, 49, 5, True), (12, 80, 14, True), (12, 80, 14, False), (12, 30, 5, True)):
    mats = mk(n, q, r, sym)
    m = n - 1
    a, ta = T(cur_exact, mats)
    b, tb = T(glynn_fast, mats)
    k = min(q, n * r)
    s = best_s(m, k, r, sym)
    cur = (m.bit_length() - 1 + bin(m).count('1') - 1) * 2 * k ** 3 + 2 * m * k * k
    new = flops(m, k, r, s, sym) + k * k
    print((n, q, r, sym), '%.3f  %.3f (s=%d)  x%.1f  model x%.1f  %.1e' % (ta, tb, s, ta / tb, cur / new, abs(a - b) / abs(a)))

print('RESTRICTED ring distance <= d  (n, q, r, sym, d): t_current(kernel)  t_new  speedup  rel.diff')
for n, q, r, sym, d in ((13, 49, 5, True, 1), (13, 49, 5, True, 2), (13, 49, 5, False, 2), (13, 49, 5, True, 4),
                        (12, 49, 14, True, 2), (12, 49, 14, False, 2)):
    mats = mk(n, q, r, sym)
    dist = np.abs(np.arange(n)[:, None] - np.arange(n)); dist = np.minimum(dist, n - dist)
    adj = (dist > 0) & (dist <= d)
    (a, name), ta = T(cur_walks, mats, adj)
    b, tb = T(walks_fast, mats, adj)
    print((n, q, r, sym, d), '%.3f (%s)  %.3f  x%.1f  %.1e' % (ta, name, tb, ta / tb, abs(a - b) / max(abs(a), 1e-300)))
