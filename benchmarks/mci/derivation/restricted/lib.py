"""Restricted Glynn sums: reference implementations (numpy only, tiny sizes)."""
import numpy as np
from itertools import permutations, combinations
from math import factorial


# ---------- generators ----------
def rand_mats(n, q, r, kind, rng):
    """kind: 'psd' (L L^T), 'sym' (U diag(lam) U^T indefinite), 'nonsym' (L R^T)."""
    out = []
    for _ in range(n):
        L = rng.standard_normal((q, r))
        if kind == 'psd':
            out.append(L @ L.T)
        elif kind == 'sym':
            out.append((L * rng.standard_normal(r)) @ L.T)
        else:
            out.append(L @ rng.standard_normal((q, r)).T)
    return out


def rand_adj(n, p, rng):
    A = np.triu(rng.random((n, n)) < p, 1)
    return (A | A.T)


def ring_adj(n, d, odd=False):
    i = np.arange(n)
    dist = np.abs(i[:, None] - i[None, :])
    dist = np.minimum(dist, n - dist)
    A = (dist >= 1) & (dist <= d)
    if odd:
        A &= (dist % 2 == 1)
    return A


# ---------- brute force ----------
def brute(mats, A):
    n = len(mats); tot = 0.0; cnt = 0
    for p in permutations(range(1, n)):
        if not A[0, p[0]] or not A[p[-1], 0]:
            continue
        if any(not A[p[i], p[i + 1]] for i in range(n - 2)):
            continue
        P = mats[0]
        for v in p:
            P = P @ mats[v]
        tot += np.trace(P); cnt += 1
    return tot, cnt


def deltas(m, fixed=(0,)):
    """all sign vectors of length m with the positions in `fixed` held at +1."""
    free = [i for i in range(m) if i not in fixed]
    for s in range(1 << len(free)):
        d = np.ones(m)
        for b, i in enumerate(free):
            if (s >> b) & 1:
                d[i] = -1.0
        yield d


# ---------- factorizations ----------
def factor(M, tol=1e-10):
    U, s, Vt = np.linalg.svd(M)
    k = max(1, int((s > s[0] * tol).sum()))
    return U[:, :k] * s[:k], Vt[:k].T


def factor_sym(M, tol=1e-10):
    """M = U diag(lam) U^T, numerical rank."""
    lam, U = np.linalg.eigh((M + M.T) / 2)
    keep = np.abs(lam) > np.abs(lam).max() * tol
    return U[:, keep], lam[keep]


# ---------- current method (dense low-rank transfer, m-1 products per term) ----------
def glynn_dense(mats, A, fixed=(0,)):
    n = len(mats); m = n - 1
    L0, R0 = factor(mats[0])
    fs = [factor(M) for M in mats[1:]]
    off = np.cumsum([0] + [l.shape[1] for l, _ in fs])
    N = off[-1]
    T = np.zeros((N, N)); Y0 = np.zeros((L0.shape[1], N)); E = np.zeros((N, L0.shape[1]))
    for u in range(m):
        if A[0, u + 1]: Y0[:, off[u]:off[u + 1]] = R0.T @ fs[u][0]
        if A[u + 1, 0]: E[off[u]:off[u + 1]] = fs[u][1].T @ L0
        for v in range(m):
            if A[u + 1, v + 1]:
                T[off[u]:off[u + 1], off[v]:off[v + 1]] = fs[u][1].T @ fs[v][0]
    rep = np.diff(off)
    tot = 0.0; nt = 0
    for d in deltas(m, fixed):
        D = np.repeat(d, rep)
        Y = Y0 * D
        for _ in range(m - 1):
            Y = (Y @ T) * D
        tot += d.prod() * np.sum(Y * E.T); nt += 1
    return tot / nt


# ---------- P1: meet in the middle, symmetric AOMs + symmetric A ----------
def glynn_mitm(mats, A, fixed=(0,)):
    """W(delta) = sum(Lam0[:,None] * (G_a Dt) * G_b), G_k = Y0 (Dt T)^k, a+b = m-1.
    floor(m/2) products instead of m-1.  S_v = U_v lam_v U_v^T, T_uv = A_uv U_u^T U_v (symmetric),
    Dt = diag(delta_v lam_v), Y0_v = A_0v U_0^T U_v."""
    n = len(mats); m = n - 1
    U0, lam0 = factor_sym(mats[0])
    fs = [factor_sym(M) for M in mats[1:]]
    off = np.cumsum([0] + [u.shape[1] for u, _ in fs]); N = off[-1]
    T = np.zeros((N, N)); Y0 = np.zeros((U0.shape[1], N))
    for u in range(m):
        if A[0, u + 1]: Y0[:, off[u]:off[u + 1]] = U0.T @ fs[u][0]
        for v in range(m):
            if A[u + 1, v + 1]:
                T[off[u]:off[u + 1], off[v]:off[v + 1]] = fs[u][0].T @ fs[v][0]
    lam = np.concatenate([l for _, l in fs]); rep = np.diff(off)
    b = m // 2; a = m - 1 - b          # a <= b, b = floor(m/2)
    tot = 0.0; nt = 0; nprod = 0
    for d in deltas(m, fixed):
        Dt = np.repeat(d, rep) * lam
        G = Y0; Ga = Y0 if a == 0 else None
        for k in range(1, b + 1):
            G = (G * Dt) @ T; nprod += 1
            if k == a: Ga = G
        tot += d.prod() * np.sum(lam0[:, None] * (Ga * Dt) * G); nt += 1
    return tot / nt, nprod / nt


# ---------- P2: block-sparse transfer with cone pruning, and through-q transfer ----------
def bfs_dist(A, src=0):
    n = len(A); dist = np.full(n, 10 ** 9); dist[src] = 0; fr = [src]
    while fr:
        nx = []
        for u in fr:
            for v in np.nonzero(A[u])[0]:
                if dist[v] > dist[u] + 1:
                    dist[v] = dist[u] + 1; nx.append(v)
        fr = nx
    return dist


def glynn_sparse(mats, A, fixed=(0,), prune=True):
    """General (non-symmetric ok). State Y[v] (r0 x r_v) only for live (step, v) pairs:
    v live at step k (k = 1..m) iff dist(0,v) <= k and dist(v,0) <= m+1-k.
    Returns value and the number of block products r0 x r x r per term."""
    n = len(mats); m = n - 1
    L0, R0 = factor(mats[0]); fs = [None] + [factor(M) for M in mats[1:]]
    dout = bfs_dist(A, 0); din = bfs_dist(A.T, 0)
    blk = {(u, v): fs[u][1].T @ fs[v][0] for u in range(1, n) for v in range(1, n) if A[u, v]}
    live = lambda k, v: (not prune) or (dout[v] <= k and din[v] <= m + 1 - k)
    tot = 0.0; nt = 0; nblk = 0
    for d in deltas(m, fixed):
        Y = {v: d[v - 1] * (R0.T @ fs[v][0]) for v in range(1, n) if A[0, v] and live(1, v)}
        for k in range(2, m + 1):
            Yn = {}
            for v in range(1, n):
                if not live(k, v): continue
                acc = None
                for u in Y:
                    if A[u, v]:
                        t = Y[u] @ blk[u, v]; nblk += 1
                        acc = t if acc is None else acc + t
                if acc is not None: Yn[v] = d[v - 1] * acc
            Y = Yn
        w = sum(np.sum(Y[v] * (fs[v][1].T @ L0).T) for v in Y if A[v, 0])
        tot += d.prod() * w; nt += 1
    return tot / nt, nblk / nt


def glynn_throughq(mats, A, fixed=(0,), selfloops=True):
    """Transfer factored through the q-space: P_u = Y_u R_u^T (r0 x q), Z_v = sum_{u~v} P_u, Y'_v = delta_v Z_v L_v.
    Self loops u = v may be added freely (walks using them repeat a vertex and cancel)."""
    n = len(mats); m = n - 1
    L0, R0 = factor(mats[0]); fs = [None] + [factor(M) for M in mats[1:]]
    B = A.copy().astype(bool)
    if selfloops: np.fill_diagonal(B, True)
    tot = 0.0; nt = 0
    for d in deltas(m, fixed):
        Y = [None] + [d[v - 1] * A[0, v] * (R0.T @ fs[v][0]) for v in range(1, n)]
        for k in range(2, m + 1):
            P = [None] + [Y[u] @ fs[u][1].T for u in range(1, n)]
            Y = [None] + [d[v - 1] * (sum(P[u] for u in range(1, n) if B[u, v]) + 0 * P[v]) @ fs[v][0] for v in range(1, n)]
        w = sum(np.sum(Y[v] * (fs[v][1].T @ L0).T) for v in range(1, n) if A[v, 0])
        tot += d.prod() * w; nt += 1
    return tot / nt


# ---------- P3: bipartite sign reduction and class-structured transfer ----------
def bipartition(A):
    """two-colouring of the graph A (connected through 0) or None."""
    n = len(A); col = np.full(n, -1); col[0] = 0; st = [0]
    while st:
        u = st.pop()
        for v in np.nonzero(A[u])[0]:
            if col[v] < 0: col[v] = 1 - col[u]; st.append(v)
            elif col[v] == col[u]: return None
    return None if (col < 0).any() else col


def bipartite_fixed(A):
    """positions (0-based among vertices 1..m) of the two signs that can be held at +1, or None
    (None also when the parts are unbalanced: then there is no Hamiltonian cycle and R = 0)."""
    col = bipartition(A); n = len(A)
    if col is None or 2 * int(col.sum()) != n: return None
    q1 = [v for v in range(1, n) if col[v] == 1][0]
    p1 = [v for v in range(1, n) if col[v] == 0]
    return (q1 - 1,) + ((p1[0] - 1,) if p1 else ())


def glynn_classes(mats, cls, Bc, fixed=(0,)):
    """A[u,v] = Bc[cls[u], cls[v]] for u != v (0 is its own class cls[0], Bc[c,c] arbitrary for a singleton).
    State Om[c] (r0 x q) per class; Om'[c] = (sum_c' Bc[c',c] Om[c']) X_c, X_c = sum_{v in c} delta_v S_v.
    W = Tr(R0^T-projected) closing with Bc[c, cls[0]]."""
    n = len(mats); m = n - 1; C = Bc.shape[0]; c0 = cls[0]
    L0, R0 = factor(mats[0])
    tot = 0.0; nt = 0
    for d in deltas(m, fixed):
        X = [sum((d[v - 1] * mats[v] for v in range(1, n) if cls[v] == c), np.zeros_like(mats[0])) for c in range(C)]
        Om = [Bc[c0, c] * (R0.T @ X[c]) for c in range(C)]
        for k in range(2, m + 1):
            Om = [sum(Bc[cp, c] * Om[cp] for cp in range(C)) @ X[c] for c in range(C)]
        w = sum(Bc[c, c0] * np.sum(Om[c] * L0.T) for c in range(C))
        tot += d.prod() * w; nt += 1
    return tot / nt


# ---------- P4: inclusion-exclusion over the missing edges ----------
def linear_forest(edges):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v); adj.setdefault(v, []).append(u)
        if len(adj[u]) > 2 or len(adj[v]) > 2: return None
    seen = set(); paths = []
    for s in [u for u in adj if len(adj[u]) == 1]:
        if s in seen: continue
        p = [s]; seen.add(s)
        while True:
            nx = [w for w in adj[p[-1]] if w not in seen]
            if not nx: break
            p.append(nx[0]); seen.add(nx[0])
        paths.append(p)
    return paths if len(seen) == len(adj) else None   # a closed cycle leaves unseen vertices


def glynn_exact(mats):
    """plain Glynn on the complete graph; returns value and number of sign terms."""
    n = len(mats); m = n - 1
    if m == 0: return np.trace(mats[0]), 1
    tot = 0.0; nt = 0
    for d in deltas(m):
        X = sum(di * M for di, M in zip(d, mats[1:]))
        tot += d.prod() * np.trace(mats[0] @ np.linalg.matrix_power(X, m)); nt += 1
    return tot / nt, nt


def glynn_complement(mats, A):
    """R_A = sum over linear forests F of missing edges of (-1)^|F| R_complete(contracted instance).
    Returns value, number of Glynn sign terms used in total, number of forests."""
    n = len(mats)
    miss = [(u, v) for u in range(n) for v in range(u + 1, n) if not A[u, v]]
    tot, nterms = glynn_exact(mats); nfor = 1
    for c in range(1, min(len(miss), n - 1) + 1):
        for F in combinations(miss, c):
            paths = linear_forest(F)
            if paths is None: continue
            touched = set(); first = None; others = []
            for p in paths:
                touched.update(p)
                fwd = mats[p[0]]
                for v in p[1:]: fwd = fwd @ mats[v]
                rev = mats[p[-1]]
                for v in reversed(p[:-1]): rev = rev @ mats[v]
                if 0 in p: first = fwd + rev
                else: others.append(fwd + rev)
            if first is None: first = mats[0]
            cm = [first] + others + [mats[i] for i in range(1, n) if i not in touched]
            if len(cm) == 1 and len(paths[0]) == n:
                # a Hamiltonian path of missing edges: contracted to one vertex, both orientations close
                # through the edge (p[-1], p[0]), which the forest did not select
                t, k = np.trace(first), 1
            else:
                t, k = glynn_exact(cm)
            tot += (-1) ** c * t; nterms += k; nfor += 1
    # F = a whole Hamiltonian cycle made of missing edges (|F| = n): contained only in itself, both directions
    if len(miss) >= n >= 3:
        for F in combinations(miss, n):
            cyc = ham_cycle(F, n)
            if cyc is None: continue
            fwd = mats[cyc[0]]
            for v in cyc[1:]: fwd = fwd @ mats[v]
            rev = mats[cyc[0]]
            for v in reversed(cyc[1:]): rev = rev @ mats[v]
            tot += (-1) ** n * (np.trace(fwd) + np.trace(rev)); nterms += 1; nfor += 1
    return tot, nterms, nfor


def ham_cycle(F, n):
    adj = {}
    for u, v in F:
        adj.setdefault(u, []).append(v); adj.setdefault(v, []).append(u)
    if len(adj) != n or any(len(a) != 2 for a in adj.values()): return None
    cyc = [0, adj[0][0]]
    while len(cyc) < n:
        nx = [w for w in adj[cyc[-1]] if w != cyc[-2]][0]
        if nx == 0: return None
        cyc.append(nx)
    return cyc
