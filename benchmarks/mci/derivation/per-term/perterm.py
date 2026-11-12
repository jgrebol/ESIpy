"""Per-term accelerations of Glynn's identity for the MCI raw sum. numpy only."""
import numpy as np
from itertools import permutations
from functools import reduce

EPS = np.finfo(float).eps


# ---------------------------------------------------------------- reference
def brute(mats, adj=None):
    m = len(mats) - 1
    tot = 0.0
    for p in permutations(range(1, m + 1)):
        if adj is not None:
            c = (0,) + p + (0,)
            if not all(adj[c[i], c[i + 1]] for i in range(m + 1)):
                continue
        tot += np.trace(reduce(np.dot, [mats[0]] + [mats[i] for i in p]))
    return tot


# ---------------------------------------------------------------- helpers
def is_sym(mats):
    return all(np.allclose(M, M.T, rtol=1e-12, atol=1e-14 * np.abs(M).max()) for M in mats)


def project(mats):
    """Orthogonal projection on the joint column+row space (keeps symmetry, keeps all traces of products)."""
    U, s, _ = np.linalg.svd(np.hstack(list(mats) + [M.T for M in mats]), full_matrices=False)
    Q = U[:, :int((s > s[0] * len(s) * EPS).sum())]
    return [Q.T @ M @ Q for M in mats] if Q.shape[1] < len(Q) else list(mats)


def factor(M, sym):
    """M = L diag(sig) R^T with numerical rank; sym: R = L (eigen-decomposition, sig may be negative)."""
    if sym:
        w, U = np.linalg.eigh((M + M.T) / 2)
        keep = np.abs(w) > np.abs(w).max() * len(w) * EPS
        return U[:, keep], w[keep], U[:, keep]
    U, s, Vt = np.linalg.svd(M)
    k = max(1, int((s > s[0] * len(s) * EPS).sum()))
    return U[:, :k], s[:k], Vt[:k].T


def symmetrize_no(aoms, occ):
    """sqrt(occ) A_i  ->  occ^(1/4) A_i occ^(1/4) on the orbitals with occ > 0 (a similarity: same traces)."""
    keep = occ > 0
    w = occ[keep] ** 0.25
    return [w[:, None] * A[np.ix_(keep, keep)] * w[None, :] for A in aoms]


def nsteps(x, s):
    """Fewest powers of two <= 2^s that add up to x."""
    return (x >> s) + bin(x & ((1 << s) - 1)).count('1')


def chain(x, s):
    out = [s] * (x >> s)
    out += [t for t in range(s - 1, -1, -1) if (x >> t) & 1]
    return out


def plan(m, s, sym):
    """(exponents to reach a, exponents from a to b) with a + b = m; non-symmetric: a = 0."""
    if not sym:
        return [], chain(m, s)
    a = min(range(m // 2 + 1), key=lambda a: nsteps(a, s) + nsteps(m - 2 * a, s))
    return chain(a, s), chain(m - 2 * a, s)


def flops(m, k, r0, s, sym):
    a, b = plan(m, s, sym)
    return s * 2 * k ** 3 + (len(a) + len(b)) * 2 * k * k * r0


def best_s(m, k, r0, sym):
    return min(range(m.bit_length()), key=lambda s: flops(m, k, r0, s, sym))


# ---------------------------------------------------------------- exact sum
def glynn_fast(mats, s=None, tbits=6, fixed=None, do_project=True, sym=None):
    """
    R = sum_sigma Tr(S_0 S_s1 .. S_sm) by Glynn with
      - fixed atom = the one of lowest rank (cyclic invariance),
      - Tr(S_0 X^m) = sum(sig0 * (R0^T X^a ... ) ...) with block matrix-vector products (m of them; ceil(m/2) if symmetric),
      - optionally s squarings of X (hybrid), chosen by the flop model when s is None,
      - X built as H + Q[b]: H follows a Gray code over the high signs (one axpy), Q is a table of 2^tbits low-sign sums.
    Memory: 2^tbits (k^2 + 2 k r0), independent of n.
    """
    if do_project:
        mats = project(mats)
    sym = is_sym(mats) if sym is None else sym
    n = len(mats)
    if fixed is None:
        fixed = int(np.argmin([np.linalg.matrix_rank(M) for M in mats]))
    mats = [mats[fixed]] + [M for i, M in enumerate(mats) if i != fixed]
    m, k = n - 1, mats[0].shape[0]
    L0, sig0, R0 = factor(mats[0], sym)
    r0 = L0.shape[1]
    if s is None:
        s = best_s(m, k, r0, sym)
    pa, pb = plan(m, s, sym)

    rest = np.array(mats[1:])
    t = min(tbits, m - 1)                       # low signs: matrices 1..t of rest (rest[0] has delta = +1)
    bits = (np.arange(1 << t)[:, None] >> np.arange(t)) & 1
    dl = 1.0 - 2 * bits
    Q = rest[0] + np.tensordot(dl, rest[1:t + 1], axes=1) if t else rest[:1].copy()
    sl = dl.prod(axis=1)
    hi = rest[t + 1:]                           # Gray code over these
    H = hi.sum(axis=0) if len(hi) else np.zeros((k, k))
    dh = np.ones(len(hi))
    sh, total = 1.0, 0.0
    V0 = np.broadcast_to(L0, (1 << t, k, r0))
    for g in range(1 << len(hi)):
        if g:
            j = (g & -g).bit_length() - 1       # bit flipped by the Gray code
            dh[j] = -dh[j]
            H += (2 * dh[j]) * hi[j]
            sh = -sh
        X = H + Q
        P = [X]
        for _ in range(s):
            P.append(np.matmul(P[-1], P[-1]))
        V = V0
        for e in pa:
            V = np.matmul(P[e], V)
        Va = V if sym else None
        for e in pb:
            V = np.matmul(P[e], V)
        if sym:
            tr = np.einsum('bka,a,bka->b', Va, sig0, V)
        else:
            tr = np.einsum('ka,a,bka->b', R0, sig0, V)
        total += sh * np.dot(sl, tr)
    return total / 2 ** (m - 1)


# ---------------------------------------------------------------- restricted sum
def walks_fast(mats, adj, fixed=None, sym=None, tbits=8):
    """
    Sum over the permutations whose cycle 0 -> s1 -> ... -> sm -> 0 only uses steps of adj (symmetric 0/1).
    Low-rank transfer recursion with block-SPARSE T (only allowed pairs), and for symmetric matrices the
    palindrome  Y0 D (T D)^(m-1) Y0^T = Z M Z^T  that needs ceil((m-1)/2) products with T instead of m-1.
    """
    adj = np.asarray(adj).astype(bool)
    assert (adj == adj.T).all()
    mats = project(mats)
    sym = is_sym(mats) if sym is None else sym
    n = len(mats)
    if fixed is None:
        fixed = int(np.argmin([np.linalg.matrix_rank(M) for M in mats]))
    order = [fixed] + [i for i in range(n) if i != fixed]
    mats, adj = [mats[i] for i in order], adj[np.ix_(order, order)]
    m, k = n - 1, mats[0].shape[0]
    L0, sig0, R0 = factor(mats[0], sym)
    r0 = L0.shape[1]
    fs = [factor(M, sym) for M in mats[1:]]
    r = max(f[0].shape[1] for f in fs)
    L, R, Sg = np.zeros((m, k, r)), np.zeros((m, k, r)), np.zeros((m, r))
    for v, (l, sg, rr) in enumerate(fs):
        L[v, :, :l.shape[1]], R[v, :, :l.shape[1]], Sg[v, :l.shape[1]] = l, rr, sg
    a = adj[1:, 1:]
    deg = max(1, int(a.sum(axis=0).max()))
    nb = np.zeros((m, deg), dtype=int)          # in-neighbours of v, padded
    G = np.zeros((m, deg, r, r))                # G[v, j] = R_u^T L_v, u = nb[v, j]
    for v in range(m):
        for j, u in enumerate(np.flatnonzero(a[:, v])):
            nb[v, j], G[v, j] = u, R[u].T @ L[v]
    G = G.reshape(m, deg * r, r)
    Y0 = adj[0, 1:, None, None] * np.matmul(R0.T, L)            # (m, r0, r)   0 -> v
    E = adj[1:, 0, None, None] * np.matmul(L0.T, R)             # (m, r0, r)   v -> 0, E_v^T = R_v^T L_0

    def step(Y):                                                # Y: (b, m, r0, r) -> Y T
        Z = Y[:, nb].transpose(0, 1, 3, 2, 4).reshape(len(Y), m, r0, deg * r)
        return np.matmul(Z, G)

    t = min(tbits, m - 1)
    nt = 1 << (m - 1)
    total = 0.0
    for a0 in range(0, nt, 1 << t):
        bits = (np.arange(a0, a0 + (1 << t))[:, None] >> np.arange(m - 1)) & 1
        delta = np.ones((1 << t, m))
        delta[:, 1:] -= 2 * bits
        sign = delta.prod(axis=1)
        D = (delta[:, :, None] * Sg)[:, :, None, :]             # (b, m, 1, r)
        Y = np.broadcast_to(Y0, (1 << t,) + Y0.shape)
        if sym:
            h = (m - 1) // 2
            for _ in range(h):
                Y = step(Y * D)                                 # Z = Y0 (D T)^h
            ZD = Y * D
            W = step(ZD) * D if (m - 1) % 2 else ZD             # odd: (Z D T D) . Z ; even: (Z D) . Z
            tr = np.einsum('bvar,a,bvar->b', W, sig0, Y)
        else:
            Y = Y * D
            for _ in range(m - 1):
                Y = step(Y) * D
            tr = np.einsum('bvar,a,var->b', Y, sig0, E)
        total += np.dot(sign, tr)
    return total / nt
