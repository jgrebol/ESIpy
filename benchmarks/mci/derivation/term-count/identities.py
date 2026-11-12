"""Term-count angle: numeric checks of every identity against explicit enumeration. numpy only, n <= 8."""
import itertools, numpy as np
from math import factorial, comb
rng = np.random.default_rng(7)

def rand_mats(n, q, kind):
    if kind == 'nonsym':
        return [rng.standard_normal((q, q)) / np.sqrt(q) for _ in range(n)]
    out = []                                   # symmetric PSD low rank
    for _ in range(n):
        L = rng.standard_normal((q, 2)); out.append(L @ L.T / q)
    return out

def brute(mats, adj=None):
    n = len(mats); m = n - 1; tot = 0.0; cnt = 0
    for p in itertools.permutations(range(1, n)):
        if adj is not None:
            c = (0,) + p + (0,)
            if not all(adj[c[i], c[i + 1]] for i in range(n)): continue
        P = mats[0]
        for i in p: P = P @ mats[i]
        tot += np.trace(P); cnt += 1
    return tot, cnt

# ---------- GF(2) machinery: the sign-symmetry group of the restricted Glynn summand ----------
def _reduce(basis, x):
    for b in basis: x = min(x, x ^ b)
    return x
def _span(vecs):
    basis = []
    for x in vecs:
        x = _reduce(basis, x)
        if x: basis.append(x); basis.sort(reverse=True)
    return basis

def defect_space(adj):
    """Basis (bitmasks over vertices 1..m -> bits 0..m-1) of D = span{ parity(w) + 1 } over all closed walks
    0->v1->...->vm->0 with v_j in 1..m.  Polynomial time: linear spans are propagated through the transfer."""
    n = len(adj); m = n - 1; H = 1 << m                      # H = homogenising bit
    V = [[(1 << v) | H] if adj[0, v + 1] else [] for v in range(m)]
    for _ in range(m - 1):
        V = [_span([x ^ ((1 << v) if x & H else 0) for u in range(m) if adj[u + 1, v + 1] for x in V[u]]) for v in range(m)]
    allv = [x for v in range(m) if adj[v + 1, 0] for x in V[v]]
    ones = H - 1
    return _span([(x & ones) ^ (ones if x & H else 0) for x in allv])

def defect_space_brute(adj):
    n = len(adj); m = n - 1; vecs = set()
    for w in itertools.product(range(m), repeat=m):
        c = (0,) + tuple(v + 1 for v in w) + (0,)
        if all(adj[c[i], c[i + 1]] for i in range(n)):
            x = (1 << m) - 1
            for v in w: x ^= 1 << v
            vecs.add(x)
    return _span(vecs)

def symmetry_pivots(adj):
    """Vertices (1..m) whose sign can be fixed to +1: pivots of G = D^perp. Returns (pivots, dim D)."""
    m = len(adj) - 1
    D = defect_space(adj)
    # G = {t : <t,d>=0 for all d in D}; brute null space by elimination on the m x m GF(2) system
    rows = list(D); piv_cols = []
    Dred = []
    for x in rows:                      # D is already an echelon-like basis (distinct leading bits after _span)
        Dred.append(x)
    lead = [x.bit_length() - 1 for x in Dred]          # leading bit of each basis vector of D
    free = [c for c in range(m) if c not in lead]      # free coordinates of the system D t = 0
    # each free coordinate gives one basis vector of G containing that coordinate and none of the other free ones:
    # so the free coordinates are pivots of G, and fixing delta=+1 there picks one representative per coset.
    return [c + 1 for c in free], len(Dred)

def walk_trace(mats, adj, delta):
    """F(delta) = sum over closed walks 0->v1..vm->0 of Tr(S_0 prod delta_v S_v)."""
    n = len(mats); m = n - 1
    W = [adj[0, v] * delta[v - 1] * (mats[0] @ mats[v]) for v in range(1, n)]
    for _ in range(m - 1):
        W = [delta[v - 1] * (sum(W[u - 1] for u in range(1, n) if adj[u, v]) + 0 * W[0]) @ mats[v] for v in range(1, n)]
    return sum(np.trace(W[v - 1]) for v in range(1, n) if adj[v, 0])

def glynn_reduced(mats, adj):
    """Restricted Glynn summed over one representative per coset of the sign-symmetry group. 2^{dim D} terms."""
    n = len(mats); m = n - 1
    piv, dimD = symmetry_pivots(adj)
    free = [v for v in range(1, n) if v not in piv]
    assert len(free) == dimD
    tot = 0.0
    for bits in itertools.product((1.0, -1.0), repeat=len(free)):
        delta = np.ones(m)
        for v, b in zip(free, bits): delta[v - 1] = b
        tot += np.prod(delta) * walk_trace(mats, adj, delta)
    return tot / 2 ** dimD, 2 ** dimD

def glynn_quotient(mats, cls, B, fix):
    """Adjacency depending only on vertex classes: adj[u,v] = B[cls[u],cls[v]] (u != v).  State = one matrix per CLASS.
    fix = list of vertices whose sign is fixed to +1."""
    n = len(mats); m = n - 1; c = B.shape[0]
    free = [v for v in range(1, n) if v not in fix]
    tot = 0.0
    for bits in itertools.product((1.0, -1.0), repeat=len(free)):
        delta = np.ones(n)
        for v, b in zip(free, bits): delta[v] = b
        X = [sum((delta[v] * mats[v] for v in range(1, n) if cls[v] == a), np.zeros_like(mats[0])) for a in range(c)]
        Z = [B[cls[0], b] * (mats[0] @ X[b]) for b in range(c)]
        for _ in range(m - 1):
            Z = [sum((Z[a] for a in range(c) if B[a, b]), np.zeros_like(mats[0])) @ X[b] for b in range(c)]
        tot += np.prod(delta[1:]) * sum(np.trace(Z[a]) for a in range(c) if B[a, cls[0]])
    return tot / 2 ** len(free), 2 ** len(free)

def glynn_perm_sum(ms):
    """sum over orderings of prod(ms) as a matrix, by Glynn (2^{k-1} terms)."""
    k = len(ms)
    if k == 0: return None
    out = 0
    for bits in itertools.product((1.0, -1.0), repeat=k - 1):
        d = (1.0,) + bits
        out = out + np.prod(d) * np.linalg.matrix_power(sum(di * M for di, M in zip(d, ms)), k)
    return out / 2 ** (k - 1)

def split_glynn(mats, k):
    n = len(mats); m = n - 1; tot = 0.0; nterms = 0
    for T in itertools.combinations(range(1, n), k):
        Tc = [v for v in range(1, n) if v not in T]
        P = mats[0] @ glynn_perm_sum([mats[v] for v in T])
        if Tc: P = P @ glynn_perm_sum([mats[v] for v in Tc])
        tot += np.trace(P); nterms += 2 ** (k - 1) + (2 ** (len(Tc) - 1) if Tc else 0)
    return tot, nterms

def relerr(a, b): return abs(a - b) / max(abs(b), 1e-300)

if __name__ == '__main__':
    q = 6
    # ---- A. defect space: polynomial propagation == brute force over all walks (random graphs, n = 4..7)
    bad = 0; ntest = 0
    for n in (4, 5, 6, 7):
        for _ in range(60):
            p = rng.uniform(0.3, 0.9)
            A = np.triu(rng.random((n, n)) < p, 1); A = (A | A.T)
            ntest += 1
            Dp, Db = defect_space(A), defect_space_brute(A)
            bad += (len(Dp) != len(Db)) or any(_reduce(Dp, x) for x in Db)
    print(f"A. defect-space propagation vs brute force: {ntest} random graphs, mismatches = {bad}")

    # ---- B. symmetry-reduced restricted Glynn vs explicit enumeration (random symmetric adjacency)
    worst = 0; hist = {}
    for kind in ('nonsym', 'sym-lowrank'):
        for n in (5, 6, 7, 8):
            done = 0
            while done < 8:
                p = rng.uniform(0.35, 0.9)
                A = np.triu(rng.random((n, n)) < p, 1); A = (A | A.T)
                mats = rand_mats(n, q, kind)
                ref, cnt = brute(mats, A)
                if cnt == 0: continue
                val, nt = glynn_reduced(mats, A)
                worst = max(worst, relerr(val, ref)); done += 1
                hist[(n, nt)] = hist.get((n, nt), 0) + 1
    print(f"B. symmetry-reduced restricted Glynn, random graphs n=5..8: max rel err = {worst:.2e}")
    print("   (n, #terms): count  ", dict(sorted(hist.items())), "  [plain Glynn: 2^(n-2)]")

    # ---- C. bipartite adjacency (random bipartite + ring 'odd distance' graphs): 2^(m-2) terms
    worst = 0
    for kind in ('nonsym', 'sym-lowrank'):
        for n in (4, 6, 8):
            for trial in range(6):
                side = np.array([i % 2 for i in range(n)])
                if trial == 0:   # ring, odd ring distances only (alg 3 with large d)
                    A = np.array([[(min(abs(i - j), n - abs(i - j)) % 2 == 1) for j in range(n)] for i in range(n)])
                else:
                    perm = rng.permutation(n); side = side[perm]
                    A = (side[:, None] != side[None, :]) & (rng.random((n, n)) < 0.85); A = A & A.T
                mats = rand_mats(n, q, kind)
                ref, cnt = brute(mats, A)
                # explicit rule: fix +1 on one vertex of each side
                m = n - 1
                fixA = next(v for v in range(1, n) if side[v] == side[0]); fixB = next(v for v in range(1, n) if side[v] != side[0])
                tot = 0.0; free = [v for v in range(1, n) if v not in (fixA, fixB)]
                for bits in itertools.product((1.0, -1.0), repeat=len(free)):
                    d = np.ones(m)
                    for v, b in zip(free, bits): d[v - 1] = b
                    tot += np.prod(d) * walk_trace(mats, A, d)
                val = tot / 2 ** (m - 2)
                worst = max(worst, abs(val - ref) / max(abs(ref), 1e-12) if cnt else abs(val))
    print(f"C. bipartite halving (2^(n-3) terms), n=4,6,8: max rel err = {worst:.2e}")

    # ---- D. class-dependent adjacency: quotient transfer (one state matrix per class)
    worst = 0
    for kind in ('nonsym', 'sym-lowrank'):
        for n in (5, 6, 7, 8):
            done = 0
            while done < 6:
                c = int(rng.integers(1, 4))
                cls = rng.integers(0, c, n)
                B = np.triu(rng.random((c, c)) < 0.75); B = B | B.T
                A = B[cls[:, None], cls[None, :]].copy(); np.fill_diagonal(A, False)
                mats = rand_mats(n, q, kind)
                ref, cnt = brute(mats, A)
                if cnt == 0: continue
                piv, dimD = symmetry_pivots(A)
                val, nt = glynn_quotient(mats, cls, B, piv)
                worst = max(worst, relerr(val, ref)); done += 1
    print(f"D. quotient (class) transfer + symmetry reduction, n=5..8, 1-3 classes: max rel err = {worst:.2e}")

    # ---- E. split Glynn (sum over subsets T of size k of Glynn(T) Glynn(T^c)): identity + term counts
    worst = 0
    for kind in ('nonsym', 'sym-lowrank'):
        for n in (5, 6, 7):
            mats = rand_mats(n, q, kind); ref, _ = brute(mats)
            for k in range(1, n - 1):
                val, nt = split_glynn(mats, k); worst = max(worst, relerr(val, ref))
    print(f"E. split Glynn identity, n=5..7, all k: max rel err = {worst:.2e}")
    viol = [(m, k) for m in range(2, 41) for k in range(1, m) if comb(m, k) * (2 ** (k - 1) + 2 ** (m - k - 1)) < 2 ** (m - 1)]
    print(f"   C(m,k)(2^(k-1)+2^(m-k-1)) < 2^(m-1) for some 2<=m<=40, 1<=k<m ?  violations = {viol}")
    print("   best ratio split/plain per m:", {m: round(min(comb(m, k) * (2 ** (k - 1) + 2 ** (m - k - 1)) for k in range(1, m)) / 2 ** (m - 1), 2) for m in (4, 8, 13, 20)})

    # ---- F. term counts 2^{dim D} for the ring approximations (ring distance <= d, and odd distances <= d)
    print("F. dim D (terms = 2^dimD; plain Glynn = 2^(n-2)) for ring graphs:")
    for n in (6, 7, 8, 10, 12, 13, 14, 16, 20):
        row = []
        for d in (1, 2, 3, 4):
            dist = np.array([[min(abs(i - j), n - abs(i - j)) for j in range(n)] for i in range(n)])
            A1 = (dist <= d) & (dist > 0); A2 = A1 & (dist % 2 == 1)
            row.append((d, len(defect_space(A1)), len(defect_space(A2))))
        print(f"   n={n:2d} n-2={n-2:2d}  (d, dimD[dist<=d], dimD[odd dist<=d]):", row)
