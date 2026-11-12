"""Sign-symmetry group of the restricted Glynn sum and the reduced term count 2^(m - dim W).

F(delta) = sum over walks 0->v1->...->vm->0 allowed by adj (v_t in 1..m) of prod_t delta_{v_t} Tr(S_0 S_v1...S_vm)
R_adj    = 2^-m sum_{delta in {+-1}^m} prod(delta) F(delta)
W = { C subset of {1..m} : every such walk visits C a number of times == |C| (mod 2) }   (GF(2) vector space)
Then prod(delta) F(delta) is invariant under flipping the signs of any C in W, so
R_adj = 2^-(m-dimW) sum over delta with delta_pivot = +1 for the pivots of a reduced basis of W.
"""
import numpy as np, sys
from itertools import product
from common import rand_mats, brute, rand_adj

def parity_space(adj):
    """All C in W as bitmasks (bit v-1 for vertex v), by signed walk counting (exact integers)."""
    n = len(adj); m = n - 1
    A = adj[1:, 1:].astype(object); a = adj[0, 1:].astype(object); b = adj[1:, 0].astype(object)
    def count(d):
        w = a * d
        for _ in range(m - 1):
            w = w.dot(A) * d
        return int(w.dot(b))
    total = count(np.ones(m, dtype=object))
    W = []
    for c in range(1 << m):
        d = np.array([-1 if c >> i & 1 else 1 for i in range(m)], dtype=object)
        if count(d) == (-1) ** bin(c).count('1') * total:
            W.append(c)
    return W, total

def cheap_parity_space(adj):
    """Polynomial-time generators: all-ones; one side of a bipartition of G (n even)
    or of G-0 (m even, the side having m/2 vertices... parity m/2 == |side|)."""
    n = len(adj); m = n - 1
    gens = [(1 << m) - 1]
    def bip(vs):
        col = {}
        for s in vs:
            if s in col: continue
            col[s] = 0; st = [s]
            while st:
                u = st.pop()
                for v in vs:
                    if adj[u, v]:
                        if v not in col: col[v] = 1 - col[u]; st.append(v)
                        elif col[v] == col[u]: return None
        return col
    def connected(vs):
        seen = {vs[0]}; st = [vs[0]]
        while st:
            u = st.pop()
            for v in vs:
                if adj[u, v] and v not in seen: seen.add(v); st.append(v)
        return len(seen) == len(vs)
    col = bip(list(range(n)))
    if col is not None and connected(list(range(n))):
        gens.append(sum(1 << (v - 1) for v in range(1, n) if col[v] != col[0]))
    else:
        vs = list(range(1, n))
        col = bip(vs)
        if col is not None and connected(vs) and m % 2 == 0:
            side = sum(1 << (v - 1) for v in vs if col[v] == 0)
            if bin(side).count('1') % 2 == (m // 2) % 2:
                gens.append(side)
    return gens

def gf2_basis(masks):
    """Reduced echelon basis; returns (basis, pivot bit positions)."""
    basis = []
    for x in masks:
        for b in basis:
            x = min(x, x ^ b)
        if x:
            basis.append(x)
    # reduce
    basis.sort(reverse=True)
    piv = []
    for i, b in enumerate(basis):
        p = b.bit_length() - 1
        piv.append(p)
        for j in range(len(basis)):
            if j != i and basis[j] >> p & 1:
                basis[j] ^= b
    return basis, piv

def glynn_walk_reduced(mats, adj, pivots):
    """Restricted Glynn sum with delta_p = +1 for p in pivots: 2^(m-len(pivots)) terms. Full-matrix transfer."""
    n = len(mats); m = n - 1
    free = [i for i in range(m) if i not in pivots]
    tot = 0.0; nterms = 0
    for bits in product((1.0, -1.0), repeat=len(free)):
        d = np.ones(m); d[free] = bits
        # W[v] = sum of walks 0 -> ... -> v
        Wv = [adj[0, v + 1] * d[v] * (mats[0] @ mats[v + 1]) for v in range(m)]
        for _ in range(m - 1):
            Wv = [d[v] * (sum(Wv[u] for u in range(m) if adj[u + 1, v + 1]) @ mats[v + 1]) if adj[1:, v + 1].any()
                  else np.zeros_like(mats[0]) for v in range(m)]
        tot += d.prod() * sum(adj[v + 1, 0] * np.trace(Wv[v]) for v in range(m)); nterms += 1
    return tot / nterms, nterms

def ring_adj(n, kind, d):
    dist = np.array([[min(abs(i - j), n - abs(i - j)) for j in range(n)] for i in range(n)])
    A = (dist > 0) & (dist <= d)
    if kind == 'odd':
        A &= dist % 2 == 1
    return A

if __name__ == '__main__':
    rng = np.random.default_rng(1)
    # 1. classification on random graphs: is W always spanned by the cheap generators?
    stats = {}; mism = 0; ncase = 0
    for n in (4, 5, 6, 7, 8):
        for trial in range(250 if n < 8 else 80):
            p = rng.choice([0.4, 0.6, 0.8])
            adj = rand_adj(n, p, rng)
            if trial % 3 == 0:  # force bipartite graphs into the sample
                side = rng.random(n) < 0.5
                adj = adj & (side[:, None] != side[None, :])
            if trial % 7 == 1:  # G-0 bipartite, 0 joined to everything
                side = rng.random(n) < 0.5
                adj = adj & (side[:, None] != side[None, :]); adj[0, 1:] = True; adj[1:, 0] = True
            W, total = parity_space(adj)
            ham = brute([np.eye(1)] * n, adj)[1]
            dimW = len(W).bit_length() - 1
            assert len(W) == 1 << dimW
            cb, _ = gf2_basis(cheap_parity_space(adj))
            key = (n, 'ham>0' if ham else ('walks>0' if total else 'no walks'), dimW, len(cb))
            stats[key] = stats.get(key, 0) + 1
            ncase += 1
            if ham and dimW != len(cb): mism += 1
    print('random graphs: (n, case, dim W exact, dim of cheap generators) -> count')
    for k in sorted(stats): print('  ', k, stats[k])
    print('graphs with a Hamiltonian cycle where exact W is larger than the cheap generators: %d of %d' % (mism, ncase))

    # 2. ring families used by the approximations
    print('ring-distance families: dim W (exact)')
    for n in range(4, 11):
        row = []
        for kind in ('all', 'odd'):
            for d in range(1, n // 2 + 1):
                adj = ring_adj(n, kind, d)
                W, total = parity_space(adj)
                row.append('%s d<=%d:%d' % (kind, d, len(W).bit_length() - 1))
        print('  n=%2d ' % n, '  '.join(row))

    # 3. the reduced sum against brute force
    maxerr = 0.0; nchk = 0
    for sym in (True, False):
        for n in (4, 5, 6, 7, 8):
            for trial in range(6):
                mats = rand_mats(n, 6, 2, sym, rng)
                adj = rand_adj(n, 0.7, rng)
                if trial % 2 == 0:
                    side = np.arange(n) % 2 == 0; side = side[rng.permutation(n)]
                    adj = side[:, None] != side[None, :]
                    if trial % 4 == 0: adj = adj & (rand_adj(n, 0.85, rng))
                if trial == 5 and n % 2 == 1:
                    side = np.zeros(n, bool); side[1:1 + (n - 1) // 2] = True
                    adj = side[:, None] != side[None, :]; adj[0] = True; adj[:, 0] = True; adj[0, 0] = False
                ref, cnt = brute(mats, adj)
                W, _ = parity_space(adj)
                basis, piv = gf2_basis(W)
                val, nt = glynn_walk_reduced(mats, adj, piv)
                scale = max(abs(ref), 1e-300) if cnt else 1.0
                err = abs(val - ref) / scale if cnt else abs(val)
                maxerr = max(maxerr, err); nchk += 1
                if trial in (0, 5) or err > 1e-9:
                    print('  sym=%d n=%d perms=%5d dimW=%d terms=%4d (Glynn %4d) rel.err=%.2e' % (sym, n, cnt, len(piv), nt, 2 ** (n - 2), err))
    print('reduced-term restricted Glynn vs brute force: %d cases, max rel. error %.3e' % (nchk, maxerr))
