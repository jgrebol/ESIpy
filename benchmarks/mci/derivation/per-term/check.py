import numpy as np, sys
from perterm import *
rng = np.random.default_rng(7)

def rand_mats(n, q, kind, r=None):
    out = []
    for _ in range(n):
        rr = r if r else q
        if kind == 'sym':      # symmetric PSD low rank
            B = rng.standard_normal((q, rr)); out.append(B @ B.T / q)
        elif kind == 'symind': # symmetric indefinite
            B = rng.standard_normal((q, rr)); out.append(B @ np.diag(rng.standard_normal(rr)) @ B.T / q)
        else:                  # non-symmetric low rank
            out.append(rng.standard_normal((q, rr)) @ rng.standard_normal((rr, q)) / q)
    return out

def rel(a, b): return abs(a - b) / max(abs(b), 1e-300)

worst = {}
def rec(tag, e): worst[tag] = max(worst.get(tag, 0), e)

# P1/P2/P3/hybrid: exact sum
for n in (3, 4, 5, 6, 7, 8, 9):
    for kind, q, r in (('gen', 6, None), ('gen', 9, 2), ('sym', 9, 2), ('sym', 6, None), ('symind', 8, 3)):
        mats = rand_mats(n, q, kind, r)
        ref = brute(mats)
        for s in range((n - 1).bit_length()):
            for tb in (0, 2, 6):
                rec('exact_%s' % ('sym' if kind != 'gen' else 'nonsym'), rel(glynn_fast(mats, s=s, tbits=tb), ref))
        rec('auto', rel(glynn_fast(mats), ref))
        # symmetric input forced through the non-symmetric path must agree too
        rec('sym_via_nonsym', rel(glynn_fast(mats, sym=False), ref))
        # P4: any fixed atom
        for f in range(n):
            rec('fixed_atom', rel(glynn_fast(mats, fixed=f), ref))
print('exact sum, max rel err:', worst)

# mixed ranks (the fixed atom is the rank-1 one)
mats = [rand_mats(1, 9, 'sym', rr)[0] for rr in (3, 3, 1, 2, 3, 2, 3, 3)]
print('mixed ranks sym  :', rel(glynn_fast(mats), brute(mats)))
mats = [rand_mats(1, 9, 'gen', rr)[0] for rr in (3, 3, 1, 2, 3, 2, 3, 3)]
print('mixed ranks nonsym:', rel(glynn_fast(mats), brute(mats)))

# natural orbitals: sqrt(occ) * A, with some zero occupations
e = 0
for n in (4, 6, 8):
    q = 8
    A = rand_mats(n, q, 'sym', 3)
    occ = rng.uniform(0.01, 2, q); occ[[1, 5]] = 0.0
    M = [np.sqrt(occ)[:, None] * a for a in A]
    Ms = symmetrize_no(A, occ)
    assert is_sym(Ms) and not is_sym(M)
    ref = brute(M)
    e = max(e, rel(glynn_fast(Ms), ref), rel(brute(Ms), ref))
print('NO symmetrisation, max rel err:', e)

# restricted sums
w = {}
for n in (4, 5, 6, 7, 8, 9):
    for kind, q, r in (('gen', 9, 2), ('sym', 9, 2), ('symind', 8, 3), ('gen', 5, None), ('sym', 5, None)):
        mats = rand_mats(n, q, kind, r)
        for trial in range(4):
            if trial == 0:   # ring, distance <= 2
                d = np.abs(np.arange(n)[:, None] - np.arange(n)); d = np.minimum(d, n - d); adj = (d > 0) & (d <= 2)
            elif trial == 1: # ring, odd distances
                d = np.abs(np.arange(n)[:, None] - np.arange(n)); d = np.minimum(d, n - d); adj = (d % 2 == 1) & (d <= 3)
            else:
                adj = rng.random((n, n)) < 0.7; adj = np.triu(adj, 1); adj = adj | adj.T
            ref = brute(mats, adj)
            if abs(ref) < 1e-12: continue
            for f in (None, 0, n - 1):
                key = 'walks_' + ('sym' if kind != 'gen' else 'nonsym')
                w[key] = max(w.get(key, 0), rel(walks_fast(mats, adj, fixed=f), ref))
            w['walks_sym_via_nonsym'] = max(w.get('walks_sym_via_nonsym', 0), rel(walks_fast(mats, adj, sym=False), ref))
print('restricted sum, max rel err:', w)
