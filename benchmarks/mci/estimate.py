"""
Cost estimates for systems too large to run here: time per Glynn term measured on synthetic AOM-like matrices
(symmetric, rank r = basis functions per atom, dimension q = occupied MOs) times the number of terms.
"""
import time
import numpy as np
from esipy import mci

rng = np.random.default_rng(1)


def aom_like(n, q, r):
    return [(lambda a: a @ a.T / q)(rng.normal(size=(q, r))) for _ in range(n)]


def per_term(n, q, r, restricted, nterms=512):
    mats = aom_like(n, q, r)
    m = n - 1
    idx = np.arange(n)
    D = np.minimum(abs(idx[:, None] - idx[None]), n - abs(idx[:, None] - idx[None]))
    if restricted:   # odd steps only (two-sided adjacency), ordered as _glynn_sum does
        adj = D % 2 == 1
        order = mci._two_sides(adj)
        mats, adj = [mats[i] for i in order], adj[np.ix_(order, order)]
        t = time.time(); mci._kernel_glynn_walks((mats, adj, 0, nterms, 64, True)); return (time.time() - t) / nterms
    t = time.time(); mci._kernel_glynn_exact((mats, None, 0, nterms, 64, True)); return (time.time() - t) / nterms


def fmt(s):
    for unit, f in (("years", 3.15e7), ("days", 86400), ("h", 3600), ("min", 60)):
        if s >= f:
            return f"{s / f:.1f} {unit}"
    return f"{s:.1f} s"


cases = [  # label, atoms in the circuit, occupied MOs, rank per atom
    ("coronene C24H12, perimeter (18)", 18, 78, 5),
    ("coronene, all 24 C", 24, 78, 5),
    ("[30]annulene", 30, 105, 5),
    ("[32]annulene", 32, 112, 5),
    ("circumcoronene C54H18, perimeter (30)", 30, 171, 5),
    ("circumcoronene, all 54 C", 54, 171, 5),
    ("C60, 20-atom circuit", 20, 180, 5),
    ("C60, 30-atom circuit", 30, 180, 5),
    ("C60, all 60 C", 60, 180, 5),
    ("[32]annulene, def2-SVP", 32, 112, 14),
]
print(f"{'system':40s} {'exact: terms':>13s} {'12 cores':>12s} | {'odd-only kmax: terms':>20s} {'12 cores':>12s}")
for label, n, q, r in cases:
    nt = 256 if n > 40 else 512
    te, ta = per_term(n, q, r, False, nt), per_term(n, q, r, True, nt)
    print(f"{label:40s} {2.0 ** (n - 2):13.2e} {fmt(te * 2.0 ** (n - 2) / 12):>12s} | {2.0 ** (n - 3):20.2e} {fmt(ta * 2.0 ** (n - 3) / 12):>12s}"
          f"   (per term {1e6 * te:.0f} / {1e6 * ta:.0f} us)", flush=True)
