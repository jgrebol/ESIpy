"""
mcialg=5 at d=kmax on even annulenes = complete bipartite graph (sides A, with atom 0, and B, k = n/2 atoms each).
  python benchmarks/mci/bipartite_check.py

Compares, on the cached RHF/STO-3G annulenes (meta-Lowdin, n <= 18) and on commuting matrices:
  enumeration   explicit sum over the k! (k-1)! alternating permutations (n <= 12)
  esipy         mci_approx (two-sided shortcut, 2^(n-3) terms)
  user          _glynn_bipartite of ESIpy-dev-mciaprox/esipy/glynn.py, Tr(S0 Y (ZY)^(k-1)), 2^(n-3) terms
  separated     the two sides as independent sums, Tr(S0 . Perm(B) . Perm(A')), 2^(k-1) + 2^(k-2) terms:
                same number of permutations, but B-block followed by A-block, not alternating
"""
import itertools, time, pickle, importlib.util, os
import numpy as np
from esipy import mci

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("user_glynn", "/home/joan/PycharmProjects/ESIpy-dev-mciaprox/esipy/glynn.py")
ug = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ug)


def perm_sum(ms):
    """Sum over the orderings of the product of ms (Glynn, 2^(len-1) terms)."""
    if len(ms) == 1:
        return ms[0]
    rest, tot = np.array(ms[1:]), 0
    for s in itertools.product([1, -1], repeat=len(ms) - 1):
        X = ms[0] + np.tensordot(np.array(s, float), rest, 1)
        tot = tot + np.prod(s) * np.linalg.matrix_power(X, len(ms))
    return tot / 2 ** (len(ms) - 1)


def methods(mats):
    n = len(mats)
    k = n // 2
    A, B = list(range(2, n, 2)), list(range(1, n, 2))     # A' (side of atom 0 without it) and B
    out = {}
    if n <= 12:
        t = time.time(); tot = 0.0
        for sb in itertools.permutations(B):
            for sa in itertools.permutations(A):
                seq = [0] + [x for pair in itertools.zip_longest(sb, sa) for x in pair if x is not None]
                tot += np.trace(np.linalg.multi_dot([mats[i] for i in seq]))
        out["enumeration"] = (tot, time.time() - t)
    idx = np.arange(n)
    D = np.minimum(abs(idx[:, None] - idx[None]), n - abs(idx[:, None] - idx[None]))
    t = time.time(); v, _ = mci._glynn_sum(mats, D % 2 == 1, 1, 64); out["esipy"] = (v, time.time() - t)
    t = time.time(); v, _ = ug._glynn_bipartite(mats, A, B, n_cores=1); out["user"] = (v, time.time() - t)
    if n <= 16:
        t = time.time()
        out["separated"] = (np.trace(mats[0] @ perm_sum([mats[b] for b in B]) @ perm_sum([mats[a] for a in A])), time.time() - t)
    return out


S = pickle.load(open(os.path.join(here, "cache", "systems.pkl"), "rb"))
rng = np.random.default_rng(0)
print(f"{'system':22s} {'n':>3s} {'Glynn terms':>11s} | " + " | ".join(f"{m:>26s}" for m in ("enumeration", "esipy", "user", "separated")))
for n in range(6, 19, 2):
    for label in ("annulene STO-3G", "commuting (diagonal)"):
        if label.startswith("annulene"):
            mats = mci._prep_matrices(S[f"annulene{n}"]["rings"]["perimeter"], S[f"annulene{n}"]["parts"]["meta-lowdin"]["aom"])
        else:
            mats = [np.diag(rng.random(8)) for _ in range(n)]
        res = methods(mats)
        ref = res.get("enumeration", res["esipy"])[0]
        cells = []
        for m in ("enumeration", "esipy", "user", "separated"):
            if m in res:
                v, t = res[m]
                cells.append(f"{v:+.6e} {t:7.3f}s {abs(v - ref) / abs(ref):7.0e}")
            else:
                cells.append(" " * 26)
        print(f"{label:22s} {n:3d} {2 ** (n - 3):11d} | " + " | ".join(cells), flush=True)
