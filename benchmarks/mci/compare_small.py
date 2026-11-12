"""
DFS vs Glynn on the annulenes C6H6, C8H8, C10H10, C12H12 of the MCIAPROX dataset (IAO, from the .fchk files).
  python benchmarks/mci/compare_small.py
For every algorithm (exact, mcialg 1-5 at each d, mcialg 6) prints the MCI, the number of evaluations
(permutations for the DFS, Glynn terms or cycles for the Glynn-based method) and the time of each method.
Values as printed by ESIpy (restricted AOMs x2).
"""
import numpy as np
from esipy.mci import mci_approx, mci_approx_dfs, _two_sides, find_node_distances, find_node_distances_onlyodd
from dataset import load

SYSTEMS = ["ANNULENES/c6h6", "ANNULENES/c8h8_d4h", "ANNULENES/c10h10_d10h", "ANNULENES/c12h12_d6h"]


def evaluations(ring, connec, alg, d, perms):
    """
    What the Glynn-based method evaluates, with the same choice as mci_approx: the cycles themselves when the
    adjacency is sparse (pruned search), otherwise 2^(n-2) Glynn terms (2^(n-3) when the steps join two equal sides).
    """
    n = len(ring)
    if alg == 0 or alg == 6:
        return 2 ** (n - 2)
    sub = {a: [b for b in connec.get(a, []) if b in ring] for a in ring}
    dist = (find_node_distances_onlyodd if alg in (4, 5) else find_node_distances)(sub)
    D = np.array([[dist[a].get(b, 999999) for b in ring] for a in ring])
    np.fill_diagonal(D, 999999)
    adj = (D <= d) & ((D % 2 == 1) if alg in (3, 5) else True)
    if adj.sum() <= 4 * n:
        return f"{perms} cycles"
    return 2 ** (n - 3) if _two_sides(adj) is not None else 2 ** (n - 2)


print(f"{'system':10s} {'alg':>4s} {'d':>3s} | {'MCI DFS':>15s} {'perms':>10s} {'time':>8s} | {'MCI Glynn':>15s} {'evaluations':>12s} {'time':>8s} | {'|diff|':>8s}")
for rel in SYSTEMS:
    data = load(rel)
    ring, aom, connec = data["rings"][0], data["aom"], data["connec"]
    f = 2 if data["kind"] == "rest" else 1
    n = len(ring)
    cases = [(0, 0, {})] + [(a, d, {}) for a in (1, 2, 3, 4, 5) for d in range(1, n // 2 + 1)] + [(6, 0, {"delta": 1e-4})]
    for alg, d, kw in cases:
        v, p, tv = mci_approx_dfs(ring, aom, "iao", alg=alg, d=d, n_cores=1, connec=connec, **kw)
        g, q, tg = mci_approx(ring, aom, "iao", alg=alg, d=d, n_cores=1, connec=connec, **kw)
        ev = evaluations(ring, connec, alg, d, int(q))
        print(f"{rel.split('/')[1][:10]:10s} {alg:4d} {d:3d} | {f * v:+15.8e} {int(p):10d} {tv:8.4f} | {f * g:+15.8e} {str(ev):>12s} {tg:8.4f} | "
              f"{abs(f * (v - g)):8.1e}{'' if int(p) == int(q) else '  PERMS DIFFER'}", flush=True)
