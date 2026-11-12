"""
Runs the MCIAPROX dataset with the Glynn-based algorithms and cross-checks with the DFS where it is affordable.
  python benchmarks/mci/run_dataset.py NCORES FOLDER [FOLDER ...]      (ANNULENES DATASET_S DATASET_T FULLERENES SURFACES)
Appends one JSON line per ring to dataset_results.jsonl (systems already there are skipped, so it can be resumed).

Per ring of n atoms:  exact (n <= 26), mcialg=3 and 5 at d = kmax, the mcialg=4 profile (n <= 20);
cage sums (n >= 20 in FULLERENES) only mcialg=5 at d=1, as in the workbook. DFS on the same AOMs when the number
of permutations is below DFS_LIMIT. Values follow ESIpy's output (restricted x2, alpha + beta for unrestricted).
"""
import sys, os, json, time, glob
from math import factorial
import numpy as np
from esipy.mci import mci_approx, mci_approx_dfs, mci_profile, find_node_distances, find_node_distances_onlyodd
from dataset import load, GEOMS

DFS_LIMIT, EXACT_MAX, PROFILE_MAX = 4e5, 26, 20
here = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(here, "dataset_results.jsonl")
ncores = int(sys.argv[1])
done = {json.loads(l)["key"] for l in open(OUT)} if os.path.exists(OUT) else set()


def spins(d):
    """(AOMs, factor) per spin channel, as ESI.mciaprox combines them."""
    if d["kind"] == "unrest":
        return [(d["aom"][0], 1), (d["aom"][1], 1)]
    return [(d["aom"], 2 if d["kind"] == "rest" else 1)]


def run(d, ring, alg, dist, method):
    val, perms, t0 = 0.0, 0, time.time()
    for aom, f in spins(d):
        v, p, _ = (mci_approx if method == "glynn" else mci_approx_dfs)(ring, aom, "iao", alg=alg, d=dist, n_cores=ncores, connec=d["connec"])
        val, perms = val + f * v, perms + p
    return val, int(perms), time.time() - t0


for folder in sys.argv[2:]:
    for hs in sorted(glob.glob(os.path.join(GEOMS, folder, "*.hs"))):
        rel = os.path.join(folder, os.path.basename(hs)[:-3])
        if not os.path.exists(os.path.join(GEOMS, rel + ".fchk")):
            continue
        try:
            d = load(rel)
        except Exception as e:
            print(rel, "could not be loaded:", repr(e)[:200], flush=True)
            continue
        for ir, ring in enumerate(d["rings"]):
            key = f"{rel}#{ir}"
            n = len(ring)
            if key in done or n < 3:
                continue
            row = {"key": key, "system": rel, "ring": ring, "n": n, "kind": d["kind"], "K": int((d["aom"][0][0] if d["kind"] != "rest" else d["aom"][0]).shape[0]), "cores": ncores}
            sub = {a: [b for b in d["connec"].get(a, []) if b in ring] for a in ring}
            dist = find_node_distances(sub)
            kmax = max(v for a in ring for v in dist[a].values())
            row["kmax"] = int(kmax)
            # mcialg=5 uses the odd-walk distances, whose maximum can exceed the plain one (odd and non-alternant rings)
            kodd = max(v for a in ring for v in find_node_distances_onlyodd(sub)[a].values())
            row["kmax_odd"] = int(kodd)
            cage = folder == "FULLERENES" and n >= 20
            for alg, dd in ([(5, 1)] if cage else [(0, 1), (3, kmax), (5, kodd)]):
                if alg == 0 and n > EXACT_MAX:
                    continue
                name = "exact" if alg == 0 else f"alg{alg}_d{dd}"
                v, p, t = run(d, ring, alg, dd, "glynn")
                row[name] = {"value": v, "perms": p, "seconds": t}
                if (factorial(n - 1) if alg == 0 else p) <= DFS_LIMIT:
                    w, q, tq = run(d, ring, alg, dd, "dfs")
                    row[name].update(dfs_value=w, dfs_perms=q, dfs_seconds=tq,
                                     match=bool(abs(v - w) <= 1e-8 * max(abs(w), 1e-4) and p == q))
            if not cage and n <= PROFILE_MAX:
                shells = None
                for aom, f in spins(d):
                    prof = mci_profile(ring, aom, "iao", alg=4, n_cores=ncores, connec=d["connec"])
                    vals = np.array([f * x[1] for x in prof])
                    shells = vals if shells is None else shells + vals
                row["alg4_shells"] = {"d": [x[0] for x in prof], "values": shells.tolist(), "sum": float(shells.sum())}
            open(OUT, "a").write(json.dumps(row) + "\n")
            flags = [k + ("=" if row[k].get("match", None) else "!" if "match" in row[k] else "?") for k in row if isinstance(row[k], dict) and "value" in row[k]]
            print(key, f"n={n}", d["kind"], " ".join(flags), flush=True)
