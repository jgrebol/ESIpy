"""
Repair pass for rows written before the mcialg=5 fix: adds mcialg=5 at the odd-walk maximum distance (kmax_odd), with the
DFS check when affordable, and rewrites dataset_results.jsonl (backup in dataset_results.before_fix.jsonl).
"""
import os, json, shutil, time
from esipy.mci import find_node_distances_onlyodd, mci_approx, mci_approx_dfs
from dataset import load

DFS_LIMIT, NCORES = 4e5, 2


def run(d, ring, alg, dist, method):
    """Same as run_dataset.run: values as printed by ESIpy (restricted x2, alpha + beta)."""
    val, perms, t0 = 0.0, 0, time.time()
    spins = [(d["aom"][0], 1), (d["aom"][1], 1)] if d["kind"] == "unrest" else [(d["aom"], 2 if d["kind"] == "rest" else 1)]
    for aom, f in spins:
        v, p, _ = (mci_approx if method == "glynn" else mci_approx_dfs)(ring, aom, "iao", alg=alg, d=dist, n_cores=NCORES, connec=d["connec"])
        val, perms = val + f * v, perms + p
    return val, int(perms), time.time() - t0

here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "dataset_results.jsonl")
shutil.copy(path, os.path.join(here, "dataset_results.before_fix.jsonl"))
rows = [json.loads(l) for l in open(path)]
for r in rows:
    if "kmax_odd" in r or "exact" not in r:
        continue
    d = load(r["system"])
    ring = r["ring"]
    sub = {a: [b for b in d["connec"].get(a, []) if b in ring] for a in ring}
    kodd = int(max(v for a in ring for v in find_node_distances_onlyodd(sub)[a].values()))
    r["kmax_odd"] = kodd
    key = f"alg5_d{kodd}"
    if key not in r:
        v, p, t = run(d, ring, 5, kodd, "glynn")
        r[key] = {"value": v, "perms": p, "seconds": t}
        if p <= DFS_LIMIT:
            w, q, tq = run(d, ring, 5, kodd, "dfs")
            r[key].update(dfs_value=w, dfs_perms=q, dfs_seconds=tq, match=bool(abs(v - w) <= 1e-8 * max(abs(w), 1e-4) and p == q))
        print(r["system"], f"n={r['n']} kmax_odd={kodd}", f"alg5/exact {100 * v / r['exact']['value']:.3f} %", r[key].get("match"), flush=True)
with open(path, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("FIX_DONE", flush=True)
