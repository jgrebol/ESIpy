"""
Test systems for the MCI benchmarks: AOMs and connectivity from ESIpy (PySCF RHF).
  python benchmarks/mci/systems.py            builds benchmarks/mci/cache/systems.pkl

Annulenes CnHn (regular polygons, n odd as cations) sweep the ring size; the acenes add fused rings
(perimeters with bridge bonds); a UHF cation and a CASSCF(6,6) benzene cover the unrestricted and
natural-orbital cases.
"""
import os, io, pickle, contextlib
import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "systems.pkl")
A, CH = 1.40, 1.08


def annulene(n):
    r = A / (2 * np.sin(np.pi / n))
    ang = 2 * np.pi * np.arange(n) / n
    atoms = [("C", (r * np.cos(a), r * np.sin(a), 0.0)) for a in ang]
    atoms += [("H", ((r + CH) * np.cos(a), (r + CH) * np.sin(a), 0.0)) for a in ang]
    return atoms, {"perimeter": list(range(1, n + 1))}, n % 2


def acene(nr):
    cs = []
    for i in range(nr):
        for a in (90, 30, -30, -90, -150, 150):
            p = (round(i * np.sqrt(3) * A + A * np.cos(np.radians(a)), 6), round(A * np.sin(np.radians(a)), 6))
            if p not in cs:
                cs.append(p)
    cs = np.array(cs)
    atoms = [("C", (x, y, 0.0)) for x, y in cs]
    for i, c in enumerate(cs):
        nb = [j for j in range(len(cs)) if j != i and np.linalg.norm(cs[j] - c) < A + 0.1]
        if len(nb) == 2:
            v = c - cs[nb].mean(axis=0)
            h = c + CH * v / np.linalg.norm(v)
            atoms.append(("H", (h[0], h[1], 0.0)))

    def ring(k):  # perimeter of the first k hexagons, ordered around its centre
        idx = [i for i, c in enumerate(cs) if any(np.linalg.norm(c - (h * np.sqrt(3) * A, 0)) < A + 0.1 for h in range(k))]
        ctr = cs[idx].mean(axis=0)
        return [i + 1 for i in sorted(idx, key=lambda i: np.arctan2(*(cs[i] - ctr)[::-1]))]

    return atoms, {f"{k}ring": ring(k) for k in range(1, nr + 1)}, 0


def run(atoms, rings, charge, partitions, kind="rhf", basis="sto-3g"):
    from pyscf import gto, scf, mcscf
    from esipy import ESI
    mol = gto.M(atom=atoms, basis=basis, charge=charge, spin=1 if kind == "uhf" else 0, verbose=0)
    myhf = (scf.UHF(mol) if kind == "uhf" else scf.RHF(mol)).run()
    mf = myhf
    if kind == "cas":
        mf = mcscf.CASSCF(myhf, 6, 6)
        mf.verbose = 0
        mf.kernel()
    parts = {}
    for part in partitions:
        with contextlib.redirect_stdout(io.StringIO()):
            esi = ESI(mol=mol, mf=mf, myhf=myhf, rings=list(rings.values())[0], partition=part)
            parts[part] = {"aom": esi.aom, "connec": esi.connec}
    return {"rings": rings, "parts": parts, "kind": kind, "basis": basis, "converged": bool(myhf.converged)}


def build():
    parts = ["mulliken", "meta-lowdin"]
    systems = {}
    for n in list(range(6, 19)) + [20, 22, 24]:
        systems[f"annulene{n}"] = run(*annulene(n), parts)
    for n in (10, 12, 14, 16, 18):
        systems[f"annulene{n}_svp"] = run(*annulene(n), ["meta-lowdin"], basis="def2-svp")
    for nr in (2, 3, 4):
        systems[f"acene{nr}"] = run(*acene(nr), parts)
    at, rg, _ = annulene(6)
    systems["benzene_uhf_cation"] = run(at, rg, 1, parts, "uhf")
    systems["benzene_cas66"] = run(at, rg, 0, parts, "cas")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    pickle.dump(systems, open(CACHE, "wb"))
    for k, s in systems.items():
        print(k, s["kind"], s["basis"], "converged" if s["converged"] else "NOT CONVERGED", {r: len(v) for r, v in s["rings"].items()})


def load():
    return pickle.load(open(CACHE, "rb"))


def add_annulenes(sizes, basis="sto-3g"):
    """Adds CnHn rings to the cache (meta-Lowdin only), keeping what is already there."""
    systems = load()
    for n in sizes:
        key = f"annulene{n}" + ("" if basis == "sto-3g" else "_" + basis.replace("def2-", ""))
        if key not in systems:
            systems[key] = run(*annulene(n), ["meta-lowdin"], basis=basis)
            pickle.dump(systems, open(CACHE, "wb"))
            print(key, "converged" if systems[key]["converged"] else "NOT CONVERGED", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "add":
        add_annulenes([int(x) for x in sys.argv[2:]])
    else:
        build()
