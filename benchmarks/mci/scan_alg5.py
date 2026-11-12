"""
Scan of CnHn annulenes: how far mcialg=5 at d=kmax can go, validated against the exact MCI while it is affordable.
  NCORES=8 python benchmarks/mci/scan_alg5.py [n_start] [n_stop] [max hours per ring]
Appends one JSON line per ring to scan_alg5.jsonl (resumable: finished rings are skipped).

Geometry: 4n+2 rings with equal C-C bonds (1.40 A), 4n rings bond-alternated (1.35/1.47 A, equiangular), both planar.
RHF/STO-3G, followed along internal instabilities until stable; meta-Lowdin AOMs. The exact MCI is computed while
n <= EXACT_MAX; the scan stops when the next ring is predicted to take longer than the time limit.
"""
import os, io, sys, json, time, contextlib
import numpy as np
from annulene_geom import polygon

here = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(here, "scan_alg5.jsonl")
NCORES = int(os.environ.get("NCORES", "8"))
EXACT_MAX = int(os.environ.get("EXACT_MAX", "26"))


def scf(n):
    from pyscf import gto, scf as pscf
    atoms = polygon(n, 1.40, 1.40) if n % 4 == 2 else polygon(n, 1.35, 1.47)
    mol = gto.M(atom=atoms, basis="sto-3g", verbose=0)
    mf = pscf.RHF(mol)
    mf.max_cycle = 200
    mf.kernel()
    for _ in range(3):  # follow internal instabilities down to a stable RHF solution
        with contextlib.redirect_stdout(io.StringIO()):
            mo, _, stable, _ = mf.stability(return_status=True)
        if stable:
            break
        dm = mf.make_rdm1(mo, mf.mo_occ)
        mf.kernel(dm)
    nocc = mol.nelectron // 2
    return mol, mf, bool(stable), 27.2114 * (mf.mo_energy[nocc] - mf.mo_energy[nocc - 1])


def main():
    from esipy import ESI
    from esipy.mci import compute_mci, mci_approx
    n0 = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    n1 = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    hours = float(sys.argv[3]) if len(sys.argv) > 3 else 24
    rows = [json.loads(l) for l in open(OUT)] if os.path.exists(OUT) else []
    done = {r["n"] for r in rows}
    last = {}
    for r in rows:
        last[r["n"]] = r["alg5_s"]
    for n in range(n0, n1 + 1, 2):
        if n in done:
            continue
        # predicted time from the growth of the two previous rings
        prev = sorted(k for k in last if k < n)
        if len(prev) >= 2 and last[prev[-1]] > 1:
            growth = last[prev[-1]] / max(last[prev[-2]], 1e-3)
            pred = last[prev[-1]] * growth ** ((n - prev[-1]) / (prev[-1] - prev[-2]))
            if pred > hours * 3600:
                print(f"stop: n={n} predicted {pred / 3600:.1f} h > {hours} h", flush=True)
                return
        t = time.time()
        mol, mf, stable, gap = scf(n)
        ts = time.time() - t
        with contextlib.redirect_stdout(io.StringIO()):
            esi = ESI(mol=mol, mf=mf, rings=list(range(1, n + 1)), partition="meta-lowdin")
            aom, connec = esi.aom, esi.connec
        ring = list(range(1, n + 1))
        v5, p5, t5 = mci_approx(ring, aom, "meta-lowdin", alg=5, d=n // 2, n_cores=NCORES, connec=connec)
        row = dict(n=n, geometry="equal" if n % 4 == 2 else "alternating", energy=mf.e_tot, converged=bool(mf.converged),
                   stable=stable, gap_eV=gap, scf_s=ts, nocc=mol.nelectron // 2, cores=NCORES,
                   alg5=2 * v5, alg5_perms=int(p5), alg5_terms=2 ** (n - 3), alg5_s=t5)
        if n <= EXACT_MAX:
            t = time.time()
            ex = 2 * compute_mci(ring, aom, "meta-lowdin", n_cores=NCORES)
            row.update(exact=ex, exact_terms=2 ** (n - 2), exact_s=time.time() - t, alg5_percent=100 * v5 * 2 / ex)
        open(OUT, "a").write(json.dumps(row) + "\n")
        last[n] = t5
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
