"""
Is the failure of mcialg=5 on 4n annulenes physical or a geometry artefact?
  python benchmarks/mci/annulene_geom.py n1 n2 ...      (appends to annulene_geom.jsonl, resumable)

Builds each CnHn twice: equal C-C bonds (Dnh polygon) and alternating 1.35/1.47 A bonds (equiangular, D(n/2)h),
runs RHF/STO-3G, checks the external RHF->UHF stability and the HOMO-LUMO gap, and compares mcialg=5 at
d=kmax with the exact MCI (meta-Lowdin, Glynn). Odd n is skipped for the alternating geometry.
"""
import os, io, sys, json, time, contextlib
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(here, "annulene_geom.jsonl")
CH = 1.08


def polygon(n, a, b):
    """Equiangular polygon with sides a, b, a, b, ... (closes for even n); H outwards on the angle bisectors."""
    sides = [a if k % 2 == 0 else b for k in range(n)]
    steps = np.array([[np.cos(2 * np.pi * k / n), np.sin(2 * np.pi * k / n)] for k in range(n)]) * np.array(sides)[:, None]
    c = np.vstack([[0, 0], np.cumsum(steps, axis=0)[:-1]])
    c -= c.mean(axis=0)
    atoms = [("C", (x, y, 0.0)) for x, y in c]
    for i in range(n):
        out = 2 * c[i] - c[i - 1] - c[(i + 1) % n]
        h = c[i] + CH * out / np.linalg.norm(out)
        atoms.append(("H", (h[0], h[1], 0.0)))
    return atoms


def run(n, geom):
    from pyscf import gto, scf
    from esipy import ESI
    from esipy.mci import compute_mci, mci_approx
    atoms = polygon(n, 1.40, 1.40) if geom == "equal" else polygon(n, 1.35, 1.47)
    mol = gto.M(atom=atoms, basis="sto-3g", charge=n % 2, verbose=0)
    mf = scf.RHF(mol).run()
    # external stability: does an unrestricted solution lie lower?
    with contextlib.redirect_stdout(io.StringIO()):
        mo_i, mo_e, stable_i, stable_e = mf.stability(external=True, return_status=True)
    nocc = mol.nelectron // 2
    gap = 27.2114 * (mf.mo_energy[nocc] - mf.mo_energy[nocc - 1])
    with contextlib.redirect_stdout(io.StringIO()):
        esi = ESI(mol=mol, mf=mf, rings=list(range(1, n + 1)), partition="meta-lowdin")
        aom, connec = esi.aom, esi.connec
    ring = list(range(1, n + 1))
    t = time.time()
    exact = 2 * compute_mci(ring, aom, "meta-lowdin", n_cores=NCORES)
    te = time.time() - t
    v5, p5, t5 = mci_approx(ring, aom, "meta-lowdin", alg=5, d=n // 2, n_cores=NCORES, connec=connec)
    return dict(n=n, geom=geom, energy=mf.e_tot, converged=bool(mf.converged), stable_internal=bool(stable_i),
                stable_external=bool(stable_e), gap_eV=gap, exact=exact, exact_s=te, alg5=2 * v5, alg5_perms=int(p5), alg5_s=t5,
                alg5_percent=100 * v5 * 2 / exact)


if __name__ == "__main__":
    NCORES = int(os.environ.get("NCORES", "2"))
    done = {(json.loads(l)["n"], json.loads(l)["geom"]) for l in open(OUT)} if os.path.exists(OUT) else set()
    for n in [int(x) for x in sys.argv[1:]]:
        for geom in ("equal", "alternating"):
            if (n, geom) in done or (geom == "alternating" and n % 2):
                continue
            r = run(n, geom)
            open(OUT, "a").write(json.dumps(r) + "\n")
            print(f"n={n:2d} {geom:11s} E={r['energy']:.6f} stable(int/ext)={r['stable_internal']}/{r['stable_external']} gap={r['gap_eV']:.2f} eV "
                  f"exact={r['exact']:+.6e} ({r['exact_s']:.1f}s) alg5={r['alg5']:+.6e} ({r['alg5_s']:.1f}s) {r['alg5_percent']:9.4f} %", flush=True)
