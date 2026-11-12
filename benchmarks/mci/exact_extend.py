"""
Exact MCI for the scan annulenes beyond EXACT_MAX (same geometry and SCF as scan_alg5.py).
  NCORES=8 python benchmarks/mci/exact_extend.py n1 [n2 ...]     -> appends to scan_exact.jsonl
"""
import os, io, sys, json, time, contextlib
from scan_alg5 import scf

here = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(here, "scan_exact.jsonl")
NCORES = int(os.environ.get("NCORES", "8"))

if __name__ == "__main__":
    from esipy import ESI
    from esipy.mci import compute_mci
    done = {json.loads(l)["n"] for l in open(OUT)} if os.path.exists(OUT) else set()
    for n in [int(x) for x in sys.argv[1:]]:
        if n in done:
            continue
        mol, mf, stable, gap = scf(n)
        with contextlib.redirect_stdout(io.StringIO()):
            aom = ESI(mol=mol, mf=mf, rings=list(range(1, n + 1)), partition="meta-lowdin").aom
        t = time.time()
        ex = 2 * compute_mci(list(range(1, n + 1)), aom, "meta-lowdin", n_cores=NCORES)
        row = dict(n=n, energy=mf.e_tot, stable=stable, gap_eV=gap, cores=NCORES, exact=ex, exact_terms=2 ** (n - 2), exact_s=time.time() - t)
        open(OUT, "a").write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
