"""
Rounding error of the Glynn sums: the MCI does not depend on which atom is first, so the spread over rotations and the
reversal of the ring measures it. Scan annulenes (same SCF as scan_alg5.py).
  NCORES=2 python benchmarks/mci/accuracy.py n1 n2 ...      -> accuracy.jsonl
"""
import os, io, sys, json, time, contextlib
import numpy as np
from scan_alg5 import scf

here = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(here, "accuracy.jsonl")
NCORES = int(os.environ.get("NCORES", "2"))

if __name__ == "__main__":
    from esipy import ESI
    from esipy.mci import compute_mci, mci_approx
    for n in [int(x) for x in sys.argv[1:]]:
        mol, mf, _, _ = scf(n)
        with contextlib.redirect_stdout(io.StringIO()):
            esi = ESI(mol=mol, mf=mf, rings=list(range(1, n + 1)), partition="meta-lowdin")
            aom, connec = esi.aom, esi.connec
        ring = list(range(1, n + 1))
        orders = [ring, ring[1:] + ring[:1], ring[n // 3:] + ring[:n // 3], ring[::-1]]
        t = time.time()
        ex = [2 * compute_mci(o, aom, "meta-lowdin", n_cores=NCORES) for o in orders]
        a5 = [2 * mci_approx(o, aom, "meta-lowdin", alg=5, d=n // 2, n_cores=NCORES, connec=connec)[0] for o in orders]
        row = dict(n=n, exact=ex, alg5=a5, exact_spread=float(np.ptp(ex)), alg5_spread=float(np.ptp(a5)),
                   exact_rel=float(np.ptp(ex) / abs(np.mean(ex))), alg5_rel=float(np.ptp(a5) / abs(np.mean(a5))), seconds=time.time() - t)
        open(OUT, "a").write(json.dumps(row) + "\n")
        print(f"n={n}: exact {np.mean(ex):+.12e} spread {row['exact_spread']:.1e} (rel {row['exact_rel']:.1e}) | "
              f"alg5 {np.mean(a5):+.12e} spread {row['alg5_spread']:.1e} (rel {row['alg5_rel']:.1e})", flush=True)
