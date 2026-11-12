"""Single-core timings of the Glynn kernels on the cached annulenes: exact MCI and mcialg=5 at d=kmax."""
import sys, time, pickle
from esipy.mci import compute_mci, mci_approx

S = pickle.load(open("benchmarks/mci/cache/systems.pkl", "rb"))
names = sys.argv[1:] or ["annulene14", "annulene16", "annulene18", "annulene20", "annulene14_svp", "annulene16_svp"]
for name in names:
    s = S[name]
    ring = s["rings"]["perimeter"]
    n = len(ring)
    for part in s["parts"]:
        aom, connec = s["parts"][part]["aom"], s["parts"][part]["connec"]
        t = time.time(); v = compute_mci(ring, aom, part, n_cores=1); te = time.time() - t
        va, p, ta = mci_approx(ring, aom, part, alg=5, d=n // 2, n_cores=1, connec=connec)
        vb, pb, tb = mci_approx(ring, aom, part, alg=1, d=3, n_cores=1, connec=connec)
        print(f"{name:15s} {part:12s} K={aom[0].shape[0]:3d} exact {te:7.2f}s {v:+.12e} | alg5 kmax {ta:7.2f}s {va:+.12e} perms {p} "
              f"| alg1 d=3 {tb:7.2f}s perms {pb}", flush=True)
