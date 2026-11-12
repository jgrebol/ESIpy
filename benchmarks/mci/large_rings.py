"""
Large annulenes with Glynn: exact MCI, mcialg=5 at d=kmax, and (optionally) the mcialg=4 profile.
  python benchmarks/mci/large_rings.py NCORES n1 [n2 ...] [--profile]
Prints, per ring: value, time, number of permutations and number of Glynn terms. Appends JSON lines to large_rings.jsonl.
"""
import sys, os, json, time, pickle
from math import factorial
from esipy.mci import compute_mci, mci_approx, mci_profile

here = os.path.dirname(os.path.abspath(__file__))
S = pickle.load(open(os.path.join(here, "cache", "systems.pkl"), "rb"))
ncores = int(sys.argv[1])
sizes = [int(a) for a in sys.argv[2:] if a.isdigit()]
part = "meta-lowdin"
out = open(os.path.join(here, "large_rings.jsonl"), "a")


def emit(**kw):
    out.write(json.dumps(kw) + "\n")
    out.flush()
    print(kw, flush=True)


for n in sizes:
    s = S[f"annulene{n}"]
    ring, aom, connec = s["rings"]["perimeter"], s["parts"][part]["aom"], s["parts"][part]["connec"]
    K = aom[0].shape[0]
    va, pa, ta = mci_approx(ring, aom, part, alg=5, d=n // 2, n_cores=ncores, connec=connec)
    emit(n=n, K=K, cores=ncores, kind="alg5 kmax", value=va, seconds=ta, perms=pa, glynn_terms=2 ** (n - 3 if n % 2 == 0 else n - 2))
    t = time.time()
    v = compute_mci(ring, aom, part, n_cores=ncores)
    emit(n=n, K=K, cores=ncores, kind="exact", value=v, seconds=time.time() - t, perms=factorial(n - 1) // 2, glynn_terms=2 ** (n - 2),
         alg5_percent=100 * va / v)
    if "--profile" in sys.argv:
        prof = mci_profile(ring, aom, part, alg=4, n_cores=ncores, connec=connec)
        for d, contrib, cum, perms, sec in prof:
            emit(n=n, K=K, cores=ncores, kind="alg4 profile", d=d, value=contrib, cumulative=cum, perms=perms, seconds=sec,
                 percent_of_exact=100 * contrib / v)
        emit(n=n, kind="alg4 profile sum check", sum=sum(p[1] for p in prof), exact=v)
