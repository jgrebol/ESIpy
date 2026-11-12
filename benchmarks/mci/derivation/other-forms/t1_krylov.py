import numpy as np, time, sys
from lib import *
from esipy.mci import _kernel_glynn_exact, _project
rng = np.random.default_rng(1)
worst = 0
for sym in (False, True):
    for n, q, r in [(4,6,2),(5,7,3),(6,8,2),(7,9,3),(8,10,2),(8,6,4),(9,10,2),(9,5,5)]:
        mats = rand_mats(n, q, r, sym, rng)
        ref = brute(mats)
        e1 = abs(glynn_krylov(mats, sym) - ref) / abs(ref)
        e2 = abs(glynn_gram(mats, sym) - ref) / abs(ref)
        e3 = abs(glynn_krylov(mats, False) - ref) / abs(ref)
        worst = max(worst, e1, e2, e3)
        print(f"sym={sym} n={n} q={q} r={r} ref={ref:+.6e} krylov {e1:.1e} gram {e2:.1e} krylov-nonsymcode {e3:.1e}")
# AOM-like: partition of identity + Hueckel
for n in (6, 8, 9):
    mats = partition_of_identity(n + 4, 12, 3, rng)[:n]; ref = brute(mats)
    e = max(abs(glynn_krylov(mats, True) - ref), abs(glynn_gram(mats, True) - ref)) / abs(ref); worst = max(worst, e)
    print(f"partition-of-identity n={n} ref={ref:+.6e} err {e:.1e}")
for n in (6, 8, 9):
    mats = huckel(n); ref = brute(mats)
    e = max(abs(glynn_krylov(mats, True) - ref), abs(glynn_gram(mats, True) - ref)) / abs(ref); worst = max(worst, e)
    print(f"hueckel n={n} ref={ref:+.6e} err {e:.1e}")
print("MAX REL ERR", worst)

# benchmark vs current kernel (single thread, a slice of the terms)
print("\nbenchmark: seconds per 1000 Glynn terms")
for n, q, r, sym, nt in [(14,49,5,True,2048),(14,49,5,False,2048),(12,180,14,True,256),(12,180,14,False,256),(16,49,5,True,2048),(10,180,14,True,256)]:
    mats = rand_mats(n, q, r, sym, rng)
    mats = _project(mats); p = mats[0].shape[0]; m = n - 1
    t = time.perf_counter(); a, _ = _kernel_glynn_exact((mats, None, 0, nt, 64)); tc = time.perf_counter() - t
    t = time.perf_counter(); b = glynn_krylov(mats, sym, batch=64, s0=0, s1=nt) * (1 << (m - 1)); tk = time.perf_counter() - t
    print(f"n={n} q={q}->p={p} r={r} sym={sym}: current {tc/nt*1e3:.4f}  krylov {tk/nt*1e3:.4f}  speedup {tc/tk:.1f}x  agree {abs(a-b)/abs(a):.1e}")
