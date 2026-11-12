import numpy as np, time
from lib import *
from krylov2 import glynn_krylov2
from esipy.mci import _kernel_glynn_exact, _project
rng = np.random.default_rng(2)
w = 0
for sym in (False, True):
    for n, q, r in [(4,6,2),(6,8,2),(8,10,2),(9,10,2),(9,5,5)]:
        mats = rand_mats(n, q, r, sym, rng); ref = brute(mats)
        e = abs(glynn_krylov2(mats, sym, low=3) / (1 << (n - 2)) - ref) / abs(ref); w = max(w, e)
print("krylov2 vs brute force, max rel err", w)
print("seconds per 1000 Glynn terms (1 thread)")
for n, q, r, sym, nt in [(14,49,5,True,4096),(14,49,5,False,4096),(12,180,14,True,512),(12,180,14,False,512),(16,49,5,True,4096),(10,180,14,True,256),(14,49,49,True,2048)]:
    mats = rand_mats(n, q, r, sym, rng)
    mats = _project(mats); p = mats[0].shape[0]; m = n - 1
    t = time.perf_counter(); a, _ = _kernel_glynn_exact((mats, None, 0, nt, 64)); tc = time.perf_counter() - t
    t = time.perf_counter(); b = glynn_krylov2(mats, sym, low=6, s0=0, s1=nt); tk = time.perf_counter() - t
    print(f"n={n} q={q}->p={p} r={r} sym={sym}: current {tc/nt*1e3:.4f}  krylov2 {tk/nt*1e3:.4f}  speedup {tc/tk:.1f}x  agree {abs(a-b)/abs(a):.1e}")
