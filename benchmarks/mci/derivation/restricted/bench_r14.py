import sys, numpy as np
from time import perf_counter as pc
from lib import *
from kernels import *
import esipy.mci as M
rng = np.random.default_rng(2)
def best(f, rep=2):
    ts = []
    for _ in range(rep):
        t = pc(); v = f(); ts.append(pc() - t)
    return v, min(ts)
q, r, n = 49, 14, 14
m = n - 1; NT = 256
mats = M._project(rand_mats(n, q, r, 'psd', rng))
print('q=%d r=%d n=%d NT=%d' % (q, r, n, NT))
for d in (2, 3, 4, 6):
    A = ring_adj(n, d)
    v0, t0 = best(lambda: M._kernel_glynn_lowrank((mats, A, 0, NT, 32))[0])
    v1, t1 = best(lambda: kernel_mitm_dense(mats, A, 0, NT, 32))
    v2, t2 = best(lambda: kernel_mitm_sparse(mats, A, 0, NT, 128))
    sc = abs(v0)
    print('d=%d | current %.3fs | mitm dense %.3fs (x%.2f) | mitm sparse+prune %.3fs (x%.2f) | rel diff %.1e %.1e'
          % (d, t0, t1, t0 / t1, t2, t0 / t2, abs(v1 - v0) / sc, abs(v2 - v0) / sc))
