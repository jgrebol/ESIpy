import sys, numpy as np
from time import perf_counter as pc
sys.path.insert(0, '.')
from lib import *
from kernels import *
import esipy.mci as M
rng = np.random.default_rng(1)
def best(f, rep=3):
    ts = []
    for _ in range(rep):
        t = pc(); v = f(); ts.append(pc() - t)
    return v, min(ts)
q, r = 49, 5
print('q=%d r=%d, symmetric PSD low-rank AOMs; time for NT Glynn terms, single thread' % (q, r))
for n in (12, 14):
    m = n - 1; NT = min(1 << (m - 1), 1024)
    mats = M._project(rand_mats(n, q, r, 'psd', rng))
    for d in (2, 3, 4, n // 2 - 1, n // 2):
        A = ring_adj(n, d)
        v0, t0 = best(lambda: M._kernel_glynn_lowrank((mats, A, 0, NT, 64))[0])
        v1, t1 = best(lambda: kernel_mitm_dense(mats, A, 0, NT, 64))
        v2, t2 = best(lambda: kernel_mitm_sparse(mats, A, 0, NT, 256))
        sc = abs(v0) + 1e-300
        print('n=%2d d=%d NT=%d | current %.3fs | mitm dense %.3fs (x%.2f) | mitm sparse+prune %.3fs (x%.2f) | rel diff %.1e %.1e'
              % (n, d, NT, t0, t1, t0 / t1, t2, t0 / t2, abs(v1 - v0) / sc, abs(v2 - v0) / sc))
    # complete graph: exact kernels
    v0, t0 = best(lambda: M._kernel_glynn_exact((mats, None, 0, NT, 64))[0])
    v1, t1 = best(lambda: kernel_exact_mitm(mats, 0, NT, 64))
    print('n=%2d complete NT=%d | current exact (squaring) %.3fs | vector mitm %.3fs (x%.2f) | rel diff %.1e'
          % (n, NT, t0, t1, t0 / t1, abs(v1 - v0) / abs(v0)))
