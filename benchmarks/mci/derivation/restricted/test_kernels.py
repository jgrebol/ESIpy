import numpy as np
from lib import *
from kernels import *
rng = np.random.default_rng(4)
e = {'exact_mitm': 0, 'mitm_dense': 0, 'mitm_sparse': 0}
for n in (4, 5, 6, 7, 8):
    for kind in ('psd', 'sym'):
        mats = rand_mats(n, 6, 2, kind, rng); m = n - 1; NT = 1 << (m - 1)
        full = ~np.eye(n, dtype=bool); ref = brute(mats, full)[0]
        e['exact_mitm'] = max(e['exact_mitm'], abs(kernel_exact_mitm(mats, 0, NT, 16) / NT - ref) / abs(ref))
        for A in (ring_adj(n, 2), rand_adj(n, 0.7, rng)):
            rA = brute(mats, A)[0]
            e['mitm_dense'] = max(e['mitm_dense'], abs(kernel_mitm_dense(mats, A, 0, NT, 16) / NT - rA) / abs(ref))
            e['mitm_sparse'] = max(e['mitm_sparse'], abs(kernel_mitm_sparse(mats, A, 0, NT, 16) / NT - rA) / abs(ref))
print(e)
