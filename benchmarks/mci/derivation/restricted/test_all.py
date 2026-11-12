import numpy as np, sys
from lib import *
rng = np.random.default_rng(12345)
err = {}
def rec(name, val, ref, scale):
    e = abs(val - ref) / scale
    err[name] = max(err.get(name, 0.0), e)

ncase = 0; nz = 0
for n in (4, 5, 6, 7, 8):
    for kind in ('psd', 'sym', 'nonsym'):
        for trial in range(4 if n < 8 else 2):
            q = int(rng.integers(4, 8)); r = int(rng.integers(1, 4))
            mats = rand_mats(n, q, r, kind, rng)
            if trial % 4 == 0: A = ring_adj(n, max(1, n // 2 - 1))
            elif trial % 4 == 1: A = ring_adj(n, 2)
            else: A = rand_adj(n, [0.5, 0.75][trial % 2], rng)
            ref, cnt = brute(mats, A)
            # scale: sum of |Tr| over ALL permutations of the complete graph (natural magnitude)
            scale = max(abs(brute(mats, ~np.eye(n, dtype=bool))[0]), abs(ref), 1e-300)
            ncase += 1; nz += cnt > 0
            rec('dense(current)', glynn_dense(mats, A), ref, scale)
            v, nb = glynn_sparse(mats, A); rec('sparse+prune', v, ref, scale)
            rec('sparse noprune', glynn_sparse(mats, A, prune=False)[0], ref, scale)
            rec('through-q+selfloops', glynn_throughq(mats, A), ref, scale)
            rec('through-q', glynn_throughq(mats, A, selfloops=False), ref, scale)
            if kind != 'nonsym':
                v, npd = glynn_mitm(mats, A); rec('mitm', v, ref, scale)
                assert npd == (n - 1) // 2, (npd, n)
            if n <= 7:
                v, nt, nf = glynn_complement(mats, A); rec('complement IE', v, ref, scale)
print('cases', ncase, 'with >=1 allowed permutation', nz)
for k, v in err.items(): print('%-22s max rel err %.2e' % (k, v))
