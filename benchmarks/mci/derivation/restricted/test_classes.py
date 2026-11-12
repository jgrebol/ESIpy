import numpy as np
from lib import *
rng = np.random.default_rng(777)
err = {}
def rec(name, val, ref, scale):
    err[name] = max(err.get(name, 0.0), abs(val - ref) / scale)
full = lambda n: ~np.eye(n, dtype=bool)

# --- bipartite sign reduction: any bipartite A, 2^(m-2) terms ---
nb = 0; bad = 0
for n in (4, 6, 8):
    for kind in ('psd', 'sym', 'nonsym'):
        for trial in range(4 if n < 8 else 2):
            q = int(rng.integers(4, 8)); r = int(rng.integers(1, 4))
            mats = rand_mats(n, q, r, kind, rng)
            if trial == 0: A = ring_adj(n, n, odd=True)            # complete bipartite (alg 3, large d)
            elif trial == 1: A = ring_adj(n, 3 if n > 6 else 1, odd=True)  # odd distances <= d
            else:                                                  # random balanced bipartite, random parts
                perm = rng.permutation(n); col = np.zeros(n, int); col[perm[:n // 2]] = 1
                A = (col[:, None] != col[None, :]) & rand_adj(n, 0.8, rng)
                A |= ring_adj(n, 1) & (col[:, None] != col[None, :])
            fx = bipartite_fixed(A)
            ref, cnt = brute(mats, A)
            scale = max(abs(brute(mats, full(n))[0]), 1e-300)
            if fx is None:
                assert cnt == 0; continue
            nb += 1
            assert len(list(deltas(n - 1, fx))) == 2 ** max(n - 3, 0)
            rec('bip dense  2^(m-2)', glynn_dense(mats, A, fx), ref, scale)
            rec('bip sparse 2^(m-2)', glynn_sparse(mats, A, fx)[0], ref, scale)
            if kind != 'nonsym': rec('bip mitm   2^(m-2)', glynn_mitm(mats, A, fx)[0], ref, scale)
print('bipartite cases', nb)
# necessity: two fixed signs on a NON-bipartite graph give a wrong value
n = 6; mats = rand_mats(n, 5, 2, 'psd', rng); A = full(n)
ref = brute(mats, A)[0]; wrong = glynn_dense(mats, A, (0, 1))
print('non-bipartite with 2 fixed signs: rel deviation %.2e (must be O(1))' % (abs(wrong - ref) / abs(ref)))

# --- class-structured transfer ---
nc = 0
for n in (4, 5, 6, 7, 8):
    for kind in ('psd', 'nonsym'):
        for trial in range(4 if n < 8 else 2):
            q = int(rng.integers(4, 8)); r = int(rng.integers(1, 4))
            mats = rand_mats(n, q, r, kind, rng)
            C = int(rng.integers(2, 5))
            cls = rng.integers(0, C, n)
            if trial == 0: Bc = ~np.eye(C, dtype=bool)                 # complete multipartite
            else:
                Bc = rand_adj(C, 0.7, rng); Bc[np.diag_indices(C)] = rng.random(C) < 0.5   # clique or independent classes
            A = Bc[cls[:, None], cls[None, :]] & full(n)
            ref, cnt = brute(mats, A); scale = max(abs(brute(mats, full(n))[0]), 1e-300)
            rec('class transfer', glynn_classes(mats, cls, Bc), ref, scale); nc += 1
print('class cases', nc)
for k, v in err.items(): print('%-22s max rel err %.2e' % (k, v))
