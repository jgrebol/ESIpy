import numpy as np
from itertools import permutations
from lib import *
rng = np.random.default_rng(99)
worst = 0.0
for n, kind in ((7, 'psd'), (8, 'psd'), (7, 'nonsym'), (8, 'sym')):
    mats = rand_mats(n, 6, 2, kind, rng)
    i = np.arange(n); dist = np.abs(i[:, None] - i[None, :]); dist = np.minimum(dist, n - dist)
    kmax = n // 2
    exact = np.zeros(kmax + 1)                  # brute force: sum over permutations whose max step is exactly d
    for p in permutations(range(1, n)):
        cyc = (0,) + p + (0,)
        dm = max(dist[cyc[j], cyc[j + 1]] for j in range(n))
        P = mats[0]
        for v in p: P = P @ mats[v]
        exact[dm] += np.trace(P)
    cum = [0.0]
    for d in range(1, kmax + 1):               # kmax cumulative runs (mitm when symmetric), shared setup
        A = ring_adj(n, d)
        cum.append(glynn_mitm(mats, A)[0] if kind != 'nonsym' else glynn_sparse(mats, A)[0])
    prof = np.diff(cum)
    e = np.abs(prof - exact[1:]).max() / np.abs(exact).sum()
    worst = max(worst, e)
    print(n, kind, 'profile max rel err %.2e' % e)
print('worst %.2e' % worst)
