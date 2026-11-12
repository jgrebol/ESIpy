import numpy as np
from perterm import *
import bench
from bench import mk, cur_exact
bench.rng = np.random.default_rng(11)
mats = mk(13, 49, 5, True)
m = 12
L0, sig0, _ = factor(mats[0], True)
L0l = L0.astype(np.longdouble); ld = [x.astype(np.longdouble) for x in mats]
tot = np.longdouble(0); mag = 0.0
for sidx in range(1 << (m - 1)):
    dl = [1] + [1 - 2 * ((sidx >> i) & 1) for i in range(m - 1)]
    X = sum(d * x for d, x in zip(dl, ld[1:]))
    V = L0l
    for _ in range(m // 2): V = X @ V
    t = ((V * sig0) * V).sum()
    tot += np.prod(dl) * t; mag = max(mag, abs(float(t)))
ref = float(tot / 2 ** (m - 1))
a, b, c = cur_exact(mats), glynn_fast(mats), glynn_fast(mats, sym=False)
print('ref %.6e  largest term / result = %.1e' % (ref, mag / abs(ref)))
print('rel err: current %.1e   new sym %.1e   new nonsym-path %.1e' % (abs(a - ref) / abs(ref), abs(b - ref) / abs(ref), abs(c - ref) / abs(ref)))
