"""Independent check with ESIpy's own Glynn counter (esipy.mci._glynn_sum returns (value, directed count))."""
import sys, json, numpy as np
sys.path.insert(0, "/home/joan/PycharmProjects/ESIpy-dev-mciaprox")
from esipy.mci import _glynn_sum
from lib import ring_connec, adjs_for
rng = np.random.default_rng(1)
for n, alg, d in ((10, 1, 2), (12, 1, 3), (11, 3, 5), (12, 3, 3), (13, 5, 6), (13, 4, 5), (14, 2, 4)):
    mats = [rng.standard_normal((2, 2)) for _ in range(n)]
    tot = 0
    for s, A in adjs_for(ring_connec(n), alg, d):
        r = _glynn_sum(mats, np.asarray(A, dtype=bool), 1, 64); tot += s * int(r[1])
    T = json.load(open("table_%d.json" % n))
    print(n, alg, d, "glynn", tot, "table", T["%d,%d" % (alg, d)], tot == T["%d,%d" % (alg, d)])
