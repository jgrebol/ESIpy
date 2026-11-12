"""Exact table: directed counts (Mulliken) for annulene C_n, algs 1..5, d=1..kmax, via Held-Karp
on the exact adjacency built with the ESIpy distance functions (mirrored in lib.py).
usage: python table.py n  -> writes table_n.json"""
import sys, json, time
from lib import ring_connec, adjs_for, hk_directed
n = int(sys.argv[1]); c = ring_connec(n); cache = {}; t = time.time()
def H(A):
    k = A.tobytes()
    if k not in cache: cache[k] = hk_directed(A)
    return cache[k]
res = {}
for alg in (1, 2, 3, 4, 5):
    for d in range(1, n // 2 + 1):
        res["%d,%d" % (alg, d)] = sum(s * H(A) for s, A in adjs_for(c, alg, d))
if n % 2:  # alg 5 beyond kmax (only-odd distances reach n-2 on odd rings)
    for d in range(n // 2 + 1, n - 1):
        res["5,%d" % d] = sum(s * H(A) for s, A in adjs_for(c, 5, d))
json.dump(res, open("table_%d.json" % n, "w"))
print(n, "HK calls", len(cache), "%.1fs" % (time.time() - t))
