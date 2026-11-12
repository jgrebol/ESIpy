"""Recurrence for H(C_n(S)) certified mod 2 primes: BM on M terms, then verify on >= D+L terms."""
from tmat import *; import time, sys, json, numpy as np
S = eval(sys.argv[1]); M = int(sys.argv[2]); s = max(S)
t = time.time(); keys, trans, v0, w = build(S); D = len(keys)
print(S, "states", D, "build %.1fs" % (time.time() - t), flush=True)
res = {}
for P in (2147483629, 2147483587):
    a, _ = sequence_mod(S, 3 * s + D + M + 10, P)
    seq = [a[n] for n in sorted(a)]
    C, L = bm_mod(seq[:M], P)
    Cm = [c % P for c in C]
    ok = all(sum(Cm[i] * seq[k - i] for i in range(L + 1)) % P == 0 for k in range(L, len(seq)))
    print("P", P, "order", L, "verified on", len(seq), "terms (need D+L =", D + L, "):", ok, "%.1fs" % (time.time() - t), flush=True)
    res[P] = C
print("same", res[2147483629] == res[2147483587])
print("C =", res[2147483629])
json.dump({"S": S, "C": res[2147483629], "n0": 3 * s, "states": D}, open("rec_%s.json" % "_".join(map(str, S)), "w"))
