import pickle, sys, time, json, numpy as np
from tmat import bm_mod
S = eval(sys.argv[1]); M = int(sys.argv[2]); NV = int(sys.argv[3]); s = max(S)
D, trans, v0, w = pickle.load(open("T_%s.pkl" % "_".join(map(str, S)), "rb"))
fr = np.array([t[0] for t in trans]); to = np.array([t[1] for t in trans]); cc = np.array([t[2] for t in trans], dtype=np.int64)
t = time.time(); res = {}
for P in (2147483629, 2147483587):
    vec = np.array([x % P for x in v0], dtype=np.int64); ww = np.array([x % P for x in w], dtype=np.int64)
    seq = []
    for k in range(NV):
        seq.append(int((vec * ww % P).sum() % P))
        nv = np.zeros(D, dtype=np.int64); np.add.at(nv, to, cc * vec[fr] % P); vec = nv % P
    C, L = bm_mod(seq[:M], P); Cm = [c % P for c in C]
    ok = all(sum(Cm[i] * seq[k - i] for i in range(L + 1)) % P == 0 for k in range(L, len(seq)))
    print("P", P, "order", L, "verified on", NV, "terms:", ok, "%.1fs" % (time.time() - t), flush=True); res[P] = C
print("same", res[2147483629] == res[2147483587]); print("C =", res[2147483629])
json.dump({"S": S, "C": res[2147483629], "n0": 3 * s, "states": D, "certified": False, "verified_terms": NV}, open("rec_%s.json" % "_".join(map(str, S)), "w"))
