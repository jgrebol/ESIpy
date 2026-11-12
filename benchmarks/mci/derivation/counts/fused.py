"""Naphthalene 10-perimeter with the 1-6 bridge versus [10]annulene."""
import json
from lib import adjs_for, hk_directed, dist_matrix, ring_connec, dfs_count
naph = {1: [2, 10, 6], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4, 6], 6: [5, 1, 7], 7: [6, 8], 8: [7, 9], 9: [8, 10], 10: [9, 1]}
D = dist_matrix(naph); Dr = dist_matrix(ring_connec(10))
import numpy as np
np.fill_diagonal(D, 0); np.fill_diagonal(Dr, 0)
print("max distance naphthalene", D.max(), " annulene", Dr.max())
print("distance histogram (pairs) naph:", {int(k): int((D[np.triu_indices(10, 1)] == k).sum()) for k in range(1, 6)},
      " annulene:", {int(k): int((Dr[np.triu_indices(10, 1)] == k).sum()) for k in range(1, 6)})
out = {}
for alg in (1, 2, 3, 4, 5):
    for d in range(1, 6):
        a = sum(s * hk_directed(A) for s, A in adjs_for(naph, alg, d))
        assert a == dfs_count(naph, alg, d)
        r = json.load(open("table_10.json"))["%d,%d" % (alg, d)]
        out["%d,%d" % (alg, d)] = (a, r)
        print("alg %d d=%d  naphthalene %7d   annulene %7d" % (alg, d, a, r))
json.dump(out, open("fused_naphthalene.json", "w"))
