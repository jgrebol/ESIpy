import numpy as np, itertools
from math import comb
# Nisan partial-derivative matrix of P = sum_sigma x_s1...x_sm (non-commuting): M_k[u, v] = coeff of word uv
for m in (3, 4, 5, 6, 7):
    out = []
    for k in range(1, m):
        rows = {}; cols = {}
        ent = []
        for p in itertools.permutations(range(m)):
            u, v = p[:k], p[k:]
            ent.append((rows.setdefault(u, len(rows)), cols.setdefault(v, len(cols))))
        M = np.zeros((len(rows), len(cols)))
        for i, j in ent: M[i, j] = 1          # rows/cols of non-injective words are zero, omitted
        out.append((np.linalg.matrix_rank(M), comb(m, k)))
    print("m =", m, "rank M_k vs C(m,k):", out, " Glynn terms 2^(m-1) =", 2 ** (m - 1))
# a product of linear forms has rank-1 M_k: check on a random product, m=4, k=2
rng = np.random.default_rng(0); m = 4
a = rng.standard_normal((m, m))          # form j = sum_i a[j,i] x_i
words = list(itertools.product(range(m), repeat=2))
M = np.array([[a[0, u[0]] * a[1, u[1]] * a[2, v[0]] * a[3, v[1]] for v in words] for u in words])
print("rank of M_2 for a product of 4 linear forms:", np.linalg.matrix_rank(M))
