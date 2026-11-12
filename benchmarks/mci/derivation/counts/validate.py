"""Cross-check every formula against the exact Held-Karp tables (table_n.json, n=6..20)."""
import json
from math import factorial as f
from tm import ham_count, ham_complement
T = {n: json.load(open("table_%d.json" % n)) for n in range(6, 21)}
g = lambda n, a, d: T[n]["%d,%d" % (a, d)]
nchk = 0
def ok(c, msg):
    global nchk; nchk += 1
    assert c, msg
# Narayana power sums for d=2 closed form
N = [3, 1, 1]
for i in range(3, 60): N.append(N[-1] + N[-3])
import pickle
G = json.load(open("growth.json"))
CIRC = {}
for key in G:
    S = tuple(json.loads(key)); CIRC[S] = {int(a): b for a, b in G[key]["seq"].items()}
D5, tr5, v05, w5 = pickle.load(open("T_1_3_5.pkl", "rb")); vec = list(v05); CIRC[(1, 3, 5)] = {}
for nn in range(15, 21):
    CIRC[(1, 3, 5)][nn] = sum(a * b for a, b in zip(vec, w5)); nv = [0] * D5
    for a_, b_, c_ in tr5: nv[b_] += c_ * vec[a_]
    vec = nv
def hc(n, S):
    S = tuple(S)
    if S in CIRC and n in CIRC[S]: return CIRC[S][n]
    if max(S) <= 3: return ham_count(n, S)
    return None
recs = {k: json.load(open("rec_%s.json" % k))["C"] for k in ("1_2", "1_2_3", "1_3", "1_2_3_4")}
for n in range(6, 21):
    k = n // 2
    for d in range(1, k + 1):
        S1 = list(range(1, d + 1)); So = list(range(1, d + 1, 2))
        # alg1 = Ham cycles of circulant C_n(1..d) (complete graph when d = kmax)
        if 2 * d >= n - 1: ok(g(n, 1, d) == f(n - 1), (n, 1, d))
        elif hc(n, S1) is not None: ok(g(n, 1, d) == hc(n, S1), (n, 1, d))
        # alg2 = alg1(d) - alg1(d-1)
        ok(g(n, 2, d) == g(n, 1, d) - (g(n, 1, d - 1) if d > 1 else 0), (n, 2, d))
        # alg3 = circulant with odd offsets
        if 2 * max(So) < n and hc(n, So) is not None: ok(g(n, 3, d) == hc(n, So), (n, 3, d))
        if n % 2 == 0:
            ok(g(n, 4, d) == g(n, 2, d) and g(n, 5, d) == g(n, 3, d), (n, "even:alg4=alg2,alg5=alg3", d))
        else:
            ok(g(n, 5, d) == g(n, 3, d), (n, "odd: alg5=alg3 for d<=kmax", d))
            ok(g(n, 4, d) == (0 if d % 2 == 0 else g(n, 5, d) - (g(n, 5, d - 2) if d > 2 else 0)), (n, "odd alg4", d))
    # alg1 d=1
    ok(g(n, 1, 1) == 2, n)
    # d=2 closed form: H_dir = 2 N_n + 2n + 1 - (-1)^n
    ok(g(n, 1, 2) == 2 * N[n] + 2 * n + 1 - (-1) ** n, (n, "d2 closed"))
    if n % 2 == 0:
        # K_{k,k}
        dk = k if k % 2 else k - 1
        ok(g(n, 3, dk) == f(k) * f(k - 1) and g(n, 3, k) == f(k) * f(k - 1), (n, "Kkk"))
    else:
        m = k; t = (n + 3) // 4
        if t - 1 <= 3: ok(g(n, 3, m) == ham_complement(n, list(range(1, t))), (n, "odd complement formula"))
        ok(g(n, 5, n - 2) == f(n - 1), (n, "odd alg5 at n-2 complete"))
        for d in range(m + 1, n - 2):  # alg5 beyond kmax: offsets odd<=m plus even j>=n-d
            Sx = [j for j in range(1, m + 1) if j % 2 == 1 or j >= n - d]
            if len(Sx) == m: ok(g(n, 5, d) == f(n - 1), (n, 5, d))
            elif hc(n, Sx) is not None: ok(g(n, 5, d) == hc(n, Sx), (n, 5, d))
# recurrences reproduce the table
for key, S in (("1_2", [1, 2]), ("1_2_3", [1, 2, 3]), ("1_3", [1, 3]), ("1_2_3_4", [1, 2, 3, 4])):
    C = recs[key]; L = len(C) - 1; s = max(S)
    for n in range(6, 21):
        if n - L >= max(6, 2 * s + 1):
            ok(sum(C[i] * g(n - i, 1 if key != "1_3" else 3, s) for i in range(L + 1)) == 0, (key, n))
# anchors from the coordinator (symmetric counts = directed/2)
sym = lambda n, a, d: g(n, a, d) // 2
ok([sym(n, 1, 2) for n in (10, 12, 14)] == [56, 110, 225], "anchor d2")
ok([sym(n, 1, 3) for n in (10, 12, 14, 16, 18, 20)] == [3241, 13426, 56272, 235832, 991355, 4192327], "anchor d3")
ok([sym(n, 5, n // 2) for n in (8, 10, 12, 14, 16, 18, 20)] == [72, 1440, 43200, 1814400, 101606400, 7315660800, 658409472000], "Kkk anchors")
ok([g(n, 5, n // 2) for n in (7, 9, 11, 13, 15, 17)] == [46, 110, 14834, 76602, 22761926, 205661622], "odd anchors")
print("all", nchk, "checks passed")
