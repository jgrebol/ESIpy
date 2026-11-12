from tmat import *; from tm import ham_count; import time, sys
S = eval(sys.argv[1]); s = max(S)
t = time.time(); keys, trans, v0, w = build(S); print(S, "states", len(keys), "nnz", len(trans), "%.1fs" % (time.time() - t))
ex, D = sequence_exact(S, 3 * s + 12)
for n in range(3 * s, 3 * s + 12): assert ex[n] == ham_count(n, S), n
print("exact TM matches frontier DP")
P1, P2 = 2147483629, 2147483587
N = 3 * s + 2 * D + 40
a1, _ = sequence_mod(S, N, P1); a2, _ = sequence_mod(S, N, P2)
C1, L1 = bm_mod([a1[n] for n in sorted(a1)], P1); C2, L2 = bm_mod([a2[n] for n in sorted(a2)], P2)
print("order", L1, L2, "same", C1 == C2, "terms", len(a1), "time %.1f" % (time.time() - t))
print("C =", C1)
