from lib import *
import time
# 1) HK vs DFS mirror of the code, all algs, small n
bad = 0
for n in range(5, 11):
    c = ring_connec(n)
    for alg in (1, 2, 3, 4, 5):
        for d in range(1, n // 2 + 1):
            a, b = count(c, alg, d), dfs_count(c, alg, d)
            if a != b: bad += 1; print("MISMATCH", n, alg, d, a, b)
print("mismatches:", bad)
# anchors
print([count(ring_connec(n), 1, 2) // 2 for n in (10, 12, 14)], "expect 56 110 225")
print([count(ring_connec(n), 5, n // 2) for n in (7, 9, 11, 13)], "expect 46 110 14834 76602")
t = time.time(); print(count(ring_connec(16), 3, 8), factorial(8) * factorial(7), time.time() - t)
