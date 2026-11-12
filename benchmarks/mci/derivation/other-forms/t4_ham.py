import numpy as np, itertools
from lib import *
rng = np.random.default_rng(5)

def hc_poly(G):
    """Hamiltonian-cycle polynomial: sum over sigma of G[0,s1] G[s1,s2] ... G[sm,0]"""
    n = len(G); tot = 0.0
    for p in itertools.permutations(range(1, n)):
        c = (0,) + p + (0,); x = 1.0
        for a, b in zip(c[:-1], c[1:]): x *= G[a, b]
        tot += x
    return tot

print("rank-1 AOMs S_i = u_i v_i^T  =>  R = HC(G), G_ij = v_i . u_j")
w = 0
for n in (4, 5, 6, 7, 8):
    q = n + 2
    U = rng.standard_normal((n, q)); V = rng.standard_normal((n, q))
    mats = [np.outer(U[i], V[i]) for i in range(n)]
    G = V @ U.T
    w = max(w, abs(brute(mats) - hc_poly(G)) / abs(hc_poly(G)))
    mats = [np.outer(U[i], U[i]) for i in range(n)]; G = U @ U.T
    w = max(w, abs(brute(mats) - hc_poly(G)) / abs(hc_poly(G)))
print("max rel err", w)

print("\nreduction: #Hamiltonian cycles of a graph from R of physical AOMs (sym PSD, rank 1, sum_ring S_i <= I, completed to I)")
w = 0
for n in (5, 6, 7, 8):
    for trial in range(3):
        A = np.triu(rng.random((n, n)) < 0.6, 1).astype(float); A = A + A.T
        lam = np.linalg.eigvalsh(A)
        M = A + (1e-9 - lam[0]) * np.eye(n)            # PSD, same off-diagonal
        M /= np.linalg.eigvalsh(M)[-1]; s = 1 / (lam[-1] - lam[0] + 1e-9)   # M = s (A + c I)
        Lc = np.linalg.cholesky(M + 1e-15 * np.eye(n))  # rows u_i, u_i.u_j = M_ij ; q = n
        mats = [np.outer(Lc[i], Lc[i]) for i in range(n)]
        ring = sum(mats); ev = np.linalg.eigvalsh(ring)
        assert ev[-1] <= 1 + 1e-9          # the rest I - ring is PSD: a legal 'rest of the molecule'
        R = glynn_krylov(mats, True)
        nham_directed = hc_poly(A)                      # = 2 * undirected Hamiltonian cycles
        est = R / s ** n
        w = max(w, abs(est - nham_directed))
        print(f"n={n}: directed Hamiltonian cycles {nham_directed:.0f}, recovered from R: {est:.6f}  (max eig of ring sum {ev[-1]:.3f})")
print("max abs err in the count", w)

print("\nsum rule from sum_k S_k = I:  sum_{k in all atoms} R(S_0..S_{m-1}, S_k) = m * R(S_0..S_{m-1})")
w = 0
for n in (4, 5, 6, 7):
    allm = partition_of_identity(n + 3, 8, 2, rng)
    ring = allm[:n - 1]; m = n - 1
    lhs = sum(brute(ring + [Sk]) for Sk in allm)
    rhs = m * brute(ring)
    w = max(w, abs(lhs - rhs) / abs(rhs))
    # non symmetric version: random matrices + remainder
    ms = rand_mats(n + 2, 6, 2, False, rng); ms.append(np.eye(6) - sum(ms))
    lhs = sum(brute(ms[:n - 1] + [Sk]) for Sk in ms); rhs = m * brute(ms[:n - 1])
    w = max(w, abs(lhs - rhs) / abs(rhs))
print("max rel err", w)
