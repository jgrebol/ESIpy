"""Numerical probe of the lower bounds.
(a) Nisan partial-derivative (flattening) rank of P = sum_sigma x_s1...x_sm at every cut k: equals C(m,k).
(b) ALS search for a decomposition of the order-m tensor of P into s rank-one terms (= s products of m
    linear forms, a strictly larger class than Glynn's s powers of one form). s = 2^(m-1) must succeed;
    we look at s = 2^(m-1) - 1 and at the flattening bound C(m, m//2)."""
import numpy as np, sys, time
from itertools import permutations
from math import comb

def ptensor(m):
    T = np.zeros((m,) * m)
    for p in permutations(range(m)):
        T[p] = 1.0
    return T

def flatten_ranks(m):
    T = ptensor(m)
    return [np.linalg.matrix_rank(T.reshape(m ** k, m ** (m - k))) for k in range(m + 1)]

def als(T, s, iters, rng, ridge=1e-10):
    m = T.ndim; d = T.shape[0]
    F = [rng.standard_normal((d, s)) for _ in range(m)]
    nT = np.linalg.norm(T)
    for it in range(iters):
        for j in range(m):
            # Khatri-Rao of the other factors
            G = np.ones((s, s)); 
            for i in range(m):
                if i != j: G *= F[i].T @ F[i]
            Tj = np.moveaxis(T, j, 0).reshape(d, -1)
            others = [F[i] for i in range(m) if i != j]
            KR = others[0]
            for O in others[1:]:
                KR = (KR[:, None, :] * O[None, :, :]).reshape(-1, s)
            F[j] = np.linalg.solve(G + ridge * np.eye(s), (Tj @ KR).T).T
    KR = F[0]
    for O in F[1:]:
        KR = (KR[:, None, :] * O[None, :, :]).reshape(-1, s)
    return np.linalg.norm(KR.sum(1) - T.ravel()) / nT

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    for m in range(2, 8):
        fr = flatten_ranks(m)
        print('m=%d flattening ranks %s  binomials %s  2^(m-1)=%d' % (m, fr, [comb(m, k) for k in range(m + 1)], 2 ** (m - 1)))
    for m, ss, restarts, iters in [(3, (3, 4), 20, 800), (4, (6, 7, 8), 20, 1500), (5, (10, 15, 16), 4, 600)]:
        T = ptensor(m)
        for s in ss:
            t = time.time()
            res = [als(T, s, iters, rng) for _ in range(restarts)]
            print('m=%d s=%2d  best rel. residual over %d ALS restarts: %.3e (median %.2e)  %.1fs' % (m, s, restarts, min(res), np.median(res), time.time() - t))
            sys.stdout.flush()
