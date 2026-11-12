"""Numeric evidence for the lower bounds (numpy only, tiny problems).
 (1) Waring rank of x1..xm: least-squares fit of  m! x1..xm = sum_k c_k (a_k.x)^m  with s terms.
 (2) tensor rank of the permanent tensor (sum of s products of m DIFFERENT linear forms), flattening ranks."""
import itertools, numpy as np
from math import factorial, comb
import sys; rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 3)

def lm(res, x, iters=400):
    lam = 1e-2; r = res(x); f = r @ r
    for _ in range(iters):
        J = np.empty((len(r), len(x)))
        for i in range(len(x)):
            h = 1e-6 * max(1, abs(x[i])); xp = x.copy(); xp[i] += h; xm = x.copy(); xm[i] -= h
            J[:, i] = (res(xp) - res(xm)) / (2 * h)
        g = J.T @ r; H = J.T @ J
        for _ in range(12):
            try: dx = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), -g)
            except np.linalg.LinAlgError: lam *= 10; continue
            r2 = res(x + dx); f2 = r2 @ r2
            if f2 < f: x, r, f, lam = x + dx, r2, f2, max(lam / 5, 1e-12); break
            lam *= 5
        if f < 1e-24: break
    return x, np.sqrt(f)

def waring(m, s, starts):
    monos = [c for c in itertools.combinations_with_replacement(range(m), m)]
    pts = rng.standard_normal((2 * len(monos), m))
    target = factorial(m) * pts.prod(axis=1)
    def res(x):
        c = x[:s]; a = x[s:].reshape(s, m)
        return ((pts @ a.T) ** m) @ c - target
    best = np.inf
    for _ in range(starts):
        x, f = lm(res, rng.standard_normal(s + s * m)); best = min(best, f / np.linalg.norm(target))
    return best

def perm_tensor(m):
    T = np.zeros((m,) * m)
    for p in itertools.permutations(range(m)): T[p] = 1
    return T

def tensor_fit(m, s, starts, iters=400):
    T = perm_tensor(m).ravel(); nT = np.linalg.norm(T)
    letters = 'abcdefg'[:m]
    sub = ','.join('k' + l for l in letters) + '->' + letters
    def res(x):
        U = x.reshape(m, s, m)
        return np.einsum(sub, *U).ravel() - T
    best = np.inf
    for _ in range(starts):
        x, f = lm(res, rng.standard_normal(m * s * m) * 0.7, iters); best = min(best, f / nT)
    return best

if __name__ == '__main__':
    import sys
    part = sys.argv[1]
    if part == 'flat':
        for m in range(2, 9):
            j = m // 2
            # flattening of the permanent tensor (rows: words of length j, cols: words of length m-j)
            if m <= 7:
                T = perm_tensor(m).reshape(m ** j, m ** (m - j)); rk = np.linalg.matrix_rank(T)
            else: rk = '-'
            print(f"m={m}: flattening rank = {rk}, C(m,m//2) = {comb(m, j)}, Glynn 2^(m-1) = {2 ** (m - 1)}, ratio = {2 ** (m - 1) / comb(m, j):.2f}")
        # dim of S/(y1, y2^2..ym^2) = 2^(m-1): trivially the square-free monomials in y2..ym
    if part == 'waring':
        for m, s, st in ((3, 3, 12), (3, 4, 6), (4, 7, 6), (4, 8, 6)):
            print(f"Waring fit m={m}, s={s} terms (2^(m-1)={2 ** (m - 1)}): best relative residual over {st} starts = {waring(m, s, st):.2e}", flush=True)
    if part == 'tensor3':
        for m, s, st in ((3, 3, 12), (3, 4, 8)):
            print(f"perm tensor m={m}, rank-{s} fit: best relative residual over {st} starts = {tensor_fit(m, s, st):.2e}", flush=True)
    if part == 'tensor4':
        s = int(sys.argv[2]); st = int(sys.argv[3])
        print(f"perm tensor m=4, rank-{s} fit: best relative residual over {st} starts = {tensor_fit(4, s, st, 250):.2e}", flush=True)
