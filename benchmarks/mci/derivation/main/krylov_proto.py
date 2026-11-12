import time, itertools, numpy as np
from esipy import mci

def sign_table(mats):
    """All 2^k signed sums +-M_1 +- ... +-M_k and their sign products (doubling: one addition per entry)."""
    sums = np.zeros((1,) + mats[0].shape); par = np.ones(1)
    for M in mats:
        sums = np.concatenate([sums + M, sums - M]); par = np.concatenate([par, -par])
    return sums, par

def krylov_exact(mats, sym, nlow=8):
    """sum_delta prod(delta) Tr(S_0 X^m) / 2^(m-1) with matrix-vector products seeded by S_0 = L R^T."""
    m = len(mats) - 1
    U, s, Vt = np.linalg.svd(mats[0]); k = int((s > s[0] * len(s) * 2.2e-16).sum())
    L, R = U[:, :k] * s[:k], Vt[:k].T
    w, Us = np.linalg.eigh((mats[0] + mats[0].T) / 2); keep = abs(w) > abs(w).max() * len(w) * 2.2e-16; w, Us = w[keep], Us[:, keep]
    rest = mats[2:]; nlow = min(nlow, len(rest))
    hi, hpar = sign_table(rest[nlow:]) if len(rest) > nlow else (np.zeros((1,) + mats[0].shape), np.ones(1))
    lo, lpar = sign_table(rest[:nlow]) if nlow else (np.zeros((1,) + mats[0].shape), np.ones(1))
    lo = lo + mats[1]
    total = 0.0
    for H, hp in zip(hi, hpar):
        X = lo + H                                   # (B, q, q)
        if sym:                                      # S_0 = sum_a w_a u_a u_a^T: u^T X^m u = (X^h u).(X^(m-h) u)
            h = m // 2
            V = np.ascontiguousarray(np.broadcast_to(Us, (len(X),) + Us.shape))
            for _ in range(h): V = X @ V
            W = X @ V if m - h > h else V
            tr = np.einsum('bqa,bqa,a->b', V, W, w)
        else:
            V = np.broadcast_to(L, (len(X),) + L.shape)
            for _ in range(m): V = X @ V
            tr = np.einsum('qa,bqa->b', R, V)
        total += hp * (lpar @ tr)
    return total / 2 ** (m - 1)

rng = np.random.default_rng(0)
def rand(n, q, r, sym):
    out = []
    for _ in range(n):
        a = rng.normal(size=(q, r)) / np.sqrt(q)
        out.append(a @ a.T if sym else a @ (rng.normal(size=(r, q)) / np.sqrt(q)))
    return out

for sym in (True, False):
    mats = rand(7, 9, 3, sym)
    ref = sum(np.trace(np.linalg.multi_dot([mats[0]] + [mats[i] for i in p])) for p in itertools.permutations(range(1, 7)))
    print("sym", sym, "rel err vs brute force", abs(krylov_exact(mats, sym, 2) - ref) / abs(ref), abs(krylov_exact(mats, False, 3) - ref) / abs(ref))

for n, q, r in ((14, 49, 5), (14, 49, 14), (16, 56, 5), (18, 63, 5)):
    mats = rand(n, q, r, True)
    t = time.time(); a, _ = mci._glynn_sum(mats, None, 1, 64); ta = time.time() - t
    t = time.time(); b = krylov_exact(mats, True); tb = time.time() - t
    t = time.time(); c = krylov_exact(mats, False); tc = time.time() - t
    print(f"n={n} q={q} r={r}: glynn {ta:.3f}s  krylov-sym {tb:.3f}s ({ta/tb:.1f}x)  krylov-nonsym {tc:.3f}s ({ta/tc:.1f}x)  rel diff {abs(a-b)/abs(a):.1e} {abs(a-c)/abs(a):.1e}", flush=True)
