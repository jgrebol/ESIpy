import numpy as np

def glynn_krylov(mats, tbits=6):
    """sum_sigma Tr(S_0 S_s1..S_sm); symmetric input -> ceil(m/2) block mat-vecs per term, else m."""
    sym = all(np.allclose(M, M.T, atol=1e-14 * np.abs(M).max()) for M in mats)
    f = int(np.argmin([np.linalg.matrix_rank(M) for M in mats]))      # lowest-rank atom fixed
    S0, rest = mats[f], np.array([M for i, M in enumerate(mats) if i != f])
    m, k = len(rest), S0.shape[0]
    if sym:
        w, U = np.linalg.eigh((S0 + S0.T) / 2); keep = np.abs(w) > np.abs(w).max() * k * 2e-16
        L0, sig, R0 = U[:, keep], w[keep], None
    else:
        U, s, Vt = np.linalg.svd(S0); r = int((s > s[0] * k * 2e-16).sum())
        L0, sig, R0 = U[:, :r], s[:r], Vt[:r].T
    t = min(tbits, m - 1)
    dl = 1.0 - 2 * ((np.arange(1 << t)[:, None] >> np.arange(t)) & 1)
    Q = rest[0] + np.tensordot(dl, rest[1:t + 1], axes=1)             # table of the low-sign sums
    sl, hi = dl.prod(axis=1), rest[t + 1:]
    H, dh, sh, total = hi.sum(axis=0), np.ones(len(hi)), 1.0, 0.0
    for g in range(1 << len(hi)):
        if g:                                                          # Gray code: one axpy
            j = (g & -g).bit_length() - 1
            dh[j] = -dh[j]; H = H + 2 * dh[j] * hi[j]; sh = -sh
        X = H + Q
        V = np.broadcast_to(L0, (len(Q),) + L0.shape)
        if sym:
            for _ in range(m // 2): V = X @ V
            W = X @ V if m % 2 else V
            tr = np.einsum('bka,a,bka->b', V, sig, W)
        else:
            for _ in range(m): V = X @ V
            tr = np.einsum('ka,a,bka->b', R0, sig, V)
        total += sh * (sl @ tr)
    return total / 2 ** (m - 1)

if __name__ == '__main__':
    from perterm import brute
    rng = np.random.default_rng(1); e = 0
    for n in range(3, 10):
        for sym in (True, False):
            mats = []
            for _ in range(n):
                B = rng.standard_normal((8, 2))
                mats.append(B @ B.T if sym else B @ rng.standard_normal((2, 8)))
            ref = brute(mats); e = max(e, abs(glynn_krylov(mats) - ref) / abs(ref))
            e = max(e, abs(glynn_krylov(mats, tbits=1) - ref) / abs(ref))
    print('max rel err', e)
