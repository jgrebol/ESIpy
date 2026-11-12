import numpy as np
from lib import factor, deltas

def glynn_krylov2(mats, sym=False, low=6, s0=0, s1=None):
    """Glynn + block Krylov + two-level sign table (constant memory 2^low q^2).
    Terms s in [s0, s1) (multiples of 2^low); returns the un-normalised sum (divide by 2^(m-1))."""
    m, q = len(mats) - 1, mats[0].shape[0]
    L0, R0 = factor(mats[0], sym)
    rest = np.array(mats[1:]).reshape(m, -1)
    low = min(low, m - 1); B = 1 << low
    # table over delta_1=+1 and the first `low` free signs
    bits = (np.arange(B)[:, None] >> np.arange(low)) & 1
    dl = np.ones((B, low + 1)); dl[:, 1:] -= 2 * bits
    T = (dl @ rest[:low + 1]).reshape(B, q, q); sl = dl.prod(axis=1)
    hi = rest[low + 1:]; nh = m - 1 - low
    s1 = (1 << (m - 1)) if s1 is None else s1
    V0 = np.broadcast_to(L0, (B,) + L0.shape)
    tot = 0.0
    for h in range(s0 >> low, s1 >> low):
        dh = 1.0 - 2 * ((h >> np.arange(nh)) & 1)
        X = T + (dh @ hi).reshape(q, q) if nh else T
        V = V0
        if sym:
            for _ in range(m // 2): V = X @ V
            W = X @ V if m & 1 else V
            t = np.einsum('bij,bij->b', V, W)
        else:
            for _ in range(m): V = X @ V
            t = np.einsum('ij,bij->b', R0, V)
        tot += dh.prod() * (sl @ t)
    return tot
