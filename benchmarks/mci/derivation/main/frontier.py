"""Prototype: merged-path (frontier) sum over the Hamiltonian cycles allowed by an adjacency matrix."""
import sys, time, pickle
import numpy as np
from esipy import mci


def frontier_sum(mats, adj, prune=True, max_mb=200):
    """Sum of Tr(S_0 S_s1...S_sm) over the directed cycles allowed by adj, their number and the largest layer."""
    mats = mci._project(mats)
    m = len(mats) - 1
    (L0, R0), L, R = mci._factors(mats)
    T = np.einsum('uka,vkb->uvab', R, L)                 # T[u, v] = R_u^T L_v
    Y0 = np.matmul(R0.T, L)                              # (m, r0, r)
    E = np.matmul(R.transpose(0, 2, 1), L0)              # (m, r, r0)
    A = adj[1:, 1:].astype(bool)
    first, last0 = adj[0, 1:].astype(bool), adj[1:, 0].astype(bool)
    nbrs = [np.flatnonzero(A[w]) for w in range(m)]
    bit = np.int64(1) << np.arange(m, dtype=np.int64)

    v = np.flatnonzero(first)
    mask, last, Y, cnt = bit[v], v, Y0[v], np.ones(len(v))
    peak = len(v)
    for depth in range(1, m):
        ok = A[last] & ((mask[:, None] & bit) == 0)      # allowed steps to an unused atom
        if ok.sum() * 8 * (3 * Y[0].size + 4) > max_mb * 1e6:   # candidates would not fit in the budget
            raise MemoryError(int(ok.sum()))
        s, v = np.nonzero(ok)
        nmask = mask[s] | bit[v]
        if prune:                                        # an unused atom needs a way in and a way out
            alive = np.ones(len(s), bool)
            for w in range(m):
                free = np.zeros(len(s), np.int64)
                for x in nbrs[w]:
                    free += ((nmask & bit[x]) == 0) | (v == x)
                alive &= ((nmask & bit[w]) != 0) | (free + last0[w] >= 2)
            s, v, nmask = s[alive], v[alive], nmask[alive]
        if len(s) == 0:
            return 0.0, 0, peak
        newY = np.matmul(Y[s], T[last[s], v])
        key = nmask * 64 + v
        uniq, inv = np.unique(key, return_inverse=True)  # merge paths with the same used set and last atom
        Y = np.zeros((len(uniq),) + newY.shape[1:]); np.add.at(Y, inv, newY)
        cnt = np.bincount(inv, weights=cnt[s], minlength=len(uniq))
        mask, last = uniq >> 6, (uniq & 63).astype(np.int64)
        peak = max(peak, len(uniq))
    close = last0[last]
    total = np.einsum('sab,sba->', Y[close], E[last[close]])
    return total, int(round(cnt[close].sum())), peak


if __name__ == "__main__":
    S = pickle.load(open("benchmarks/mci/cache/systems.pkl", "rb"))
    part = "meta-lowdin"
    for n in (10, 12, 14, 18, 24):
        s = S[f"annulene{n}"]; ring = s["rings"]["perimeter"]; aom = s["parts"][part]["aom"]
        mats = mci._prep_matrices(ring, aom)
        idx = np.arange(n); D = np.abs(idx[:, None] - idx[None]); D = np.minimum(D, n - D)
        exact = mci.compute_mci_glynn(ring, aom, part, n_cores=1) if n <= 18 else None
        for alg, ds in ((1, (1, 2, 3, 4, 5)), (5, (1, 3, 5, 7))):
            for d in ds:
                if d > n // 2: continue
                adj = (D <= d) & (D > 0) & ((D % 2 == 1) if alg == 5 else True)
                t = time.time()
                try:
                    tot, cnt, peak = frontier_sum(mats, adj)
                except MemoryError as e:
                    print(f"n={n} alg={alg} d={d}: over the memory budget ({e} candidate states), stop", flush=True); break
                tf = time.time() - t
                val = 2 ** (n - 1) * 0.5 * tot
                line = f"n={n} alg={alg} d={d}: frontier {tf:7.3f}s  perms {cnt:>14}  peak states {peak:>8}  MCI {val:+.6e}"
                if exact is not None:
                    line += f"  ({100 * val / exact:7.2f}% of exact)"
                if n <= 14:
                    t = time.time(); g, p = mci._glynn_sum(mats, adj, 1, 64); tg = time.time() - t
                    line += f" | glynn {tg:6.3f}s rel diff {abs(g - tot) / max(abs(g), 1e-300):.1e} perms equal {p == cnt}"
                print(line, flush=True)
