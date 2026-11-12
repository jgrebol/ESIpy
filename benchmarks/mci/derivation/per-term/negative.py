import numpy as np
from perterm import *
rng = np.random.default_rng(3)

def sym_nullity(mats):
    """dim of {M = M^T : M S_i = S_i^T M for all i}; a symmetrising similarity T gives M = T^T T > 0 in it."""
    q = mats[0].shape[0]
    idx = [(i, j) for i in range(q) for j in range(i, q)]
    rows = []
    for S in mats:
        cols = []
        for (i, j) in idx:
            M = np.zeros((q, q)); M[i, j] = M[j, i] = 1
            cols.append((M @ S - S.T @ M).ravel())
        rows.append(np.array(cols).T)
    sv = np.linalg.svd(np.vstack(rows), compute_uv=False)
    return int((sv < sv[0] * 1e-10).sum()), sv[-1] / sv[0]

# synthetic Mulliken AOMs: S_A = C^T Pi_A S C, C^T S C = I, sum_A S_A = I
N, q, nat = 12, 6, 4
B = rng.standard_normal((N, N)); S = B @ B.T / N + np.eye(N)
C = rng.standard_normal((N, q)); w, U = np.linalg.eigh(C.T @ S @ C); C = C @ U / np.sqrt(w)
mull = []
for A in range(nat):
    Pi = np.zeros((N, N)); sl = slice(3 * A, 3 * A + 3); Pi[sl, sl] = np.eye(3)
    mull.append(C.T @ Pi @ S @ C)
print('sum of Mulliken AOMs = I:', np.abs(sum(mull) - np.eye(q)).max())
a, b, c = mull[:3]
t1, t2 = np.trace(a @ b @ c), np.trace(c @ b @ a)
print('Mulliken Tr(ABC) =', t1, ' Tr(CBA) =', t2, ' rel diff', abs(t1 - t2) / abs(t1))
print('Mulliken: dim of symmetriser space, smallest sv ratio:', sym_nullity(mull))
# natural orbitals
A_ = []
for _ in range(nat):
    Bq = rng.standard_normal((q, 3)); A_.append(Bq @ Bq.T)
occ = rng.uniform(0.05, 2, q)
no = [np.sqrt(occ)[:, None] * x for x in A_]
print('NO: dim of symmetriser space:', sym_nullity(no))
a, b, c = no[:3]
print('NO Tr(ABC) - Tr(CBA):', np.trace(a @ b @ c) - np.trace(c @ b @ a))

# Gray-code update of the Krylov block: rank of X'^j L0 - X^j L0 after flipping one sign
q, r, m = 30, 3, 9
mats = []
for _ in range(m + 1):
    Bq = rng.standard_normal((q, r)); mats.append(Bq @ Bq.T / q)
L0 = factor(mats[0], True)[0]
X = sum(mats[1:]); X2 = X - 2 * mats[4]
V, V2 = L0.copy(), L0.copy()
for j in range(1, 6):
    V, V2 = X @ V, X2 @ V2
    print('j=%d rank(V_j) = %d, rank(V_j - V_j_flipped) = %d, rank(X^j - X2^j) = %d' % (
        j, np.linalg.matrix_rank(V), np.linalg.matrix_rank(V - V2), np.linalg.matrix_rank(np.linalg.matrix_power(X, j) - np.linalg.matrix_power(X2, j))))
