import numpy as np, itertools
from lib import *
rng = np.random.default_rng(4)
LD = np.longdouble

def terms(mats, kind, dtype=float):
    """signed terms; Glynn: 2^(m-1) terms /2^(m-1); Ryser: 2^m subset terms (-1)^(m-|S|) Tr(S0 X_S^m)."""
    m, q = len(mats) - 1, len(mats[0])
    rest = np.array(mats[1:], dtype=dtype).reshape(m, -1); S0 = np.array(mats[0], dtype=dtype)
    out = []
    N = 1 << (m - 1) if kind == 'glynn' else 1 << m
    for a in range(0, N, 512):
        idx = np.arange(a, min(a + 512, N))
        if kind == 'glynn':
            bits = (idx[:, None] >> np.arange(m - 1)) & 1
            c = np.ones((len(idx), m), dtype=dtype); c[:, 1:] -= 2 * bits; sg = c.prod(axis=1) / N
        else:
            c = ((idx[:, None] >> np.arange(m)) & 1).astype(dtype); sg = (-1.0) ** (m - c.sum(axis=1))
        X = (c @ rest).reshape(-1, q, q)
        V = np.broadcast_to(S0, X.shape)
        for _ in range(m): V = np.matmul(V, X)
        out.append(sg * np.trace(V, axis1=1, axis2=2))
    return np.concatenate(out)

def ryser_c(mats, c):
    """shifted (Glynn/Ryser family) X = sum (b_i - c) S_i, b in {0,1}; c=0 Ryser, c=1/2 Glynn (2^m terms)"""
    pass

print("brute-force validation of the Ryser form and Glynn form (n<=8)")
w = 0
for sym in (False, True):
    for n in (4, 6, 8):
        mats = rand_mats(n, 6, 2, sym, rng); ref = brute(mats)
        w = max(w, abs(terms(mats, 'glynn').sum() - ref) / abs(ref), abs(terms(mats, 'ryser').sum() - ref) / abs(ref))
print("max rel err", w)

print("\ncondition number kappa = sum|t_k| / |sum t_k| and measured float64 error vs longdouble reference")
print("model            n   R              kappa_Glynn  err_Glynn  kappa_Ryser  err_Ryser")
for name in ('hueckel', 'partition'):
    for n in (6, 8, 10, 12, 14):
        if name == 'hueckel': mats = huckel(n)
        else: mats = partition_of_identity(n + 6, 10, 2, rng)[:n]
        ref = terms(mats, 'glynn', LD).sum()
        tg = terms(mats, 'glynn'); tr = terms(mats, 'ryser')
        print(f"{name:10s} {n:7d}  {float(ref):+.6e}  {np.abs(tg).sum()/abs(tg.sum()):10.2e}  {abs(tg.sum()-ref)/abs(ref):9.1e}  {np.abs(tr).sum()/abs(tr.sum()):10.2e}  {abs(tr.sum()-ref)/abs(ref):9.1e}")

# roots of unity instead of signs: k-th roots, k=3: 3^(m-1) terms
def glynn_roots(mats, k):
    m, q = len(mats) - 1, len(mats[0]); w = np.exp(2j * np.pi / k); tot = 0
    for e in itertools.product(range(k), repeat=m - 1):
        c = np.array([1] + [w ** x for x in e])
        X = np.tensordot(c, np.array(mats[1:]), 1)
        tot += np.conj(c).prod() * np.trace(mats[0] @ np.linalg.matrix_power(X, m))
    return (tot / k ** (m - 1)).real
print("\nroots of unity (k=3,4) in place of signs: exact but k^(m-1) terms, complex arithmetic")
w = 0
for sym in (False, True):
    for n, k in [(5, 3), (6, 3), (7, 3), (6, 4), (8, 3)]:
        mats = rand_mats(n, 6, 2, sym, rng); ref = brute(mats)
        e = abs(glynn_roots(mats, k) - ref) / abs(ref); w = max(w, e)
print("max rel err", w)
