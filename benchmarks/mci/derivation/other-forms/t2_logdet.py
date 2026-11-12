import numpy as np, time, itertools
from lib import *
rng = np.random.default_rng(3)

def nlogdet(M):            # -log det(I - M), complex principal branch
    s, l = np.linalg.slogdet(np.eye(len(M)) - M)
    return -(np.log(s + 0j) + l)

def R_logdet_filter(mats, K, rho):
    """R = 2^-m sum_{eps, eps_0=+1} prod(eps) [z^n](-log det(I - z Y_eps)), [z^n] by K-point root-of-unity filter."""
    n = len(mats); m = n - 1; q = len(mats[0]); tot = 0.0
    z = rho * np.exp(2j * np.pi * np.arange(K) / K)
    for bits in range(1 << m):
        e = np.array([1.0] + [1 - 2 * ((bits >> i) & 1) for i in range(m)])
        Y = np.tensordot(e, np.array(mats), 1)
        g = np.mean([nlogdet(zk * Y) * zk ** (-n) for zk in z]).real
        tot += e.prod() * g
    return tot / (1 << m)

def R_logdet_fd(mats, h):
    """plain finite difference: 2^-n h^-n sum_{eps in +-1^n} prod(eps) f(h eps), error O(h^2) + eps_mach/h^n"""
    n = len(mats); tot = 0.0
    for bits in range(1 << n):
        e = np.array([1 - 2 * ((bits >> i) & 1) for i in range(n)], float)
        tot += e.prod() * nlogdet(h * np.tensordot(e, np.array(mats), 1)).real
    return tot / (1 << n) / h ** n

def R_trace_n(mats):
    """same sign sum with the exact degree-n part Tr(Y^n)/n  (proves the multilinear-coefficient statement)"""
    n = len(mats); m = n - 1; tot = 0.0
    for bits in range(1 << m):
        e = np.array([1.0] + [1 - 2 * ((bits >> i) & 1) for i in range(m)])
        Y = np.tensordot(e, np.array(mats), 1)
        tot += e.prod() * np.trace(np.linalg.matrix_power(Y, n)) / n
    return tot / (1 << m)

print("identity check  R = [t0..tm] -log det(I - sum t_i S_i)")
w1 = w2 = 0
for sym in (False, True):
    for n, q, r in [(4,6,2),(5,7,3),(6,8,2),(7,9,3),(8,8,2),(9,6,2)]:
        mats = rand_mats(n, q, r, sym, rng); ref = brute(mats)
        nrm = n * 0.6
        K = 2 * n + 2 if n < 9 else n + 3
        a = abs(R_trace_n(mats) - ref) / abs(ref)
        b = abs(R_logdet_filter(mats, K, 0.5 / nrm) - ref) / abs(ref)
        w1 = max(w1, a); w2 = max(w2, b)
        print(f"sym={sym} n={n}: Tr(Y^n)/n sign-sum err {a:.1e};  logdet+filter(K={K}, rho*|Y|<=0.5) err {b:.1e}")
print("MAX: trace form", w1, " logdet filter", w2)
print("\nplain finite differences of log det (n=5, n=7):")
for n in (5, 7):
    mats = rand_mats(n, 7, 2, True, rng); ref = brute(mats)
    print(n, [(h, f"{abs(R_logdet_fd(mats, h) - ref)/abs(ref):.1e}") for h in (0.3, 0.1, 0.03, 0.01, 0.003)])
print("\nfilter radius trade-off n=8 (K=18):")
mats = rand_mats(8, 8, 2, True, rng); ref = brute(mats)
for x in (0.9, 0.5, 0.2, 0.05):
    print("rho*|Y|max =", x, f"err {abs(R_logdet_filter(mats, 18, x/(8*0.6)) - ref)/abs(ref):.1e}")
# cost of one evaluation
q = 49; Y = rng.standard_normal((q, q)); Y = (Y + Y.T) / 20; U0 = rng.standard_normal((q, 5))
def tm(f, k=300):
    t = time.perf_counter()
    for _ in range(k): f()
    return (time.perf_counter() - t) / k * 1e6
Yc = Y.astype(complex)
print("\nmicroseconds per term, q=49, m=13:")
print(" one complex slogdet            ", tm(lambda: np.linalg.slogdet(np.eye(q) - 0.3j * Yc)), " (needs K >= n+1 = 15 of them per term)")
print(" eigvalsh (spectrum, sym only)  ", tm(lambda: np.linalg.eigvalsh(Y)))
print(" eigvals (nonsym)               ", tm(lambda: np.linalg.eigvals(Y)))
print(" repeated squaring X^13 (5 prod)", tm(lambda: np.linalg.matrix_power(Y, 13)))
def kry():
    V = U0
    for _ in range(6): V = Y @ V
    return np.sum(V * (Y @ V))
print(" block Krylov, r0=5, 7 matvecs  ", tm(kry))
