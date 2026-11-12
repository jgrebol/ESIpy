"""Growth constants lambda_d and prefactors C_d (H_dir(n) ~ C_d lambda_d^n), from certified recurrences."""
import json, mpmath, sympy
from tm import ham_count
from tmat import sequence_exact
mpmath.mp.dps = 40
x = sympy.symbols('x')
out = {}
for S in ([1, 2], [1, 2, 3], [1, 3], [1, 2, 3, 4]):
    r = json.load(open("rec_%s.json" % "_".join(map(str, S)))); C = r['C']; L = len(C) - 1; s = max(S)
    n0 = 3 * s
    ex, _ = sequence_exact(S, n0 + L + 6)
    init = {n: ex[n] for n in range(n0, n0 + L)}
    a = dict(init)
    for n in range(n0 + L, 401): a[n] = -sum(C[i] * a[n - i] for i in range(1, L + 1))
    # check recurrence reproduces frontier DP beyond the initial window
    for n in range(n0 + L, n0 + L + 6): assert a[n] == ex[n], (S, n)
    p = sympy.Poly(sum(c * x ** (L - i) for i, c in enumerate(C)), x)
    roots = []
    for fac, mlt in sympy.factor_list(p)[1]:
        roots += sympy.Poly(fac, x).nroots(n=40, maxsteps=500)
    lam = max(roots, key=lambda z: abs(z)); lam = mpmath.mpf(str(sympy.re(lam)))
    same_mod = [complex(z) for z in roots if abs(abs(complex(z)) - float(lam)) < 1e-9]
    ce = mpmath.mpf(a[400]) / lam ** 400 if True else None
    ce2 = mpmath.mpf(a[399]) / lam ** 399
    print(S, "lambda = %s" % mpmath.nstr(lam, 15), "roots on |z|=lambda:", same_mod,
          "C(even n) = %s  C(odd n) = %s" % (mpmath.nstr(ce, 12), mpmath.nstr(ce2, 12)))
    out[str(S)] = {"lambda": float(lam), "C_even": float(ce), "C_odd": float(ce2), "seq": {n: a[n] for n in range(n0, 81)}}
json.dump(out, open("growth.json", "w"), indent=0)
